"""Mode handlers — pure functions over a ZImagePipeline + params dict."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from PIL import Image

import lora
import preprocessors
import upscale

try:
    from diffsynth.diffusion.base_pipeline import ControlNetInput
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class ControlNetInput:  # type: ignore[no-redef]
        image: Any
        scale: float = 1.0


class T2IParams(TypedDict, total=False):
    prompt: str
    negative_prompt: str
    model: str  # "Base" | "Turbo"
    steps: int
    cfg: float
    width: int
    height: int
    seed: int
    lora_path: Path | None
    lora_strength: float


def _swap_transformer(pipe: Any, model_name: str) -> None:
    """Swap the active transformer between Base (index 0) and Turbo (index 1).

    ``backend._build_pipeline`` loads both transformers into ``pipe._zis_pool``
    and stores them under the same name ``z_image_dit``. DiffSynth's
    ``ModelPool.fetch_model`` doesn't expose a variant kwarg — both entries
    share the same name — so we index into ``pool.model`` directly. MODEL_CONFIGS
    loads Base first, then Turbo (so index 0 = Base, index 1 = Turbo).

    No-op if the pool is unavailable (e.g. mocked tests) or only one transformer
    was loaded.
    """
    variant = "z_image" if model_name == "Base" else "z_image_turbo"
    pool = getattr(pipe, "_zis_pool", None)
    if pool is not None:
        dits = [m for m, n in zip(pool.model, pool.model_name, strict=False) if n == "z_image_dit"]
        if len(dits) >= 2:
            pipe.dit = dits[0 if model_name == "Base" else 1]
    try:
        pipe.dit._zis_variant = variant
    except (AttributeError, RuntimeError):
        pass


def call_t2i(pipe: Any, params: T2IParams) -> tuple[Image.Image, dict[str, Any]]:
    """Text-to-image. Routes to base (cfg=4, 25 steps) or turbo (cfg=1, 8 steps)."""
    model_name = params.get("model", "Turbo")
    is_base = model_name == "Base"
    _swap_transformer(pipe, model_name)

    kwargs: dict[str, Any] = dict(
        prompt=params["prompt"],
        cfg_scale=float(params.get("cfg", 4.0 if is_base else 1.0)),
        num_inference_steps=int(params.get("steps", 25 if is_base else 8)),
        sigma_shift=3.0,
        height=int(params.get("height", 1024)),
        width=int(params.get("width", 1024)),
        seed=int(params.get("seed", 0)),
    )
    if is_base and params.get("negative_prompt"):
        kwargs["negative_prompt"] = params["negative_prompt"]

    with lora.applied_lora(pipe, params.get("lora_path"), params.get("lora_strength", 0.0)):
        image = pipe(**kwargs)

    meta = dict(
        mode="t2i",
        model=model_name,
        steps=kwargs["num_inference_steps"],
        cfg=kwargs["cfg_scale"],
        seed=kwargs["seed"],
        width=kwargs["width"],
        height=kwargs["height"],
        lora=str(params.get("lora_path")) if params.get("lora_path") else None,
        lora_strength=params.get("lora_strength", 0.0),
    )
    return image, meta


def call_controlnet(pipe: Any, params: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    """ControlNet — Turbo + Z-Image-Turbo-Fun-Controlnet-Union-2.1."""
    input_image: Image.Image | None = params.get("input_image")
    if input_image is None:
        raise ValueError("ControlNet mode requires an input image")

    preproc_mode = params.get("preprocessor", "Canny")
    try:
        control_image = preprocessors.run(preproc_mode, input_image)
    except Exception as e:
        import sys

        print(
            f"[modes] preprocessor {preproc_mode!r} failed: {e}; falling back to raw input", file=sys.stderr, flush=True
        )
        control_image = input_image

    # Same modulus-of-16 dance as call_upscale: DiffSynth's VAE encode rounds *down*
    # for control_latents while the noise allocator rounds *up* for inpaint_mask, so
    # an unaligned image makes torch.concat on control_context raise.
    w, h = control_image.size
    aligned_w, aligned_h = (w // 16) * 16, (h // 16) * 16
    if (aligned_w, aligned_h) != (w, h):
        control_image = control_image.crop((0, 0, aligned_w, aligned_h))

    _swap_transformer(pipe, "Turbo")

    cn_input = ControlNetInput(image=control_image, scale=float(params.get("controlnet_scale", 1.0)))

    kwargs: dict[str, Any] = dict(
        prompt=params["prompt"],
        cfg_scale=1.0,
        num_inference_steps=int(params.get("steps", 9)),
        sigma_shift=3.0,
        height=control_image.size[1],
        width=control_image.size[0],
        seed=int(params.get("seed", 0)),
        controlnet_inputs=[cn_input],
    )

    with lora.applied_lora(pipe, params.get("lora_path"), params.get("lora_strength", 0.0)):
        image = pipe(**kwargs)

    meta = dict(
        mode="controlnet",
        model="Turbo",
        preprocessor=preproc_mode,
        controlnet_scale=cn_input.scale,
        steps=kwargs["num_inference_steps"],
        cfg=1.0,
        seed=kwargs["seed"],
        width=kwargs["width"],
        height=kwargs["height"],
        lora=str(params.get("lora_path")) if params.get("lora_path") else None,
        lora_strength=params.get("lora_strength", 0.0),
    )
    return image, meta


def call_upscale(pipe: Any, params: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    """Upscale — RealESRGAN x4 → 0.5 resize → Z-Image-Turbo img2img refinement."""
    input_image: Image.Image | None = params.get("input_image")
    if input_image is None:
        raise ValueError("Upscale mode requires an input image")

    upscaled = upscale.realesrgan_2x(input_image, model_path=params["esrgan_model_path"])

    # DiffSynth rounds height/width *up* to multiples of 16 when allocating noise,
    # but its VAE rounds the encoded image *down* to the same modulus. If we hand it
    # an upscaled PIL whose dims aren't already aligned, the two latents come back
    # at different shapes and add_noise crashes (RuntimeError: tensor a vs b on dim 3).
    # Crop to the floor-multiple-of-16 here so both paths land on the same shape.
    w, h = upscaled.size
    aligned_w, aligned_h = (w // 16) * 16, (h // 16) * 16
    if (aligned_w, aligned_h) != (w, h):
        upscaled = upscaled.crop((0, 0, aligned_w, aligned_h))

    _swap_transformer(pipe, "Turbo")

    kwargs: dict[str, Any] = dict(
        prompt=params.get("prompt", "masterpiece, 8k"),
        cfg_scale=1.0,
        num_inference_steps=int(params.get("refine_steps", 5)),
        sigma_shift=3.0,
        input_image=upscaled,
        denoising_strength=float(params.get("refine_denoise", 0.33)),
        height=upscaled.size[1],
        width=upscaled.size[0],
        seed=int(params.get("seed", 0)),
    )

    with lora.applied_lora(pipe, params.get("lora_path"), params.get("lora_strength", 0.0)):
        image = pipe(**kwargs)

    meta = dict(
        mode="upscale",
        model="Turbo",
        refine_steps=kwargs["num_inference_steps"],
        refine_denoise=kwargs["denoising_strength"],
        seed=kwargs["seed"],
        width=upscaled.size[0],
        height=upscaled.size[1],
        lora=str(params.get("lora_path")) if params.get("lora_path") else None,
        lora_strength=params.get("lora_strength", 0.0),
    )
    return image, meta
