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
    """Swap the active transformer in the pipeline's model pool."""
    variant = "z_image" if model_name == "Base" else "z_image_turbo"
    pipe.dit = pipe.model_pool.fetch_model("z_image_dit", variant=variant)
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
    control_image = preprocessors.run(preproc_mode, input_image)

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

    _swap_transformer(pipe, "Turbo")

    kwargs: dict[str, Any] = dict(
        prompt=params.get("prompt", "masterpiece, 8k"),
        cfg_scale=1.0,
        num_inference_steps=int(params.get("refine_steps", 5)),
        sigma_shift=3.0,
        input_image=upscaled,
        denoising_strength=float(params.get("refine_denoise", 0.33)),
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
