"""ZImageStudioBackend — wraps the DiffSynth pipeline; applies @spaces.GPU on HF Spaces."""

from __future__ import annotations

import os
from typing import Any

# Spaces import is optional — running locally we don't have it.
try:
    import spaces  # type: ignore
except ImportError:
    spaces = None  # type: ignore[assignment]

import modes

_BASE_DURATION_S: dict[str, int] = {
    "t2i": 20,  # fixed setup + decode
    "controlnet": 30,  # + preprocessor + control patch
    "upscale": 50,  # + realesrgan pixel-space step
}
_PER_STEP_S: dict[tuple[str, str], float] = {
    ("t2i", "Base"): 2.4,
    ("t2i", "Turbo"): 1.6,
    ("controlnet", "Turbo"): 2.0,
    ("upscale", "Turbo"): 1.6,
}


def duration_for(
    mode: str,
    params: dict[str, Any],
    multiplier: float = 1.0,
) -> int:
    """Estimate ZeroGPU duration for a request. Pure function; clamped to [60, 180]."""
    model = params.get("model", "Turbo")
    steps = int(params.get("steps") or params.get("refine_steps") or 8)
    width = int(params.get("width", 1024))
    height = int(params.get("height", 1024))

    eff_multiplier = float(params.get("__retry_multiplier__", multiplier))

    base = _BASE_DURATION_S.get(mode, 30)
    per_step = _PER_STEP_S.get((mode, model), _PER_STEP_S.get((mode, "Turbo"), 1.6))
    size_factor = (width * height) / (1024 * 1024)
    cold_buffer = 15  # CPU→GPU copy on first call after a quiet period

    est = (base + per_step * steps + cold_buffer) * size_factor * eff_multiplier
    return max(60, min(int(est), 180))


def _identity(fn):
    return fn


_ON_SPACES = bool(os.environ.get("SPACES_ZERO_GPU"))
_GPU = (
    spaces.GPU(duration=lambda *a, **kw: duration_for(*a[1:3], **kw))
    if (spaces is not None and _ON_SPACES)
    else _identity
)


def _build_pipeline() -> Any:
    """Construct a ZImagePipeline carrying BOTH Base and Turbo transformers.

    DiffSynth's ``ZImagePipeline.from_pretrained`` builds a fresh ``ModelPool``
    locally and throws it away after attaching ``pipe.dit`` etc. — so a later
    transformer swap has nothing to switch between. We replicate the same
    initialization manually and keep the pool on ``pipe._zis_pool`` so
    :func:`modes._swap_transformer` can flip ``pipe.dit`` between the two
    ``z_image_dit`` entries (Base loaded first, Turbo second per MODEL_CONFIGS).
    """
    import torch
    from diffsynth.pipelines.z_image import ZImagePipeline
    from transformers import AutoTokenizer

    import models

    device = models.auto_device()
    vram_cfg: dict[str, Any] = {}
    if device != "cpu":
        vram_cfg = dict(
            offload_dtype=torch.bfloat16,
            offload_device="cpu",
            onload_dtype=torch.bfloat16,
            onload_device="cpu",
            preparing_dtype=torch.bfloat16,
            preparing_device=device,
            computation_dtype=torch.bfloat16,
            computation_device=device,
        )

    pipe = ZImagePipeline(device=device, torch_dtype=torch.bfloat16)

    # Load every safetensors listed in MODEL_CONFIGS — both transformers + shared
    # text encoder + VAE + controlnet — into one pool.
    pool = pipe.download_and_load_models(
        models.build_diffsynth_configs(vram_cfg=vram_cfg),
        vram_limit=models.vram_limit_for(device),
    )
    pipe._zis_pool = pool

    pipe.text_encoder = pool.fetch_model("z_image_text_encoder")
    pipe.dit = pool.fetch_model("z_image_dit")  # first match = Base per load order
    pipe.vae_encoder = pool.fetch_model("flux_vae_encoder")
    pipe.vae_decoder = pool.fetch_model("flux_vae_decoder")
    pipe.controlnet = pool.fetch_model("z_image_controlnet")
    # Optional image encoders that DiffSynth's ZImagePipeline references but
    # aren't in our preload (Omni / image2lora). fetch_model returns None when
    # absent — that's the documented "not an error" path.
    pipe.image_encoder = pool.fetch_model("siglip_vision_model_428m")
    pipe.siglip2_image_encoder = pool.fetch_model("siglip2_image_encoder")
    pipe.dinov3_image_encoder = pool.fetch_model("dinov3_image_encoder")
    pipe.image2lora_style = pool.fetch_model("z_image_image2lora_style")

    # Tokenizer (Qwen3-4B tokenizer dir under Z-Image)
    tok_cfg = models.build_diffsynth_configs((models.TOKENIZER_CONFIG,), vram_cfg=None)[0]
    tok_cfg.download_if_necessary()
    pipe.tokenizer = AutoTokenizer.from_pretrained(tok_cfg.path)

    pipe.vram_management_enabled = pipe.check_vram_management_state()
    return pipe


_DISPATCH = {
    "t2i": modes.call_t2i,
    "controlnet": modes.call_controlnet,
    "upscale": modes.call_upscale,
}


class ZImageStudioBackend:
    """One-process backend wrapping the DiffSynth ZImagePipeline."""

    def __init__(self) -> None:
        self.pipeline = _build_pipeline()

    @_GPU
    def generate(self, mode: str, params: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        handler = _DISPATCH.get(mode)
        if handler is None:
            raise ValueError(f"unknown mode: {mode!r}; expected one of {list(_DISPATCH)}")
        return handler(self.pipeline, params)


def generate_with_retry(
    backend_instance: ZImageStudioBackend,
    mode: str,
    params: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Call backend_instance.generate; on ZeroGPU timeout, retry once with 2x duration budget."""
    try:
        return backend_instance.generate(mode, params)
    except Exception as e:
        msg = str(e).lower()
        if "gpu task aborted" in msg or ("gpu" in msg and "aborted" in msg):
            retry_params = dict(params)
            retry_params["__retry_multiplier__"] = 2.0
            return backend_instance.generate(mode, retry_params)
        raise
