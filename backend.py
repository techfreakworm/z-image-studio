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
    "t2i":        20,   # fixed setup + decode
    "controlnet": 30,   # + preprocessor + control patch
    "upscale":    50,   # + realesrgan pixel-space step
}
_PER_STEP_S: dict[tuple[str, str], float] = {
    ("t2i", "Base"):  2.4,
    ("t2i", "Turbo"): 1.6,
    ("controlnet", "Turbo"): 2.0,
    ("upscale", "Turbo"):    1.6,
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

    base = _BASE_DURATION_S.get(mode, 30)
    per_step = _PER_STEP_S.get((mode, model), _PER_STEP_S.get((mode, "Turbo"), 1.6))
    size_factor = (width * height) / (1024 * 1024)
    cold_buffer = 15  # CPU→GPU copy on first call after a quiet period

    est = (base + per_step * steps + cold_buffer) * size_factor * multiplier
    return max(60, min(int(est), 180))


def _identity(fn):
    return fn


_ON_SPACES = bool(os.environ.get("SPACES_ZERO_GPU"))
_GPU = spaces.GPU(duration=lambda *a, **kw: duration_for(*a[1:3], **kw)) \
       if (spaces is not None and _ON_SPACES) else _identity


def _build_pipeline() -> Any:
    """Construct the DiffSynth ZImagePipeline. Imported lazily to keep tests fast."""
    import torch
    from diffsynth.pipelines.z_image import ZImagePipeline

    import models

    device = models.auto_device()
    vram_cfg: dict[str, Any] = {}
    if device != "cpu":
        vram_cfg = dict(
            offload_dtype=torch.bfloat16, offload_device="cpu",
            onload_dtype=torch.bfloat16,  onload_device="cpu",
            preparing_dtype=torch.bfloat16, preparing_device=device,
            computation_dtype=torch.bfloat16, computation_device=device,
        )

    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=models.build_diffsynth_configs(vram_cfg=vram_cfg),
        tokenizer_config=models.build_diffsynth_configs(
            (models.TOKENIZER_CONFIG,), vram_cfg=None,
        )[0],
        vram_limit=models.vram_limit_for(device),
    )
    return pipe


_DISPATCH = {
    "t2i":        modes.call_t2i,
    "controlnet": modes.call_controlnet,
    "upscale":    modes.call_upscale,
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
