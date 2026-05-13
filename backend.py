"""ZImageStudioBackend — wraps the DiffSynth pipeline; applies @spaces.GPU on HF Spaces."""
from __future__ import annotations

import os
from typing import Any

# Spaces import is optional — running locally we don't have it.
try:
    import spaces  # type: ignore
except ImportError:
    spaces = None  # type: ignore[assignment]


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
