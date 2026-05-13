"""RealESRGAN x4plus wrapper + 0.5-resize bridge.

This module only handles the *pixel-space* upscale. The Z-Image-Turbo refinement
pass (img2img at denoise=0.33) lives in :mod:`modes` since it shares the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def realesrgan_2x(image: Image.Image | None, model_path: Path | str) -> Image.Image:
    """RealESRGAN x4plus → ``image.resize(0.5)`` → net 2x upscale."""
    if image is None:
        raise ValueError("upscale needs an input image")
    upscaled = _realesrgan_4x(model_path, image)
    w, h = upscaled.size
    return upscaled.resize((w // 2, h // 2), Image.LANCZOS)


_MODEL_CACHE: dict[str, Any] = {}


def _realesrgan_4x(model_path: Path | str, image: Image.Image) -> Image.Image:
    """Run RealESRGAN x4plus on ``image``. Caches the model in-process."""
    import numpy as np
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    key = str(model_path)
    if key not in _MODEL_CACHE:
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        _MODEL_CACHE[key] = RealESRGANer(
            scale=4,
            model_path=key,
            model=net,
            tile=512,  # split into tiles to avoid OOM on large inputs
            tile_pad=10,
            pre_pad=0,
            half=False,  # bf16 elsewhere; keep this fp32 for stability
            gpu_id=None,
        )

    upsampler = _MODEL_CACHE[key]
    arr = np.array(image.convert("RGB"))
    out_arr, _ = upsampler.enhance(arr, outscale=4)
    return Image.fromarray(out_arr)
