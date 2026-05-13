"""ControlNet preprocessors — lazy imports so an unused mode pays no cost."""

from __future__ import annotations

from typing import Any

from PIL import Image

MODES: tuple[str, ...] = ("Canny", "Depth", "Pose", "Pre-processed")


def run(mode: str, image: Image.Image | None) -> Image.Image:
    if image is None:
        raise ValueError("preprocessor needs an input image")
    if mode == "Canny":
        return _run_canny(image)
    if mode == "Depth":
        return _run_depth(image)
    if mode == "Pose":
        return _run_pose(image)
    if mode == "Pre-processed":
        return image
    raise ValueError(f"unknown preprocessor mode: {mode!r}; expected one of {MODES}")


def _run_canny(image: Image.Image) -> Image.Image:
    import cv2
    import numpy as np

    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def _run_depth(image: Image.Image) -> Image.Image:
    # controlnet_aux's Processor takes "depth_midas", NOT "midas".
    # Plain "midas" is not in its MODELS dict and raises KeyError.
    proc = _get_processor("depth_midas")
    out: Any = proc(image)
    if isinstance(out, Image.Image):
        return out.convert("RGB")
    return Image.fromarray(out).convert("RGB")


def _run_pose(image: Image.Image) -> Image.Image:
    proc = _get_processor("openpose")
    out: Any = proc(image)
    if isinstance(out, Image.Image):
        return out.convert("RGB")
    return Image.fromarray(out).convert("RGB")


_PROCESSOR_CACHE: dict[str, Any] = {}


def _get_processor(name: str) -> Any:
    """Lazy-init and cache a controlnet_aux Processor."""
    if name not in _PROCESSOR_CACHE:
        from controlnet_aux.processor import Processor

        _PROCESSOR_CACHE[name] = Processor(name)
    return _PROCESSOR_CACHE[name]
