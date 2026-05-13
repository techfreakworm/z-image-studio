"""LoRA file validation and apply/revert context manager."""
from __future__ import annotations

import json
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

ZIMAGE_LORA_PREFIXES = ("transformer.", "dit.", "model.transformer.")


class LoRAValidationError(ValueError):
    """Raised when a LoRA safetensors file doesn't match Z-Image's key layout."""


@dataclass(frozen=True)
class LoRAInfo:
    path: Path
    rank: int
    target: str  # which submodule it applies to ("transformer" for Z-Image)
    size_bytes: int


def sniff(path: Path | str) -> LoRAInfo:
    """Read just the safetensors header to verify and infer rank + target.

    Doesn't load tensors. Doesn't allocate GPU memory. Cheap enough to call before
    @spaces.GPU fires.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 8:
        raise LoRAValidationError(f"{path.name}: file too short to be safetensors")
    (header_len,) = struct.unpack("<Q", raw[:8])
    if header_len <= 0 or header_len + 8 > len(raw):
        raise LoRAValidationError(f"{path.name}: not a valid safetensors header")
    try:
        header = json.loads(raw[8 : 8 + header_len])
    except json.JSONDecodeError as e:
        raise LoRAValidationError(f"{path.name}: safetensors header is not JSON ({e})") from e

    tensor_keys = [k for k in header.keys() if not k.startswith("__")]
    if not tensor_keys:
        raise LoRAValidationError(f"{path.name}: no tensors in file")

    bad = [k for k in tensor_keys if not k.startswith(ZIMAGE_LORA_PREFIXES)]
    if bad:
        sample = bad[0]
        raise LoRAValidationError(
            f"{path.name}: unexpected key '{sample}' — Z-Image LoRAs must target "
            f"{ZIMAGE_LORA_PREFIXES} (got {len(bad)}/{len(tensor_keys)} mismatched keys)"
        )

    meta = header.get("__metadata__") or {}
    rank = int(meta.get("rank", 0))
    if not rank:
        # Infer from any A/B tensor pair shape
        for k, v in header.items():
            if "lora_A" in k or "lora_down" in k:
                shape = v.get("shape") or []
                if shape:
                    rank = int(min(shape))
                    break

    return LoRAInfo(
        path=path,
        rank=rank,
        target="transformer",
        size_bytes=path.stat().st_size,
    )


@contextmanager
def applied_lora(pipe: Any, path: Path | str | None, strength: float) -> Iterator[None]:
    """Apply a LoRA to the pipeline's dit for the duration of the context.

    Reverts on exit (including exception path) so the cached GPU model is left clean.
    If ``path`` is ``None``, this is a no-op.

    Validates the LoRA file with :func:`sniff` before touching the pipeline so a bad
    file is rejected before any GPU work begins.
    """
    if path is None:
        yield
        return

    sniff(path)  # raises LoRAValidationError on bad input
    _apply_lora_impl(pipe, path, strength)
    try:
        yield
    finally:
        _revert_lora_impl(pipe)


def _apply_lora_impl(pipe: Any, path: Path | str, strength: float) -> None:
    """Apply a LoRA to ``pipe.dit``. Imports DiffSynth lazily for testability."""
    from diffsynth.utils.lora import merge_lora
    merge_lora(pipe.dit, str(path), alpha=float(strength))


def _revert_lora_impl(pipe: Any) -> None:
    """Revert the most recent LoRA from ``pipe.dit``.

    Tries DiffSynth's ``unmerge_lora`` first; falls back to re-fetching clean
    weights from the model pool if unavailable.
    """
    try:
        from diffsynth.utils.lora import unmerge_lora
        unmerge_lora(pipe.dit)
        return
    except ImportError:
        pass

    if hasattr(pipe, "model_pool"):
        variant = getattr(pipe.dit, "_zis_variant", None)
        if variant:
            pipe.dit = pipe.model_pool.fetch_model("z_image_dit", variant=variant)
