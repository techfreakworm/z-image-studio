"""Device autodetect, ZImagePipeline ModelConfig registry, and (Task 4) HF cache mirror."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Avoid importing torch at module load — keeps `import models` fast in CI.


def on_spaces() -> bool:
    """True iff we are running inside a Hugging Face ZeroGPU Space."""
    return bool(os.environ.get("SPACES_ZERO_GPU"))


def auto_device() -> str:
    """Detect the best available compute device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def vram_limit_for(device: str, free_gb: float | None = None) -> float:
    """Conservative VRAM limit (GB) passed to DiffSynth's vram_management.

    - CUDA: keep ~5% headroom (loaded models + scratch).
    - MPS: half of unified memory (CPU still needs RAM), capped.
    - CPU: 0.0 (no offload budget).
    """
    if device == "cpu":
        return 0.0
    if free_gb is None:
        import torch

        if device == "cuda":
            free_gb = torch.cuda.mem_get_info()[1] / (1024**3)
        else:  # mps
            # torch.mps has no mem_get_info on most builds; fall back to a safe constant.
            free_gb = 24.0
    if device == "mps":
        # Use half of unified memory; clamp to 8 GB floor for safety.
        return max(8.0, free_gb / 2)
    # cuda
    return max(8.0, free_gb - 4.0)


@dataclass(frozen=True)
class ModelConfig:
    """Lightweight wrapper around DiffSynth's ModelConfig.

    Stored as plain data so this module imports cheaply in CI. The real
    ``diffsynth.core.ModelConfig`` instance is built on demand by
    :func:`build_diffsynth_configs`.
    """

    model_id: str
    origin_file_pattern: str
    description: str = ""


MODEL_CONFIGS: tuple[ModelConfig, ...] = (
    # Base
    ModelConfig("Tongyi-MAI/Z-Image", "transformer/*.safetensors", "Z-Image base transformer (25 steps, cfg=4)"),
    ModelConfig(
        "Tongyi-MAI/Z-Image", "text_encoder/*.safetensors", "Qwen3-4B text encoder — shared between base + turbo"
    ),
    ModelConfig(
        "Tongyi-MAI/Z-Image", "vae/diffusion_pytorch_model.safetensors", "Flux-family VAE — shared between base + turbo"
    ),
    # Turbo (transformer only — encoder + VAE come from the Z-Image entry above)
    ModelConfig("Tongyi-MAI/Z-Image-Turbo", "transformer/*.safetensors", "Z-Image-Turbo transformer (8 steps, cfg=1)"),
    # ControlNet Union 2.1 (eager preload per spec; can move to lazy if RAM is tight)
    ModelConfig(
        "PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
        "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors",
        "ControlNet Union 2.1 — canny/depth/pose",
    ),
)

TOKENIZER_CONFIG = ModelConfig("Tongyi-MAI/Z-Image", "tokenizer/", "Qwen3-4B tokenizer")


def build_diffsynth_configs(
    configs: tuple[ModelConfig, ...] = MODEL_CONFIGS,
    vram_cfg: dict[str, Any] | None = None,
) -> list[Any]:
    """Build DiffSynth ``ModelConfig`` instances from our lightweight dataclasses.

    Called at app boot; not at module import. ``vram_cfg`` is the disk-offload
    block (offload_dtype, offload_device, etc.) that DiffSynth's low-VRAM examples use.
    """
    from diffsynth.core import ModelConfig as DSConfig

    return [
        DSConfig(model_id=c.model_id, origin_file_pattern=c.origin_file_pattern, **(vram_cfg or {})) for c in configs
    ]


def mirror_preload_hf_cache(src_root: Path | str, dst_root: Path | str) -> None:
    """Mirror a read-only HF cache tree (preload_from_hub) into a writable tree.

    - ``blobs/<sha>`` files -> **hardlinked** (zero-copy, shared inode).
    - ``snapshots/<commit>/...`` symlinks -> **preserved** with original relative target.
    - ``refs/<branch>`` files -> **byte-copied** (HF lib overwrites on etag check).
    - Directories -> ``mkdir`` so the runtime user owns them.

    Falls back to ``symlink`` when ``os.link()`` raises EXDEV (cross-device).
    """
    import errno
    import shutil

    src_root = Path(src_root)
    dst_root = Path(dst_root)

    if not (src_root / "hub").exists():
        return  # nothing preloaded -- no-op

    for src_dir, _, files in os.walk(src_root / "hub"):
        rel = Path(src_dir).relative_to(src_root)
        dst_dir = dst_root / rel
        dst_dir.mkdir(parents=True, exist_ok=True)

        for name in files:
            src_path = Path(src_dir) / name
            dst_path = dst_dir / name
            if dst_path.exists():
                continue

            # Refs get byte-copied
            if "refs/" in str(rel).replace("\\", "/"):
                shutil.copy2(src_path, dst_path)
                continue

            # Symlinks (snapshot files) preserve their relative target
            if src_path.is_symlink():
                target = os.readlink(src_path)
                dst_path.symlink_to(target)
                continue

            # Regular files (blobs) hardlink with EXDEV fallback
            try:
                os.link(src_path, dst_path)
            except OSError as e:
                if e.errno == errno.EXDEV:
                    dst_path.symlink_to(src_path)
                else:
                    raise
