import importlib
import os
from unittest import mock

import pytest

import models


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_auto_device_returns_cuda_or_mps_or_cpu():
    dev = models.auto_device()
    assert dev in ("cuda", "mps", "cpu")


def test_on_spaces_reads_env_var():
    with mock.patch.dict(os.environ, {"SPACES_ZERO_GPU": "1"}, clear=False):
        assert models.on_spaces() is True
    with mock.patch.dict(os.environ, {}, clear=True):
        assert models.on_spaces() is False


def test_model_configs_contains_both_transformers():
    configs = models.MODEL_CONFIGS
    repos = {c.model_id for c in configs}
    assert "Tongyi-MAI/Z-Image" in repos
    assert "Tongyi-MAI/Z-Image-Turbo" in repos
    assert "PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1" in repos


def test_vram_limit_for_cuda_is_reasonable():
    limit = models.vram_limit_for("cuda", free_gb=80.0)
    assert 60.0 <= limit <= 80.0  # leave headroom


def test_vram_limit_for_mps_is_unified_memory_aware():
    limit = models.vram_limit_for("mps", free_gb=24.0)
    assert 12.0 <= limit <= 22.0  # half of unified, headroom


def test_vram_limit_for_cpu_is_zero():
    assert models.vram_limit_for("cpu", free_gb=64.0) == 0.0


def test_mirror_hardlinks_blobs(tmp_path):
    """Blobs (content-addressed files) get hardlinked into the mirror."""
    src = tmp_path / "src" / "hub"
    dst = tmp_path / "rw"
    blob_dir = src / "blobs"
    blob_dir.mkdir(parents=True)
    blob = blob_dir / "abcdef"
    blob.write_bytes(b"hello")

    models.mirror_preload_hf_cache(src.parent, dst)

    mirrored = dst / "hub" / "blobs" / "abcdef"
    assert mirrored.exists()
    assert mirrored.stat().st_ino == blob.stat().st_ino, "should be hardlinked"


def test_mirror_preserves_snapshot_symlinks(tmp_path):
    """Snapshot symlinks point at relative blob paths -- preserve as-is."""
    src = tmp_path / "src" / "hub"
    dst = tmp_path / "rw"
    (src / "blobs").mkdir(parents=True)
    blob = src / "blobs" / "abc"
    blob.write_bytes(b"content")
    snap_dir = src / "snapshots" / "v1"
    snap_dir.mkdir(parents=True)
    link = snap_dir / "model.safetensors"
    link.symlink_to("../../blobs/abc")

    models.mirror_preload_hf_cache(src.parent, dst)

    mirrored_link = dst / "hub" / "snapshots" / "v1" / "model.safetensors"
    assert mirrored_link.is_symlink()
    target = os.readlink(mirrored_link)
    assert target == "../../blobs/abc"


def test_mirror_byte_copies_refs(tmp_path):
    """Refs are rewritten by HF lib on etag; must be a real copy, not hardlink."""
    src = tmp_path / "src" / "hub"
    dst = tmp_path / "rw"
    refs_dir = src / "refs" / "main"
    refs_dir.mkdir(parents=True)
    ref = refs_dir / "v1"
    ref.write_text("commit-sha\n")

    models.mirror_preload_hf_cache(src.parent, dst)

    mirrored_ref = dst / "hub" / "refs" / "main" / "v1"
    assert mirrored_ref.read_text() == "commit-sha\n"
    assert mirrored_ref.stat().st_ino != ref.stat().st_ino, "must be a real copy"
