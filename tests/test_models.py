import os
from unittest import mock

import models


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
