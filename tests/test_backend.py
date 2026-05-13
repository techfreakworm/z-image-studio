from unittest.mock import MagicMock

import pytest
from PIL import Image

import backend


def test_duration_t2i_turbo_is_short():
    d = backend.duration_for(mode="t2i", params=dict(model="Turbo", steps=8, width=1024, height=1024))
    assert 60 <= d <= 90


def test_duration_t2i_base_is_longer():
    d = backend.duration_for(mode="t2i", params=dict(model="Base", steps=25, width=1024, height=1024))
    assert d > 60


def test_duration_clamps_at_180():
    d = backend.duration_for(mode="t2i", params=dict(model="Base", steps=200, width=2048, height=2048))
    assert d == 180


def test_duration_clamps_at_60():
    d = backend.duration_for(mode="t2i", params=dict(model="Turbo", steps=1, width=256, height=256))
    assert d == 60


def test_duration_multiplier_scales_up():
    base = backend.duration_for(mode="t2i", params=dict(model="Turbo", steps=8, width=1024, height=1024))
    retry = backend.duration_for(
        mode="t2i", params=dict(model="Turbo", steps=8, width=1024, height=1024), multiplier=2.0
    )
    assert retry > base


def test_duration_upscale_has_realesrgan_overhead():
    t2i = backend.duration_for(mode="t2i", params=dict(model="Turbo", steps=8, width=1024, height=1024))
    upsc = backend.duration_for(mode="upscale", params=dict(refine_steps=5, width=1024, height=1024))
    assert upsc > t2i


@pytest.fixture
def fake_backend(monkeypatch):
    """A ZImageStudioBackend whose constructor doesn't actually build a pipeline."""
    monkeypatch.setattr(backend, "_build_pipeline", lambda *a, **kw: MagicMock())
    b = backend.ZImageStudioBackend()
    b.pipeline.return_value = Image.new("RGB", (32, 32))
    b.pipeline.dit = MagicMock()
    b.pipeline.model_pool = MagicMock()
    return b


def test_backend_generate_routes_t2i(fake_backend):
    img, meta = fake_backend.generate(
        mode="t2i",
        params=dict(
            prompt="cat",
            negative_prompt="",
            model="Turbo",
            steps=8,
            cfg=1.0,
            width=1024,
            height=1024,
            seed=42,
            lora_path=None,
            lora_strength=0.0,
        ),
    )
    assert isinstance(img, Image.Image)
    assert meta["mode"] == "t2i"
    assert meta["model"] == "Turbo"


def test_backend_generate_routes_controlnet(fake_backend, monkeypatch):
    monkeypatch.setattr(backend.modes, "preprocessors", type("P", (), {"run": staticmethod(lambda m, i: i)}))
    _img, meta = fake_backend.generate(
        mode="controlnet",
        params=dict(
            prompt="cat",
            input_image=Image.new("RGB", (64, 64)),
            preprocessor="Canny",
            controlnet_scale=1.0,
            steps=9,
            seed=0,
            lora_path=None,
            lora_strength=0.0,
        ),
    )
    assert meta["mode"] == "controlnet"


def test_backend_generate_unknown_mode_raises(fake_backend):
    with pytest.raises(ValueError):
        fake_backend.generate(mode="dance", params={})


def test_generate_with_retry_retries_on_gpu_aborted(fake_backend, monkeypatch):
    call_count = {"n": 0}
    original_generate = fake_backend.generate

    def flaky(mode, params):
        call_count["n"] += 1
        if call_count["n"] == 1:
            from gradio.exceptions import Error

            raise Error("GPU task aborted")
        return original_generate(mode, params)

    fake_backend.generate = flaky

    _img, meta = backend.generate_with_retry(
        fake_backend,
        mode="t2i",
        params=dict(
            prompt="x",
            negative_prompt="",
            model="Turbo",
            steps=8,
            cfg=1.0,
            width=1024,
            height=1024,
            seed=0,
            lora_path=None,
            lora_strength=0.0,
        ),
    )
    assert call_count["n"] == 2  # one fail + one retry
    assert meta["mode"] == "t2i"


def test_generate_with_retry_does_not_retry_other_errors(fake_backend):
    fake_backend.generate = lambda *a, **kw: (_ for _ in ()).throw(ValueError("not a gpu issue"))
    with pytest.raises(ValueError):
        backend.generate_with_retry(fake_backend, mode="t2i", params={})


def test_duration_honors_retry_multiplier_in_params():
    normal = backend.duration_for(mode="t2i", params=dict(model="Turbo", steps=8, width=1024, height=1024))
    retry = backend.duration_for(
        mode="t2i",
        params=dict(model="Turbo", steps=8, width=1024, height=1024, __retry_multiplier__=2.0),
    )
    assert retry > normal
