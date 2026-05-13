from unittest import mock
import pytest
from PIL import Image

import upscale


@pytest.fixture
def small_image():
    return Image.new("RGB", (256, 256), color=(120, 50, 200))


def test_realesrgan_2x_produces_2x_image(small_image, monkeypatch):
    """RealESRGAN runs 4x then we scale down 0.5 → net 2x."""
    def fake_run_4x(_model_path, image):
        w, h = image.size
        return image.resize((w * 4, h * 4), Image.LANCZOS)
    monkeypatch.setattr(upscale, "_realesrgan_4x", fake_run_4x)

    out = upscale.realesrgan_2x(small_image, model_path="/dev/null")
    assert out.size == (512, 512)


def test_realesrgan_2x_rejects_none():
    with pytest.raises(ValueError):
        upscale.realesrgan_2x(None, model_path="/dev/null")
