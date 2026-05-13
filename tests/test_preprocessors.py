import numpy as np
import pytest
from PIL import Image

import preprocessors


@pytest.fixture
def gradient_image():
    arr = np.linspace(0, 255, 256 * 256, dtype=np.uint8).reshape(256, 256)
    return Image.fromarray(arr).convert("RGB")


def test_modes_are_listed():
    assert preprocessors.MODES == ("Canny", "Depth", "Pose", "Pre-processed")


def test_canny_returns_rgb_image_of_same_size(gradient_image):
    out = preprocessors.run("Canny", gradient_image)
    assert isinstance(out, Image.Image)
    assert out.size == gradient_image.size
    assert out.mode == "RGB"


def test_passthrough_returns_input_unchanged(gradient_image):
    out = preprocessors.run("Pre-processed", gradient_image)
    assert out is gradient_image


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        preprocessors.run("Sobel", Image.new("RGB", (32, 32)))


def test_run_with_image_none_raises():
    with pytest.raises(ValueError):
        preprocessors.run("Canny", None)
