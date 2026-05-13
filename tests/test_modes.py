from unittest.mock import MagicMock

import pytest
from PIL import Image

import modes


@pytest.fixture
def fake_pipe():
    """Stand-in pipeline that records its __call__ args and returns a dummy image."""
    pipe = MagicMock()
    pipe.dit = MagicMock()
    pipe.model_pool = MagicMock()
    pipe.return_value = Image.new("RGB", (64, 64), color=(255, 176, 46))
    return pipe


def test_t2i_turbo_builds_minimal_call(fake_pipe):
    out, meta = modes.call_t2i(
        fake_pipe,
        params=dict(
            prompt="a cat",
            negative_prompt="",
            model="Turbo",
            steps=8, cfg=1.0,
            width=1024, height=1024,
            seed=42,
            lora_path=None, lora_strength=0.0,
        ),
    )
    fake_pipe.assert_called_once()
    kwargs = fake_pipe.call_args.kwargs
    assert kwargs["prompt"] == "a cat"
    assert kwargs["cfg_scale"] == 1.0
    assert kwargs["num_inference_steps"] == 8
    assert kwargs["width"] == 1024
    assert kwargs["seed"] == 42
    assert kwargs["sigma_shift"] == 3.0
    assert "negative_prompt" not in kwargs or not kwargs.get("negative_prompt")
    assert meta["model"] == "Turbo"
    assert meta["steps"] == 8
    assert isinstance(out, Image.Image)


def test_t2i_base_passes_negative_prompt_and_cfg4(fake_pipe):
    modes.call_t2i(
        fake_pipe,
        params=dict(
            prompt="a cat", negative_prompt="blurry, lowres",
            model="Base", steps=25, cfg=4.0,
            width=1024, height=1024, seed=42,
            lora_path=None, lora_strength=0.0,
        ),
    )
    kwargs = fake_pipe.call_args.kwargs
    assert kwargs["negative_prompt"] == "blurry, lowres"
    assert kwargs["cfg_scale"] == 4.0
    assert kwargs["num_inference_steps"] == 25


def test_t2i_swaps_transformer_via_model_pool(fake_pipe):
    modes.call_t2i(
        fake_pipe,
        params=dict(prompt="x", negative_prompt="", model="Base", steps=25, cfg=4.0,
                    width=1024, height=1024, seed=0, lora_path=None, lora_strength=0.0),
    )
    fake_pipe.model_pool.fetch_model.assert_called()
    call = fake_pipe.model_pool.fetch_model.call_args
    assert call.args[0] == "z_image_dit"
