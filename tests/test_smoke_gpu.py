import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def real_backend():
    """Build a real backend with real weights. ~30 GB download on first run."""
    import backend

    return backend.ZImageStudioBackend()


def test_t2i_turbo_produces_image(real_backend):
    from PIL import Image

    image, meta = real_backend.generate(
        mode="t2i",
        params=dict(
            prompt="a red apple on a wooden table",
            negative_prompt="",
            model="Turbo",
            steps=8,
            cfg=1.0,
            width=384,
            height=384,
            seed=42,
            lora_path=None,
            lora_strength=0.0,
        ),
    )
    assert isinstance(image, Image.Image)
    assert image.size == (384, 384)
    assert meta["model"] == "Turbo"


def test_t2i_base_produces_image(real_backend):
    from PIL import Image

    image, _meta = real_backend.generate(
        mode="t2i",
        params=dict(
            prompt="a red apple on a wooden table",
            negative_prompt="blurry",
            model="Base",
            steps=15,
            cfg=4.0,
            width=384,
            height=384,
            seed=42,
            lora_path=None,
            lora_strength=0.0,
        ),
    )
    assert isinstance(image, Image.Image)


def test_controlnet_produces_image(real_backend):
    import numpy as np
    from PIL import Image

    arr = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    image, _meta = real_backend.generate(
        mode="controlnet",
        params=dict(
            prompt="a portrait of a person, dramatic light",
            input_image=Image.fromarray(arr),
            preprocessor="Canny",
            controlnet_scale=1.0,
            steps=9,
            seed=42,
            lora_path=None,
            lora_strength=0.0,
        ),
    )
    assert isinstance(image, Image.Image)


def test_upscale_produces_image(real_backend, tmp_path):
    import numpy as np
    from huggingface_hub import hf_hub_download
    from PIL import Image

    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    image, _meta = real_backend.generate(
        mode="upscale",
        params=dict(
            prompt="masterpiece, 8k",
            input_image=Image.fromarray(arr),
            refine_steps=5,
            refine_denoise=0.33,
            seed=42,
            lora_path=None,
            lora_strength=0.0,
            esrgan_model_path=hf_hub_download("xinntao/Real-ESRGAN", "RealESRGAN_x4plus.pth"),
        ),
    )
    assert image.size == (512, 512)
