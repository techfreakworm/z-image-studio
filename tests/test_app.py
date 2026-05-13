import app


def test_on_model_change_returns_base_defaults():
    assert app._on_model_change("Base") == (25, 4.0)


def test_on_model_change_returns_turbo_defaults():
    assert app._on_model_change("Turbo") == (8, 1.0)


def test_on_model_change_unknown_falls_back_to_turbo():
    assert app._on_model_change("Edit") == (8, 1.0)


def test_preview_cn_returns_none_when_no_image():
    assert app._preview_cn(None, "Canny") is None


def test_preview_cn_falls_back_to_input_on_error():
    """If the preprocessor raises (e.g. missing optional dep), pass-through."""

    from PIL import Image

    img = Image.new("RGB", (16, 16), "white")
    # "BogusMode" raises ValueError inside preprocessors.run — _preview_cn should
    # swallow it and return the raw input.
    out = app._preview_cn(img, "BogusMode")
    assert out is img
