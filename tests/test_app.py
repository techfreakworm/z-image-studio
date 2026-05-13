import app


def test_on_model_change_returns_base_defaults():
    steps, cfg, _lora_update = app._on_model_change("Base")
    assert (steps, cfg) == (25, 4.0)


def test_on_model_change_returns_turbo_defaults():
    steps, cfg, _lora_update = app._on_model_change("Turbo")
    assert (steps, cfg) == (8, 1.0)


def test_on_model_change_unknown_falls_back_to_turbo():
    steps, cfg, _lora_update = app._on_model_change("Edit")
    assert (steps, cfg) == (8, 1.0)


def test_on_model_change_emits_lora_label_hinting_model_compat():
    """Third element is a gr.update on the LoRA toggle whose label names the model."""
    *_, base_update = app._on_model_change("Base")
    *_, turbo_update = app._on_model_change("Turbo")
    # gr.update returns a dict-like with the keyword args; both styles seen in Gradio 5.
    base_label = getattr(base_update, "get", lambda *_: None)("label") or getattr(base_update, "label", None)
    turbo_label = getattr(turbo_update, "get", lambda *_: None)("label") or getattr(turbo_update, "label", None)
    assert base_label and "Z-Image" in base_label and "Turbo" not in base_label
    assert turbo_label and "Z-Image-Turbo" in turbo_label


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
