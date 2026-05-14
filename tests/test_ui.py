import gradio as gr
import pytest

import ui


@pytest.fixture(autouse=True)
def _blocks_ctx():
    """Each builder must be called inside a gr.Blocks() context."""
    with gr.Blocks():
        yield


def test_model_soon_row_links_to_zoo_twice():
    html = ui._model_soon_row_html()
    assert html.count(ui.MODEL_ZOO_URL) == 2
    assert "Edit" in html and "Omni Base" in html
    assert "(coming soon)" in html
    assert 'target="_blank"' in html
    assert 'rel="noopener noreferrer"' in html


def test_model_soon_row_uses_dim_link_class():
    html = ui._model_soon_row_html()
    assert 'class="zis-soon-row"' in html


def test_build_t2i_tab_returns_components():
    components = ui.build_t2i_tab()
    expected = {
        "prompt",
        "negative_prompt",
        "model",
        "base_group",
        "lora_enabled",
        "lora_group",
        "steps",
        "cfg",
        "width",
        "height",
        "seed",
        "lora_path",
        "lora_strength",
        "generate_btn",
        "output_image",
        "output_meta",
    }
    assert expected.issubset(components.keys())


def test_build_t2i_tab_model_is_native_radio():
    components = ui.build_t2i_tab()
    assert isinstance(components["model"], gr.Radio)
    assert components["model"].value == "Turbo"
    # Confirm the radio carries both choices. Gradio stores them as (label, value)
    # tuples on the .choices attribute.
    values = [c[1] for c in components["model"].choices]
    assert values == ["Base", "Turbo"]


def test_build_t2i_tab_lora_group_starts_hidden():
    components = ui.build_t2i_tab()
    assert components["lora_group"].visible is False
    assert components["lora_enabled"].value is False


def test_build_t2i_tab_base_group_starts_hidden():
    """Turbo is the default model, so the Base-only fields are hidden up front."""
    components = ui.build_t2i_tab()
    assert components["base_group"].visible is False


def test_build_controlnet_tab_returns_components():
    components = ui.build_controlnet_tab()
    expected = {
        "prompt",
        "input_image",
        "preview_image",
        "preprocessor",
        "controlnet_scale",
        "steps",
        "seed",
        "lora_enabled",
        "lora_group",
        "lora_path",
        "lora_strength",
        "generate_btn",
        "output_image",
        "output_meta",
    }
    assert expected.issubset(components.keys())


def test_build_controlnet_tab_preview_is_non_interactive_image():
    components = ui.build_controlnet_tab()
    assert isinstance(components["preview_image"], gr.Image)
    assert components["preview_image"].interactive is False


def test_build_upscale_tab_returns_components():
    components = ui.build_upscale_tab()
    expected = {
        "prompt",
        "input_image",
        "refine_steps",
        "refine_denoise",
        "seed",
        "generate_btn",
        "output_image",
        "output_meta",
    }
    assert expected.issubset(components.keys())


def test_build_upscale_tab_has_no_lora_components():
    """LoRA is intentionally not wired on Upscale — the refinement pass uses
    too low a denoising window for style LoRAs to meaningfully apply."""
    components = ui.build_upscale_tab()
    for key in ("lora_enabled", "lora_group", "lora_path", "lora_strength"):
        assert key not in components
