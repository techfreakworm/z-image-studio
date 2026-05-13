import gradio as gr
import pytest

import ui


def test_labeled_label_returns_html_string():
    out = ui.labeled_label("Steps", "Denoising steps.")
    assert isinstance(out, str)
    assert "<label" in out and "</label>" in out
    assert ">Steps<" in out
    assert 'data-info="Denoising steps."' in out
    assert ">i<" in out  # the icon glyph


def test_labeled_label_escapes_html_chars():
    out = ui.labeled_label("Steps <x>", 'A "quoted" hint')
    assert "<x>" not in out
    assert "&lt;x&gt;" in out
    assert "&quot;quoted&quot;" in out


def test_model_selector_html_marks_current_as_on():
    out = ui.model_selector_html(current="Turbo")
    assert 'class="zis-model on" data-value="Turbo"' in out
    assert 'class="zis-model" data-value="Base"' in out


def test_model_selector_html_includes_both_soon_cards_with_github_link():
    out = ui.model_selector_html(current="Turbo")
    assert out.count("github.com/Tongyi-MAI/Z-Image#model-zoo") == 2
    assert "Edit" in out
    assert "Omni Base" in out
    assert "soon-tag" in out
    assert 'target="_blank"' in out
    assert 'rel="noopener noreferrer"' in out


def test_model_selector_html_defaults_to_turbo():
    out = ui.model_selector_html()
    assert 'class="zis-model on" data-value="Turbo"' in out


def test_model_selector_html_escapes_current_value():
    out = ui.model_selector_html(current="<script>alert(1)</script>")
    assert "<script>" not in out


@pytest.fixture(autouse=True)
def _blocks_ctx():
    """Each builder must be called inside a gr.Blocks() context."""
    with gr.Blocks():
        yield


def test_build_t2i_tab_returns_components():
    components = ui.build_t2i_tab()
    expected = {
        "prompt",
        "negative_prompt",
        "model_state",
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


def test_build_controlnet_tab_returns_components():
    components = ui.build_controlnet_tab()
    expected = {
        "prompt",
        "input_image",
        "preprocessor",
        "controlnet_scale",
        "steps",
        "seed",
        "lora_path",
        "lora_strength",
        "generate_btn",
        "output_image",
        "output_meta",
    }
    assert expected.issubset(components.keys())


def test_build_upscale_tab_returns_components():
    components = ui.build_upscale_tab()
    expected = {
        "prompt",
        "input_image",
        "refine_steps",
        "refine_denoise",
        "seed",
        "lora_path",
        "lora_strength",
        "generate_btn",
        "output_image",
        "output_meta",
    }
    assert expected.issubset(components.keys())
