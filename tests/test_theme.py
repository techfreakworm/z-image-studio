import gradio as gr

import theme


def test_palette_tokens_match_soft_dark_restraint_spec():
    pal = theme.PALETTE
    assert pal["body_bg"] == "#1A1614"
    assert pal["text"] == "#F0E8DD"
    assert pal["text_dim"] == "#988B7C"
    assert pal["border"] == "#2A241E"
    assert pal["accent"] == "#FFB02E"
    assert pal["accent_text"] == "#1A1208"
    assert pal["radius"] == "6px"


def test_build_theme_returns_gradio_base():
    th = theme.build_theme()
    assert isinstance(th, gr.themes.Base)


def test_theme_drops_mono_font():
    """Redesign uses Inter only — no Geist / Geist Mono / mono custom font."""
    th = theme.build_theme()
    css = th._get_theme_css()
    assert "Geist" not in css
    assert "Geist Mono" not in css


def test_css_includes_soon_row_styling():
    css = theme.CSS
    assert ".zis-soon-row" in css
    assert ".zis-soon-row a" in css


def test_css_includes_compact_lora_file_widget():
    css = theme.CSS
    assert ".zis-lora-file" in css
    assert "min-height: 56px" in css


def test_css_does_not_reference_deleted_selectors():
    css = theme.CSS
    # Old (i) tooltip pattern is gone — info= flows through gr.* directly.
    assert ".zis-info" not in css
    # Old custom card grid is gone — gr.Radio replaces it.
    assert ".zis-models" not in css
    assert ".zis-model.on" not in css
    assert ".zis-model.soon" not in css
