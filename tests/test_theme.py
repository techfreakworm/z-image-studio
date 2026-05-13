import theme


def test_amber_palette_tokens_match_spec():
    pal = theme.AMBER
    assert pal["body_bg"] == "#0F0C08"
    assert pal["text"] == "#FAF1E3"
    assert pal["text_dim"] == "#A89478"
    assert pal["border"] == "#2A2218"
    assert pal["accent"] == "#FFB02E"
    assert pal["accent_text"] == "#1A1208"
    assert pal["radius"] == "8px"


def test_build_theme_returns_gradio_base():
    import gradio as gr

    th = theme.build_theme()
    assert isinstance(th, gr.themes.Base)


def test_css_string_contains_critical_selectors():
    css = theme.CSS
    # warm vignette + amber button glow are the two decorations the spec calls out
    assert "radial-gradient" in css
    assert "rgba(255,176,46" in css.lower() or "255, 176, 46" in css.lower()


def test_fonts_geist_and_geist_mono():
    th = theme.build_theme()
    # gr.themes.GoogleFont stringifies to its name
    fonts = [str(f) for f in th.font]
    assert any("Geist" in f for f in fonts)
    monos = [str(f) for f in th.font_mono]
    assert any("Geist Mono" in f for f in monos)
