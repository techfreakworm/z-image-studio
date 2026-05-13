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

    # (a) Iterable lists: .font_list / .font_mono_list expose the original entries.
    fonts = [str(f) for f in th.font_list]
    assert any("Geist" in f for f in fonts)
    monos = [str(f) for f in th.font_mono_list]
    assert any("Geist Mono" in f for f in monos)

    # (b) CSS variables: _get_theme_css() must emit --font and --font-mono that
    #     reference Geist / Geist Mono so the browser actually loads the fonts.
    #     This assertion is what catches the original bug where property setters
    #     redirected self.font → font_str, causing --font-str to be emitted instead.
    css = th._get_theme_css()
    assert "--font:" in css, "--font CSS variable missing from generated theme CSS"
    assert "--font-mono:" in css, "--font-mono CSS variable missing from generated theme CSS"
    assert "Geist" in css, "Geist font name missing from generated theme CSS"
    assert "Geist Mono" in css, "Geist Mono font name missing from generated theme CSS"
