"""Onyx Amber theme — palette tokens, gr.themes.Base subclass, and CSS string."""

from __future__ import annotations

import gradio as gr

AMBER: dict[str, str] = {
    "body_bg": "#0F0C08",
    "panel_bg": "#0F0C08",
    "input_bg": "#0F0C08",
    "canvas_bg": "#110D08",
    "border": "#2A2218",
    "text": "#FAF1E3",
    "text_dim": "#A89478",
    "accent": "#FFB02E",
    "accent_text": "#1A1208",
    "radius": "8px",
    "radius_sm": "6px",
}


class _OnyxAmberBase(gr.themes.Base):
    """gr.themes.Base subclass that exposes font/font_mono as lists of font objects.

    In Gradio 5.x, Base.__init__ collapses font lists to a CSS string and assigns
    them to self.font / self.font_mono.  The internal lists are kept on self._font
    and self._font_mono.  This subclass redirects the public attributes to the
    internal lists so callers can iterate over GoogleFont / str entries directly.
    The CSS strings are preserved on self.font_str / self.font_mono_str.
    """

    @property  # type: ignore[override]
    def font(self) -> list:  # type: ignore[override]
        return self._font

    @font.setter
    def font(self, val: str) -> None:
        self.font_str = val

    @property  # type: ignore[override]
    def font_mono(self) -> list:  # type: ignore[override]
        return self._font_mono

    @font_mono.setter
    def font_mono(self, val: str) -> None:
        self.font_mono_str = val


def build_theme() -> gr.themes.Base:
    """Return a Gradio theme matching the Onyx Amber palette."""
    return _OnyxAmberBase(
        primary_hue=gr.themes.Color(
            c50="#FFF8E6",
            c100="#FFEFC2",
            c200="#FFE08A",
            c300="#FFD161",
            c400="#FFC042",
            c500=AMBER["accent"],
            c600="#E69926",
            c700="#B37A1F",
            c800="#805717",
            c900="#4D3510",
            c950="#1A1208",
        ),
        neutral_hue=gr.themes.Color(
            c50="#FAF1E3",
            c100="#E8DCC4",
            c200="#D4C2A1",
            c300="#A89478",
            c400="#867054",
            c500="#5C4D38",
            c600="#3C3225",
            c700="#2A2218",
            c800="#1C170F",
            c900="#100C08",
            c950="#0A0805",
        ),
        font=[gr.themes.GoogleFont("Geist"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Geist Mono"), "ui-monospace", "monospace"],
        radius_size=gr.themes.sizes.radius_md,
    ).set(
        body_background_fill=AMBER["body_bg"],
        body_text_color=AMBER["text"],
        body_text_color_subdued=AMBER["text_dim"],
        background_fill_primary=AMBER["panel_bg"],
        background_fill_secondary=AMBER["canvas_bg"],
        block_background_fill=AMBER["panel_bg"],
        block_border_color=AMBER["border"],
        block_border_width="1px",
        block_radius=AMBER["radius"],
        input_background_fill=AMBER["input_bg"],
        input_border_color=AMBER["border"],
        button_primary_background_fill=AMBER["accent"],
        button_primary_background_fill_hover=AMBER["accent"],
        button_primary_text_color=AMBER["accent_text"],
        button_primary_border_color=AMBER["accent"],
        slider_color=AMBER["accent"],
        color_accent=AMBER["accent"],
        color_accent_soft="rgba(255,176,46,0.12)",
    )


CSS: str = """
/* Onyx Amber — atmospheric layer that Gradio's theme can't express alone */

body, .gradio-container {
    background-image: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(255,176,46,0.06), transparent 70%);
}

/* Amber glow on primary button */
.gradio-container button.primary {
    box-shadow: 0 0 0 1px rgba(255,176,46,0.4), 0 8px 24px -8px rgba(255,176,46,0.35);
}

/* Slim status line typography */
.zis-status {
    font-family: 'Geist Mono', ui-monospace, monospace;
    font-size: 11px;
    letter-spacing: 0.06em;
    color: #A89478;
}

/* LoRA file slot — solid amber border + slim icon when a file is loaded */
.zis-lora.loaded {
    border: 1px solid #FFB02E !important;
}
""".strip()
