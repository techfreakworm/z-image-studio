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
    """gr.themes.Base subclass with helpers for iterating over font lists.

    Gradio collapses font lists into a CSS string at __init__; we expose the
    original lists via .font_list / .font_mono_list for code that needs to
    iterate.  The public self.font / self.font_mono attributes are left alone so
    that _get_theme_css() emits the correct --font / --font-mono CSS variables.
    """

    @property
    def font_list(self) -> list:
        """Return the original font list (GoogleFont + str entries)."""
        return self._font

    @property
    def font_mono_list(self) -> list:
        """Return the original monospace font list (GoogleFont + str entries)."""
        return self._font_mono


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

/* ===== Param tooltip — (i) icon next to labels (spec § 4.6) ===== */

.zis-row-label {
    display: inline-flex; align-items: center;
    font-size: 11px; color: #A89478; font-weight: 500;
    margin-bottom: 6px;
}
.zis-info {
    display: inline-flex; align-items: center; justify-content: center;
    width: 12px; height: 12px;
    font: italic 600 8px 'Geist', system-ui, sans-serif;
    border: 1px solid #2A2218; border-radius: 50%;
    color: #A89478; vertical-align: super;
    margin-left: 3px; cursor: help; position: relative;
    transition: border-color 0.12s, color 0.12s;
}
.zis-info:hover { border-color: #FFB02E; color: #FFB02E; }
.zis-info::after {
    content: attr(data-info);
    position: absolute; bottom: 100%; left: 50%;
    transform: translateX(-50%) translateY(-4px);
    background: #1C170F; color: #FAF1E3;
    border: 1px solid #2A2218; border-radius: 6px;
    padding: 6px 10px;
    font: 400 11px 'Geist', system-ui, sans-serif; line-height: 1.4;
    width: 200px; white-space: normal;
    opacity: 0; pointer-events: none;
    transition: opacity 0.12s; z-index: 50;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}
.zis-info:hover::after, .zis-info.shown::after { opacity: 1; }

/* ===== Custom model selector — 2-col phone / 4-col tablet+ (spec § 4.7) ===== */

.zis-models {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 10px;
}
@media (min-width: 768px) {
    .zis-models { grid-template-columns: repeat(4, 1fr); }
}
.zis-model {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 12px;
    border: 1px solid #2A2218; border-radius: 8px;
    background: transparent; cursor: pointer;
    color: #FAF1E3;
    font: 500 12px 'Geist', system-ui, sans-serif;
    text-decoration: none;
    transition: opacity 0.15s, border-color 0.15s, background 0.15s;
}
.zis-model .dot {
    width: 10px; height: 10px; border-radius: 50%;
    border: 1px solid #2A2218; flex-shrink: 0;
}
.zis-model .name { flex: 1; text-align: left; }
.zis-model.on {
    background: #FFB02E; color: #1A1208; border-color: #FFB02E;
}
.zis-model.on .dot { background: #1A1208; border-color: #1A1208; }
.zis-model.soon {
    opacity: 0.55;
    background: rgba(255,176,46,0.04);
    border-style: dashed;
    position: relative;
}
.zis-model.soon .name { color: #A89478; }
.zis-model.soon .name .ext {
    font-size: 10px; color: #FFB02E;
    margin-left: 4px; vertical-align: super;
}
.zis-model.soon .soon-tag {
    font-family: 'Geist Mono', ui-monospace, monospace;
    font-size: 8.5px; letter-spacing: 0.12em; text-transform: uppercase;
    background: rgba(255,176,46,0.18); color: #FFB02E;
    padding: 2px 6px; border-radius: 100px;
    flex-shrink: 0;
}
.zis-model.soon:hover { opacity: 0.78; border-color: #FFB02E; }
.zis-model.soon::after {
    content: "Coming soon — opens GitHub";
    position: absolute; bottom: 100%; left: 50%;
    transform: translateX(-50%) translateY(-4px);
    background: #1C170F; color: #FAF1E3;
    border: 1px solid #2A2218; border-radius: 6px;
    padding: 6px 10px;
    font: 400 11px 'Geist', system-ui, sans-serif;
    white-space: nowrap;
    opacity: 0; pointer-events: none;
    transition: opacity 0.12s; z-index: 50;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}
.zis-model.soon:hover::after { opacity: 1; }
""".strip()
