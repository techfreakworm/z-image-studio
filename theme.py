"""Soft Dark Restraint theme — warm-toned dark surface with a single amber accent.

Palette tokens, ``gr.themes.Base`` configuration, and a small CSS string that
applies the touches Gradio's theme tokens can't express alone (link row under
the model radio + compact LoRA file widget).
"""

from __future__ import annotations

import gradio as gr

# Single source of truth for the palette. The mockup at
# ``.superpowers/brainstorm/47889-1778679653/content/simple-v1.html`` is the
# locked spec for these values (Variant A — "Soft Dark Restraint").
PALETTE: dict[str, str] = {
    "body_bg": "#1A1614",
    "panel_bg": "#1F1B17",
    "input_bg": "#14110F",
    "text": "#F0E8DD",
    "text_dim": "#988B7C",
    "border": "#2A241E",
    "border_strong": "#3A3128",
    "accent": "#FFB02E",
    "accent_text": "#1A1208",
    "radius": "6px",
}


def build_theme() -> gr.themes.Base:
    """Return a Gradio theme matching the Soft Dark Restraint palette.

    Uses Gradio's default font chain (Inter + system-ui fallbacks). No mono
    font — the redesign drops monospaced UI text entirely.
    """
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#FFF8E6",
            c100="#FFEFC2",
            c200="#FFE08A",
            c300="#FFD161",
            c400="#FFC042",
            c500=PALETTE["accent"],
            c600="#E69926",
            c700="#B37A1F",
            c800="#805717",
            c900="#4D3510",
            c950="#1A1208",
        ),
        neutral_hue=gr.themes.Color(
            c50="#F0E8DD",
            c100="#E0D5C3",
            c200="#C8B89E",
            c300="#988B7C",
            c400="#7A6E60",
            c500="#5C5246",
            c600="#3A3128",
            c700="#2A241E",
            c800="#1F1B17",
            c900="#1A1614",
            c950="#14110F",
        ),
        radius_size=gr.themes.sizes.radius_sm,
    ).set(
        body_background_fill=PALETTE["body_bg"],
        body_text_color=PALETTE["text"],
        body_text_color_subdued=PALETTE["text_dim"],
        background_fill_primary=PALETTE["panel_bg"],
        background_fill_secondary=PALETTE["body_bg"],
        block_background_fill=PALETTE["panel_bg"],
        block_border_color=PALETTE["border"],
        block_border_width="1px",
        block_radius=PALETTE["radius"],
        input_background_fill=PALETTE["input_bg"],
        input_border_color=PALETTE["border"],
        button_primary_background_fill=PALETTE["accent"],
        button_primary_background_fill_hover=PALETTE["accent"],
        button_primary_text_color=PALETTE["accent_text"],
        button_primary_border_color=PALETTE["accent"],
        slider_color=PALETTE["accent"],
        color_accent=PALETTE["accent"],
        color_accent_soft="rgba(255,176,46,0.12)",
    )


CSS: str = """
/* Soft Dark Restraint — calm, single-accent decorations Gradio tokens can't express. */

/* Small dim link row under the model radio (Edit / Omni Base · coming soon). */
.zis-soon-row {
    margin-top: 6px;
    font-size: 12px;
    color: #988B7C;
    line-height: 1.45;
}
.zis-soon-row a {
    color: #988B7C;
    text-decoration: underline;
    text-decoration-color: #3A3128;
    text-underline-offset: 3px;
}
.zis-soon-row a:hover {
    color: #F0E8DD;
    text-decoration-color: #988B7C;
}
.zis-soon-row .sep {
    margin: 0 6px;
    color: #3A3128;
}
.zis-soon-row .dim {
    margin-left: 6px;
    color: #635A4E;
}

/* Compact LoRA file widget — tighten Gradio's default 400px drop zone. */
.zis-lora-file .upload-container { min-height: 56px !important; padding: 8px 12px !important; }
.zis-lora-file .icon-wrap, .zis-lora-file svg { display: none !important; }
.zis-lora-file .wrap > * { text-align: left; }

/* Brand period uses the accent — the only place the accent appears in chrome. */
.zis-brand-period { color: #FFB02E; }

/* Live-status dot — accent, matches the single-accent rule. */
.zis-status-dot::before {
    content: "";
    display: inline-block;
    width: 6px; height: 6px; border-radius: 50%;
    background: #FFB02E;
    margin-right: 6px;
    vertical-align: middle;
}

/* CTA bar — single-line "drop a like / follow for what's next". Subtle by
   design: dim text, faintest amber wash, hairline underline. Avoids the
   needy "PLEASE STAR THE REPO" energy. */
.zis-cta {
    margin: 4px 0 10px 0;
    padding: 8px 14px;
    font-size: 12px;
    line-height: 1.5;
    color: #988B7C;
    background:
      linear-gradient(180deg, rgba(255,176,46,0.05), rgba(255,176,46,0.02));
    border: 1px solid #2A241E;
    border-radius: 8px;
    text-align: center;
}
.zis-cta strong { color: #F0E8DD; font-weight: 600; }
.zis-cta .zis-cta-heart {
    display: inline-block;
    color: #FFB02E;
    transform: translateY(-1px);
    margin: 0 1px;
    animation: zis-cta-pulse 2.4s ease-in-out infinite;
}
@keyframes zis-cta-pulse {
    0%, 60%, 100% { transform: translateY(-1px) scale(1); }
    30% { transform: translateY(-1px) scale(1.18); }
}
.zis-cta .zis-cta-sep { margin: 0 10px; color: #3A3128; }
.zis-cta a {
    color: #FFB02E;
    text-decoration: none;
    border-bottom: 1px dashed rgba(255,176,46,0.4);
    transition: border-color 0.15s, color 0.15s;
}
.zis-cta a:hover {
    color: #FFC85A;
    border-bottom-color: #FFB02E;
}

@media (max-width: 600px) {
    .zis-cta { font-size: 11px; padding: 8px 10px; }
    .zis-cta .zis-cta-sep { display: block; height: 4px; margin: 0; visibility: hidden; }
}
""".strip()
