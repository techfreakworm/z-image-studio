"""Gradio UI builders + small HTML helpers for the (i) tooltip pattern and the custom model selector."""
from __future__ import annotations

from html import escape

GITHUB_MODEL_ZOO_URL = "https://github.com/Tongyi-MAI/Z-Image#model-zoo"


def labeled_label(text: str, info_text: str) -> str:
    """Return HTML for a label with an (i) tooltip icon next to it.

    Use immediately before a ``gr.Slider`` / ``gr.Textbox`` / ``gr.File`` etc.
    that itself has ``show_label=False``. The CSS for ``.zis-row-label`` and
    ``.zis-info`` is defined in :mod:`theme`.
    """
    return (
        f'<label class="zis-row-label">{escape(text)}'
        f'<span class="zis-info" data-info="{escape(info_text)}">i</span>'
        f'</label>'
    )


def model_selector_html(current: str = "Turbo") -> str:
    """Custom T2I model selector — 2-col phone / 4-col tablet+ grid of cards.

    Two functional ``<button>`` cards (Base, Turbo) — clicks fire
    ``zis.setModel('<name>')`` defined in app.py's ``head=`` script.

    Two coming-soon ``<a>`` cards (Edit, Omni Base) — open the Z-Image GitHub
    README's Model Zoo section in a new tab. Marked with a `.soon` class and a
    "soon" pill that doesn't overlap the model name (separate flex children).
    """
    current_safe = escape(current)
    cards: list[str] = []
    for name in ("Base", "Turbo"):
        cls = "zis-model on" if name == current else "zis-model"
        cards.append(
            f'<button type="button" class="{cls}" data-value="{name}" '
            f'onclick="zis.setModel(\'{name}\')">'
            f'<span class="dot"></span>'
            f'<span class="name">{name}</span>'
            f'</button>'
        )
    for name in ("Edit", "Omni Base"):
        cards.append(
            f'<a class="zis-model soon" '
            f'href="{GITHUB_MODEL_ZOO_URL}" '
            f'target="_blank" rel="noopener noreferrer">'
            f'<span class="dot"></span>'
            f'<span class="name">{name}<span class="ext">↗</span></span>'
            f'<span class="soon-tag">soon</span>'
            f'</a>'
        )
    _ = current_safe  # current is matched in cls above; this line keeps escape() exercised
    return f'<div class="zis-models">{"".join(cards)}</div>'
