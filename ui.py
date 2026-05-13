"""Gradio UI builders + small HTML helpers for the (i) tooltip pattern and the custom model selector."""

from __future__ import annotations

from html import escape

import gradio as gr

import preprocessors
from tooltips import TOOLTIPS

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
        f"</label>"
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
            f"onclick=\"zis.setModel('{name}')\">"
            f'<span class="dot"></span>'
            f'<span class="name">{name}</span>'
            f"</button>"
        )
    for name in ("Edit", "Omni Base"):
        cards.append(
            f'<a class="zis-model soon" '
            f'href="{GITHUB_MODEL_ZOO_URL}" '
            f'target="_blank" rel="noopener noreferrer">'
            f'<span class="dot"></span>'
            f'<span class="name">{name}<span class="ext">↗</span></span>'
            f'<span class="soon-tag">soon</span>'
            f"</a>"
        )
    _ = current_safe  # current is matched in cls above; this line keeps escape() exercised
    return f'<div class="zis-models">{"".join(cards)}</div>'


def build_t2i_tab() -> dict[str, gr.components.Component]:
    with gr.Row():
        with gr.Column(scale=4):
            gr.HTML(labeled_label("Prompt", TOOLTIPS["prompt"]))
            prompt = gr.Textbox(lines=4, show_label=False, placeholder="A latina model peeking through pine branches…")
            gr.HTML(labeled_label("Negative prompt (Base only)", TOOLTIPS["negative_prompt"]))
            negative_prompt = gr.Textbox(lines=2, show_label=False, placeholder="blurry, lowres, distorted")
            gr.HTML(labeled_label("Model", TOOLTIPS["model"]))
            model_state = gr.Textbox(value="Turbo", visible=False, elem_id="zis-model-state")
            gr.HTML(model_selector_html(current="Turbo"))
            with gr.Row():
                with gr.Column():
                    gr.HTML(labeled_label("LoRA (optional)", TOOLTIPS["lora"]))
                    lora_path = gr.File(file_types=[".safetensors"], type="filepath", show_label=False)
                with gr.Column():
                    gr.HTML(labeled_label("LoRA strength", TOOLTIPS["lora_strength"]))
                    lora_strength = gr.Slider(0.0, 1.5, value=0.8, step=0.05, show_label=False)
            with gr.Row():
                with gr.Column():
                    gr.HTML(labeled_label("Steps", TOOLTIPS["steps"]))
                    steps = gr.Slider(1, 50, value=8, step=1, show_label=False)
                with gr.Column():
                    gr.HTML(labeled_label("CFG (Base only)", TOOLTIPS["cfg"]))
                    cfg = gr.Slider(0.5, 12.0, value=1.0, step=0.1, show_label=False)
            with gr.Row():
                with gr.Column():
                    gr.HTML(labeled_label("Width", TOOLTIPS["width"]))
                    width = gr.Slider(384, 1536, value=1024, step=64, show_label=False)
                with gr.Column():
                    gr.HTML(labeled_label("Height", TOOLTIPS["height"]))
                    height = gr.Slider(384, 1536, value=1024, step=64, show_label=False)
                with gr.Column():
                    gr.HTML(labeled_label("Seed (0 = random)", TOOLTIPS["seed"]))
                    seed = gr.Number(value=0, precision=0, show_label=False)
            generate_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=5):
            gr.HTML(labeled_label("Output", TOOLTIPS["output"]))
            output_image = gr.Image(type="pil", height=512, show_download_button=True, show_label=False)
            output_meta = gr.JSON(label="Meta", value={})
    return dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_state=model_state,
        steps=steps,
        cfg=cfg,
        width=width,
        height=height,
        seed=seed,
        lora_path=lora_path,
        lora_strength=lora_strength,
        generate_btn=generate_btn,
        output_image=output_image,
        output_meta=output_meta,
    )


def build_controlnet_tab() -> dict[str, gr.components.Component]:
    with gr.Row():
        with gr.Column(scale=4):
            gr.HTML(labeled_label("Prompt", TOOLTIPS["prompt"]))
            prompt = gr.Textbox(lines=3, show_label=False)
            gr.HTML(labeled_label("Control image", TOOLTIPS["controlnet_image"]))
            input_image = gr.Image(type="pil", height=240, show_label=False)
            with gr.Row():
                with gr.Column():
                    gr.HTML(labeled_label("Preprocessor", TOOLTIPS["controlnet_preprocessor"]))
                    preprocessor = gr.Dropdown(list(preprocessors.MODES), value="Canny", show_label=False)
                with gr.Column():
                    gr.HTML(labeled_label("ControlNet scale", TOOLTIPS["controlnet_scale"]))
                    controlnet_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, show_label=False)
            with gr.Row():
                with gr.Column():
                    gr.HTML(labeled_label("LoRA (optional)", TOOLTIPS["lora"]))
                    lora_path = gr.File(file_types=[".safetensors"], type="filepath", show_label=False)
                with gr.Column():
                    gr.HTML(labeled_label("LoRA strength", TOOLTIPS["lora_strength"]))
                    lora_strength = gr.Slider(0.0, 1.5, value=0.8, step=0.05, show_label=False)
            with gr.Row():
                with gr.Column():
                    gr.HTML(labeled_label("Steps", TOOLTIPS["steps"]))
                    steps = gr.Slider(1, 30, value=9, step=1, show_label=False)
                with gr.Column():
                    gr.HTML(labeled_label("Seed (0 = random)", TOOLTIPS["seed"]))
                    seed = gr.Number(value=0, precision=0, show_label=False)
            generate_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=5):
            gr.HTML(labeled_label("Output", TOOLTIPS["output"]))
            output_image = gr.Image(type="pil", height=512, show_download_button=True, show_label=False)
            output_meta = gr.JSON(label="Meta", value={})
    return dict(
        prompt=prompt,
        input_image=input_image,
        preprocessor=preprocessor,
        controlnet_scale=controlnet_scale,
        steps=steps,
        seed=seed,
        lora_path=lora_path,
        lora_strength=lora_strength,
        generate_btn=generate_btn,
        output_image=output_image,
        output_meta=output_meta,
    )


def build_upscale_tab() -> dict[str, gr.components.Component]:
    with gr.Row():
        with gr.Column(scale=4):
            gr.HTML(labeled_label("Refinement prompt", TOOLTIPS["prompt"]))
            prompt = gr.Textbox(value="masterpiece, 8k", lines=2, show_label=False)
            gr.HTML(labeled_label("Input image", TOOLTIPS["upscale_image"]))
            input_image = gr.Image(type="pil", height=240, show_label=False)
            with gr.Row():
                with gr.Column():
                    gr.HTML(labeled_label("Refine steps", TOOLTIPS["refine_steps"]))
                    refine_steps = gr.Slider(1, 20, value=5, step=1, show_label=False)
                with gr.Column():
                    gr.HTML(labeled_label("Refine denoise", TOOLTIPS["refine_denoise"]))
                    refine_denoise = gr.Slider(0.0, 1.0, value=0.33, step=0.01, show_label=False)
            with gr.Row():
                with gr.Column():
                    gr.HTML(labeled_label("LoRA (optional)", TOOLTIPS["lora"]))
                    lora_path = gr.File(file_types=[".safetensors"], type="filepath", show_label=False)
                with gr.Column():
                    gr.HTML(labeled_label("LoRA strength", TOOLTIPS["lora_strength"]))
                    lora_strength = gr.Slider(0.0, 1.5, value=0.8, step=0.05, show_label=False)
            gr.HTML(labeled_label("Seed (0 = random)", TOOLTIPS["seed"]))
            seed = gr.Number(value=0, precision=0, show_label=False)
            generate_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=5):
            gr.HTML(labeled_label("Output (2x upscaled)", TOOLTIPS["output"]))
            output_image = gr.Image(type="pil", height=512, show_download_button=True, show_label=False)
            output_meta = gr.JSON(label="Meta", value={})
    return dict(
        prompt=prompt,
        input_image=input_image,
        refine_steps=refine_steps,
        refine_denoise=refine_denoise,
        seed=seed,
        lora_path=lora_path,
        lora_strength=lora_strength,
        generate_btn=generate_btn,
        output_image=output_image,
        output_meta=output_meta,
    )
