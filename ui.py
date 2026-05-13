"""Gradio UI builders for z-image-studio (Soft Dark Restraint redesign).

Each ``build_*_tab()`` returns a dict of components so ``app.py:build_app``
can wire ``.click()`` / ``.change()`` handlers without reaching into local
scopes. All param help text flows through Gradio's native ``info=`` parameter
(dim subtitle under each label) — no custom popover or (i) icon helper.
"""

from __future__ import annotations

import gradio as gr

import preprocessors
from tooltips import TOOLTIPS

# Link targets for the small dim row under the model radio. The Z-Image
# README's Model Zoo section is the canonical "where to find more models"
# anchor; Omni Base lives in the same README. When more models ship, swap
# these constants and nothing else needs to change.
MODEL_ZOO_URL = "https://github.com/Tongyi-MAI/Z-Image#-model-zoo"


def _model_soon_row_html() -> str:
    """Return the small dim link row that lives directly under the model radio.

    Two anchor links + a "(coming soon)" qualifier. Static — no state, no JS.
    """
    return (
        '<div class="zis-soon-row">'
        f'<a href="{MODEL_ZOO_URL}" target="_blank" rel="noopener noreferrer">Edit ↗</a>'
        '<span class="sep">·</span>'
        f'<a href="{MODEL_ZOO_URL}" target="_blank" rel="noopener noreferrer">Omni Base ↗</a>'
        '<span class="dim">(coming soon)</span>'
        "</div>"
    )


def build_t2i_tab() -> dict[str, gr.components.Component]:
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(
                label="Prompt",
                info=TOOLTIPS["prompt"],
                lines=4,
                placeholder="A latina model peeking through pine branches…",
            )

            model = gr.Radio(
                ["Base", "Turbo"],
                value="Turbo",
                label="Model",
                info=TOOLTIPS["model"],
            )
            model_soon_row = gr.HTML(_model_soon_row_html())

            with gr.Group(visible=False) as base_group:
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    info=TOOLTIPS["negative_prompt"],
                    lines=2,
                    placeholder="blurry, lowres, distorted",
                )
                cfg = gr.Slider(
                    0.5,
                    12.0,
                    value=1.0,
                    step=0.1,
                    label="CFG",
                    info=TOOLTIPS["cfg"],
                )

            lora_enabled = gr.Checkbox(label="Use a LoRA (compatible with Z-Image-Turbo)", value=False)
            with gr.Group(visible=False) as lora_group:
                lora_path = gr.File(
                    label="LoRA file",
                    file_types=[".safetensors"],
                    type="filepath",
                    elem_classes=["zis-lora-file"],
                )
                lora_strength = gr.Slider(
                    0.0,
                    1.5,
                    value=0.8,
                    step=0.05,
                    label="LoRA strength",
                    info=TOOLTIPS["lora_strength"],
                )

            steps = gr.Slider(1, 50, value=8, step=1, label="Steps", info=TOOLTIPS["steps"])

            with gr.Accordion("Advanced", open=False):
                width = gr.Slider(384, 1536, value=1024, step=64, label="Width", info=TOOLTIPS["width"])
                height = gr.Slider(384, 1536, value=1024, step=64, label="Height", info=TOOLTIPS["height"])
                seed = gr.Number(value=0, precision=0, label="Seed", info=TOOLTIPS["seed"])

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=5):
            gr.Markdown(f"**Output**  \n<span style='color:#988B7C;font-size:12px;'>{TOOLTIPS['output']}</span>")
            output_image = gr.Image(
                show_label=False,
                type="pil",
                height=512,
                show_download_button=True,
            )
            output_meta = gr.JSON(label="Meta", value={})

    return dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model=model,
        model_soon_row=model_soon_row,
        base_group=base_group,
        lora_enabled=lora_enabled,
        lora_group=lora_group,
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
            prompt = gr.Textbox(
                label="Prompt",
                info=TOOLTIPS["prompt"],
                lines=3,
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        f"**Control image**  \n<span style='color:#988B7C;font-size:12px;'>"
                        f"{TOOLTIPS['controlnet_image']}</span>"
                    )
                    input_image = gr.Image(
                        show_label=False,
                        type="pil",
                        height=240,
                    )
                with gr.Column():
                    gr.Markdown(
                        "**Preprocessor preview**  \n<span style='color:#988B7C;font-size:12px;'>"
                        "Live edge map / depth map / pose. Updates as you change the preprocessor.</span>"
                    )
                    preview_image = gr.Image(
                        show_label=False,
                        type="pil",
                        height=240,
                        interactive=False,
                    )

            with gr.Row():
                preprocessor = gr.Dropdown(
                    list(preprocessors.MODES),
                    value="Canny",
                    label="Preprocessor",
                    info=TOOLTIPS["controlnet_preprocessor"],
                )
                controlnet_scale = gr.Slider(
                    0.0,
                    2.0,
                    value=1.0,
                    step=0.05,
                    label="ControlNet scale",
                    info=TOOLTIPS["controlnet_scale"],
                )

            lora_enabled = gr.Checkbox(label="Use a LoRA (compatible with Z-Image-Turbo)", value=False)
            with gr.Group(visible=False) as lora_group:
                lora_path = gr.File(
                    label="LoRA file",
                    file_types=[".safetensors"],
                    type="filepath",
                    elem_classes=["zis-lora-file"],
                )
                lora_strength = gr.Slider(
                    0.0,
                    1.5,
                    value=0.8,
                    step=0.05,
                    label="LoRA strength",
                    info=TOOLTIPS["lora_strength"],
                )

            steps = gr.Slider(1, 30, value=9, step=1, label="Steps", info=TOOLTIPS["steps"])

            with gr.Accordion("Advanced", open=False):
                seed = gr.Number(value=0, precision=0, label="Seed", info=TOOLTIPS["seed"])

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=5):
            gr.Markdown(f"**Output**  \n<span style='color:#988B7C;font-size:12px;'>{TOOLTIPS['output']}</span>")
            output_image = gr.Image(
                show_label=False,
                type="pil",
                height=512,
                show_download_button=True,
            )
            output_meta = gr.JSON(label="Meta", value={})

    return dict(
        prompt=prompt,
        input_image=input_image,
        preview_image=preview_image,
        preprocessor=preprocessor,
        controlnet_scale=controlnet_scale,
        steps=steps,
        seed=seed,
        lora_enabled=lora_enabled,
        lora_group=lora_group,
        lora_path=lora_path,
        lora_strength=lora_strength,
        generate_btn=generate_btn,
        output_image=output_image,
        output_meta=output_meta,
    )


def build_upscale_tab() -> dict[str, gr.components.Component]:
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(
                label="Refinement prompt",
                info=TOOLTIPS["prompt"],
                value="masterpiece, 8k",
                lines=2,
            )
            gr.Markdown(
                f"**Input image**  \n<span style='color:#988B7C;font-size:12px;'>{TOOLTIPS['upscale_image']}</span>"
            )
            input_image = gr.Image(
                show_label=False,
                type="pil",
                height=240,
            )

            with gr.Row():
                refine_steps = gr.Slider(
                    1,
                    20,
                    value=5,
                    step=1,
                    label="Refine steps",
                    info=TOOLTIPS["refine_steps"],
                )
                refine_denoise = gr.Slider(
                    0.0,
                    1.0,
                    value=0.33,
                    step=0.01,
                    label="Refine denoise",
                    info=TOOLTIPS["refine_denoise"],
                )

            lora_enabled = gr.Checkbox(label="Use a LoRA (compatible with Z-Image-Turbo)", value=False)
            with gr.Group(visible=False) as lora_group:
                lora_path = gr.File(
                    label="LoRA file",
                    file_types=[".safetensors"],
                    type="filepath",
                    elem_classes=["zis-lora-file"],
                )
                lora_strength = gr.Slider(
                    0.0,
                    1.5,
                    value=0.8,
                    step=0.05,
                    label="LoRA strength",
                    info=TOOLTIPS["lora_strength"],
                )

            with gr.Accordion("Advanced", open=False):
                seed = gr.Number(value=0, precision=0, label="Seed", info=TOOLTIPS["seed"])

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=5):
            gr.Markdown(
                f"**Output (2x upscaled)**  \n<span style='color:#988B7C;font-size:12px;'>{TOOLTIPS['output']}</span>"
            )
            output_image = gr.Image(
                show_label=False,
                type="pil",
                height=512,
                show_download_button=True,
            )
            output_meta = gr.JSON(label="Meta", value={})

    return dict(
        prompt=prompt,
        input_image=input_image,
        refine_steps=refine_steps,
        refine_denoise=refine_denoise,
        seed=seed,
        lora_enabled=lora_enabled,
        lora_group=lora_group,
        lora_path=lora_path,
        lora_strength=lora_strength,
        generate_btn=generate_btn,
        output_image=output_image,
        output_meta=output_meta,
    )
