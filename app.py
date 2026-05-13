"""z-image-studio — Gradio entrypoint.

On HF Spaces, ``_bootstrap`` runs once on import to mirror the read-only preload
cache into a writable tree.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import gradio as gr

import backend
import lora as lora_mod  # avoid shadowing the gr.File `lora_path` name
import models
import theme
import ui


# ----- HF Spaces bootstrap ---------------------------------------------------

def _bootstrap() -> None:
    """Mirror the preload_from_hub cache once, then point HF env at the mirror."""
    if not models.on_spaces():
        return
    src = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    dst = Path.home() / "hf-cache-rw"
    models.mirror_preload_hf_cache(src, dst)
    os.environ["HF_HOME"] = str(dst)
    os.environ["HF_HUB_CACHE"] = str(dst / "hub")


_bootstrap()


# ----- Eager backend boot ----------------------------------------------------

_BACKEND: backend.ZImageStudioBackend | None = None


def get_backend() -> backend.ZImageStudioBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = backend.ZImageStudioBackend()
    return _BACKEND


# ----- Generation event handlers --------------------------------------------

def _maybe_random_seed(seed: int) -> int:
    return seed if seed and seed > 0 else random.randint(1, 2_147_483_647)


def _coerce_lora(lora_path: str | None) -> Path | None:
    if not lora_path:
        return None
    p = Path(lora_path)
    lora_mod.sniff(p)  # validate cheaply; raises LoRAValidationError if bad
    return p


def _esrgan_path() -> str:
    """Locate the preloaded RealESRGAN_x4plus.pth."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download("xinntao/Real-ESRGAN", "RealESRGAN_x4plus.pth")


def on_t2i_generate(prompt, negative_prompt, model, steps, cfg,
                    width, height, seed, lora_path, lora_strength):
    try:
        lora_p = _coerce_lora(lora_path)
    except lora_mod.LoRAValidationError as e:
        raise gr.Error(str(e)) from e

    params = dict(
        prompt=prompt, negative_prompt=negative_prompt or "",
        model=model, steps=int(steps), cfg=float(cfg),
        width=int(width), height=int(height),
        seed=_maybe_random_seed(int(seed)),
        lora_path=lora_p, lora_strength=float(lora_strength),
    )
    image, meta = get_backend().generate(mode="t2i", params=params)
    return image, meta


def on_controlnet_generate(prompt, input_image, preprocessor, controlnet_scale,
                           steps, seed, lora_path, lora_strength):
    try:
        lora_p = _coerce_lora(lora_path)
    except lora_mod.LoRAValidationError as e:
        raise gr.Error(str(e)) from e

    params = dict(
        prompt=prompt, input_image=input_image,
        preprocessor=preprocessor, controlnet_scale=float(controlnet_scale),
        steps=int(steps), seed=_maybe_random_seed(int(seed)),
        lora_path=lora_p, lora_strength=float(lora_strength),
    )
    image, meta = get_backend().generate(mode="controlnet", params=params)
    return image, meta


def on_upscale_generate(prompt, input_image, refine_steps, refine_denoise,
                        seed, lora_path, lora_strength):
    try:
        lora_p = _coerce_lora(lora_path)
    except lora_mod.LoRAValidationError as e:
        raise gr.Error(str(e)) from e

    params = dict(
        prompt=prompt or "masterpiece, 8k",
        input_image=input_image,
        refine_steps=int(refine_steps),
        refine_denoise=float(refine_denoise),
        seed=_maybe_random_seed(int(seed)),
        lora_path=lora_p, lora_strength=float(lora_strength),
        esrgan_model_path=_esrgan_path(),
    )
    image, meta = get_backend().generate(mode="upscale", params=params)
    return image, meta


# ----- Blocks ----------------------------------------------------------------

HEADER_HTML = """
<div style="display:flex;justify-content:space-between;align-items:baseline;padding:8px 0 4px 0;">
  <div style="font-family:'Geist',sans-serif;font-size:16px;font-weight:600;letter-spacing:-0.02em;">
    z<span style="color:#FFB02E;">·</span>image studio
  </div>
  <div class="zis-status">ready</div>
</div>
""".strip()


_HEAD_JS = """
<script>
window.zis = {
    setModel: function(name) {
        document.querySelectorAll('.zis-model').forEach(el => {
            el.classList.toggle('on', el.dataset.value === name);
        });
        const hidden = document.querySelector('#zis-model-state textarea, #zis-model-state input');
        if (hidden) {
            hidden.value = name;
            hidden.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }
};
// Tap-to-pin tooltips on mobile
document.addEventListener('touchstart', function(e) {
    const tip = e.target.closest('.zis-info');
    document.querySelectorAll('.zis-info.shown').forEach(el => {
        if (el !== tip) el.classList.remove('shown');
    });
    if (tip) tip.classList.toggle('shown');
}, { passive: true });
</script>
""".strip()


def build_app() -> gr.Blocks:
    with gr.Blocks(theme=theme.build_theme(), css=theme.CSS, head=_HEAD_JS, title="z-image-studio") as demo:
        gr.HTML(HEADER_HTML)

        with gr.Tabs():
            with gr.Tab("Text → Image"):
                t = ui.build_t2i_tab()
                t["generate_btn"].click(
                    fn=on_t2i_generate,
                    inputs=[t["prompt"], t["negative_prompt"], t["model_state"],
                            t["steps"], t["cfg"], t["width"], t["height"], t["seed"],
                            t["lora_path"], t["lora_strength"]],
                    outputs=[t["output_image"], t["output_meta"]],
                )

            with gr.Tab("ControlNet"):
                c = ui.build_controlnet_tab()
                c["generate_btn"].click(
                    fn=on_controlnet_generate,
                    inputs=[c["prompt"], c["input_image"],
                            c["preprocessor"], c["controlnet_scale"],
                            c["steps"], c["seed"], c["lora_path"], c["lora_strength"]],
                    outputs=[c["output_image"], c["output_meta"]],
                )

            with gr.Tab("Upscale"):
                u = ui.build_upscale_tab()
                u["generate_btn"].click(
                    fn=on_upscale_generate,
                    inputs=[u["prompt"], u["input_image"],
                            u["refine_steps"], u["refine_denoise"],
                            u["seed"], u["lora_path"], u["lora_strength"]],
                    outputs=[u["output_image"], u["output_meta"]],
                )
    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
