"""z-image-studio — Gradio entrypoint.

On HF Spaces, ``_bootstrap`` runs once on import to mirror the read-only preload
cache into a writable tree.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

# DiffSynth defaults to ModelScope; force HF so preload_from_hub + HF cache work.
# Must be set before any diffsynth import path is taken (backend imports it lazily).
os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")

# Apple Silicon: let PyTorch fall back to CPU for the small set of ops MPS doesn't
# implement (some scaled-dot-product flavors, certain index ops). Without this,
# DiffSynth crashes mid-pipeline on the first unsupported op rather than degrading.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gradio as gr

import backend
import lora as lora_mod
import models
import preprocessors
import theme
import ui

# ----- HF Spaces bootstrap ---------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parent
_DIFFSYNTH_MODELS_DIR = _REPO_ROOT / "models"


def _bootstrap() -> None:
    """Mirror the preload_from_hub cache, then symlink snapshots into DiffSynth's
    expected ``./models/<repo>/`` layout so the pipeline reuses preloaded weights
    instead of re-downloading on first call.

    On Spaces: cache is read-only owned by the build user → mirror to ~/hf-cache-rw
    first, then point HF env there, then symlink into ./models.

    Locally: skip the mirror (we own the dirs); just symlink from the user's HF
    cache to ./models so DiffSynth finds the snapshots.
    """
    if models.on_spaces():
        src = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        dst = Path.home() / "hf-cache-rw"
        models.mirror_preload_hf_cache(src, dst)
        os.environ["HF_HOME"] = str(dst)
        os.environ["HF_HUB_CACHE"] = str(dst / "hub")
        cache_hub = dst / "hub"
    else:
        cache_hub = Path(os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub")))

    # Point DiffSynth at our project-local models dir + symlink every cached
    # snapshot so DiffSynth's ModelConfig finds them without re-downloading.
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(_DIFFSYNTH_MODELS_DIR))
    _DIFFSYNTH_MODELS_DIR.mkdir(exist_ok=True)
    models.symlink_hf_cache_to_diffsynth_layout(cache_hub, _DIFFSYNTH_MODELS_DIR)


_bootstrap()


# ----- Lazy backend singleton ------------------------------------------------

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


def _on_model_change(model_name: str):
    """When the user picks Base / Turbo in the radio, update steps + cfg defaults
    and the LoRA-compatibility hint on the toggle label."""
    if model_name == "Base":
        return 25, 4.0, gr.update(label="Use a LoRA (compatible with Z-Image)")
    return 8, 1.0, gr.update(label="Use a LoRA (compatible with Z-Image-Turbo)")


def _preview_cn(image, mode):
    """Render the live preprocessor preview next to the input on the ControlNet tab.

    Wrapped in ``try/except`` so that a missing optional dep (controlnet_aux for
    Depth / Pose) never breaks the form — it just falls back to the raw input.
    We DO log the exception to stderr so the next failure surfaces in the logs
    rather than silently showing the user the original image (the previous
    behavior hid a typo'd processor name for weeks).
    """
    if image is None:
        return None
    try:
        return preprocessors.run(mode, image)
    except Exception as e:
        import sys
        import traceback

        print(f"[preview_cn] {mode!r} failed: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return image


def _esrgan_path() -> str:
    """Locate the preloaded RealESRGAN_x4plus.pth."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download("lllyasviel/Annotators", "RealESRGAN_x4plus.pth")


def on_t2i_generate(
    prompt,
    negative_prompt,
    model,
    steps,
    cfg,
    width,
    height,
    seed,
    lora_enabled,
    lora_path,
    lora_strength,
    progress=gr.Progress(track_tqdm=True),  # noqa: B008
):
    if not lora_enabled:
        lora_path = None
    try:
        lora_p = _coerce_lora(lora_path)
    except lora_mod.LoRAValidationError as e:
        raise gr.Error(str(e)) from e

    params = dict(
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        model=model,
        steps=int(steps),
        cfg=float(cfg),
        width=int(width),
        height=int(height),
        seed=_maybe_random_seed(int(seed)),
        lora_path=lora_p,
        lora_strength=float(lora_strength),
    )
    image, meta = backend.generate_with_retry(get_backend(), mode="t2i", params=params)
    return image, meta


def on_controlnet_generate(
    prompt,
    input_image,
    preprocessor,
    controlnet_scale,
    steps,
    seed,
    lora_enabled,
    lora_path,
    lora_strength,
    progress=gr.Progress(track_tqdm=True),  # noqa: B008
):
    if not lora_enabled:
        lora_path = None
    try:
        lora_p = _coerce_lora(lora_path)
    except lora_mod.LoRAValidationError as e:
        raise gr.Error(str(e)) from e

    params = dict(
        prompt=prompt,
        input_image=input_image,
        preprocessor=preprocessor,
        controlnet_scale=float(controlnet_scale),
        steps=int(steps),
        seed=_maybe_random_seed(int(seed)),
        lora_path=lora_p,
        lora_strength=float(lora_strength),
    )
    image, meta = backend.generate_with_retry(get_backend(), mode="controlnet", params=params)
    return image, meta


def on_upscale_generate(
    prompt,
    input_image,
    refine_steps,
    refine_denoise,
    seed,
    progress=gr.Progress(track_tqdm=True),  # noqa: B008
):
    params = dict(
        prompt=prompt or "masterpiece, 8k",
        input_image=input_image,
        refine_steps=int(refine_steps),
        refine_denoise=float(refine_denoise),
        seed=_maybe_random_seed(int(seed)),
        esrgan_model_path=_esrgan_path(),
    )
    image, meta = backend.generate_with_retry(get_backend(), mode="upscale", params=params)
    return image, meta


# ----- Blocks ----------------------------------------------------------------

HEADER_HTML = """
<div style="display:flex;justify-content:space-between;align-items:baseline;padding:8px 0 4px 0;">
  <div style="font-size:16px;font-weight:600;letter-spacing:-0.01em;">
    Z-Image Studio<span class="zis-brand-period">.</span>
  </div>
  <div class="zis-status-dot" style="font-size:11px;color:#988B7C;letter-spacing:0.02em;">ready</div>
</div>
""".strip()


CTA_HTML = """
<div class="zis-cta">
  Built with care.
  <strong>Drop a <span class="zis-cta-heart">♥</span> at the top</strong> to support it
  <span class="zis-cta-sep">·</span>
  Follow <a href="https://huggingface.co/techfreakworm" target="_blank" rel="noopener noreferrer">@techfreakworm</a>
  for what's next.
</div>
""".strip()


def build_app() -> gr.Blocks:
    with gr.Blocks(theme=theme.build_theme(), css=theme.CSS, title="Z-Image Studio") as demo:
        gr.HTML(HEADER_HTML)
        gr.HTML(CTA_HTML)

        with gr.Tabs():
            with gr.Tab("Text → Image"):
                t = ui.build_t2i_tab()
                t["generate_btn"].click(
                    fn=on_t2i_generate,
                    inputs=[
                        t["prompt"],
                        t["negative_prompt"],
                        t["model"],
                        t["steps"],
                        t["cfg"],
                        t["width"],
                        t["height"],
                        t["seed"],
                        t["lora_enabled"],
                        t["lora_path"],
                        t["lora_strength"],
                    ],
                    outputs=[t["output_image"], t["output_meta"]],
                )
                # Radio change → update step / cfg defaults + LoRA-compatibility hint
                # on the toggle label + reveal Base-only fields.
                t["model"].change(
                    fn=_on_model_change,
                    inputs=[t["model"]],
                    outputs=[t["steps"], t["cfg"], t["lora_enabled"]],
                )
                t["model"].change(
                    fn=lambda m: gr.Group(visible=(m == "Base")),
                    inputs=[t["model"]],
                    outputs=[t["base_group"]],
                )
                # LoRA checkbox → reveal file + strength.
                t["lora_enabled"].change(
                    fn=lambda v: gr.Group(visible=v),
                    inputs=[t["lora_enabled"]],
                    outputs=[t["lora_group"]],
                )

            with gr.Tab("ControlNet"):
                c = ui.build_controlnet_tab()
                c["generate_btn"].click(
                    fn=on_controlnet_generate,
                    inputs=[
                        c["prompt"],
                        c["input_image"],
                        c["preprocessor"],
                        c["controlnet_scale"],
                        c["steps"],
                        c["seed"],
                        c["lora_enabled"],
                        c["lora_path"],
                        c["lora_strength"],
                    ],
                    outputs=[c["output_image"], c["output_meta"]],
                )
                # Live preprocessor preview — fires on input change or mode change.
                c["input_image"].change(
                    fn=_preview_cn,
                    inputs=[c["input_image"], c["preprocessor"]],
                    outputs=[c["preview_image"]],
                )
                c["preprocessor"].change(
                    fn=_preview_cn,
                    inputs=[c["input_image"], c["preprocessor"]],
                    outputs=[c["preview_image"]],
                )
                # LoRA checkbox → reveal file + strength.
                c["lora_enabled"].change(
                    fn=lambda v: gr.Group(visible=v),
                    inputs=[c["lora_enabled"]],
                    outputs=[c["lora_group"]],
                )

            with gr.Tab("Upscale"):
                u = ui.build_upscale_tab()
                u["generate_btn"].click(
                    fn=on_upscale_generate,
                    inputs=[
                        u["prompt"],
                        u["input_image"],
                        u["refine_steps"],
                        u["refine_denoise"],
                        u["seed"],
                    ],
                    outputs=[u["output_image"], u["output_meta"]],
                )
    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
