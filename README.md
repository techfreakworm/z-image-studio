---
title: Z-Image Studio
emoji: ⚡
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
python_version: "3.11"
suggested_hardware: zero-a10g
hf_oauth: false
preload_from_hub:
  - Tongyi-MAI/Z-Image transformer/*,text_encoder/*,vae/*,tokenizer/*,scheduler/*,model_index.json
  - Tongyi-MAI/Z-Image-Turbo transformer/*,scheduler/*,model_index.json
  - alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors
  - lllyasviel/Annotators RealESRGAN_x4plus.pth
---

# Z-Image Studio

A single-process Gradio app that wraps [Tongyi-MAI Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) and [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) with ControlNet and a 2× upscaler under one focused UI. Runs locally on Apple Silicon (MPS) or NVIDIA (CUDA), deploys to Hugging Face Spaces (ZeroGPU).

[![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Live-FFB02E?style=flat-square)](https://huggingface.co/spaces/techfreakworm/z-image-studio)
[![GitHub stars](https://img.shields.io/github/stars/techfreakworm/z-image-studio?style=flat-square&color=FFB02E)](https://github.com/techfreakworm/z-image-studio/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-FFB02E?style=flat-square)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-FFB02E?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Backed by DiffSynth-Studio](https://img.shields.io/badge/backend-DiffSynth--Studio-FFB02E?style=flat-square)](https://github.com/modelscope/DiffSynth-Studio)

→ **Live demo:** https://huggingface.co/spaces/techfreakworm/z-image-studio

---

## What's inside

Three tabs. Same DiffSynth `ZImagePipeline` underneath. Progressive disclosure — the form starts short and reveals controls only when you ask for them.

| Mode | Model | What it does |
|---|---|---|
| **Text → Image** | Z-Image (25 steps, cfg=4) · Z-Image-Turbo (8 steps, cfg=1) | Prompt-to-image. Toggle the model on the fly; the form swaps Steps / CFG / Negative-Prompt defaults to match. |
| **ControlNet** | Z-Image-Turbo + Fun-Controlnet-Union 2.1 | Canny / Depth / Pose preprocessors with a **live preview** of the processed control image. |
| **Upscale** | RealESRGAN x4 → Z-Image-Turbo refinement | Effective 2× upscale with diffusion-based detail restoration (5-step img2img at denoise 0.33). |

Each tab carries an optional LoRA toggle. When enabled, exposes a compact `.safetensors` slot + strength slider. The toggle label tells you which model's LoRA is accepted (Z-Image vs Z-Image-Turbo) and updates as you flip the radio.

---

## Quick start (local)

Requires **Python 3.11**, ~50 GB free disk for the weight set, and ~24 GB VRAM (CUDA) or ~32 GB unified memory (Apple Silicon).

```bash
git clone https://github.com/techfreakworm/z-image-studio
cd z-image-studio
bash setup.sh           # creates .venv, installs requirements
source .venv/bin/activate
python app.py           # http://127.0.0.1:7860
```

The first run resolves model weights into your HF cache (`~/.cache/huggingface/hub/`). Subsequent starts are fast — the app symlinks the cache snapshots into DiffSynth's expected `./models/<repo>/` layout so nothing re-downloads.

**Apple Silicon notes:** `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically so the few MPS-unsupported ops fall back to CPU. DiffSynth's free-VRAM check (CUDA-only) is bypassed on MPS — module swapping still works.

## Quick start (HF Spaces)

```bash
git remote add space https://huggingface.co/spaces/<your-handle>/z-image-studio
git push space main
```

The Space's `preload_from_hub` directive pre-downloads the ~47 GB weight set at build time. `app.py:_bootstrap()` mirrors the read-only build cache into `~/hf-cache-rw/` and symlinks every snapshot into `./models/<repo>/`. Pipeline construction at first request finds everything locally; no network on inference 2 onward.

## Architecture

```
                ┌──────────────────────────────┐
   browser ──▶  │   app.py — Gradio Blocks     │
                │   (header + CTA + 3 tabs)    │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │   backend.py                 │
                │   ZImageStudioBackend        │
                │   @spaces.GPU(duration=…)    │
                │   one DiffSynth pipeline,    │
                │   two transformers in pool   │
                └──────────────┬───────────────┘
                               │
   ┌───────────────┬───────────┴────────┬──────────────────┐
   ▼               ▼                    ▼                  ▼
modes.py     preprocessors.py     upscale.py          lora.py
3 handlers   Canny/Depth/Pose     RealESRGAN x4       safetensors
             (controlnet_aux)     + 0.5 resize        sniff + apply/revert
```

**One pipeline instance**, both transformers (Base + Turbo) preloaded into the pool, swapped per request by indexing into `pool.model`. Shared encoder + VAE + tokenizer between Base and Turbo — no duplication.

`@spaces.GPU(duration=callable)` decorates the generate method at module load time on Spaces. The duration estimator clamps to `[60, 180] s` based on mode, model, steps, and image area. ZeroGPU "GPU task aborted" surfaces auto-retry once at 2× duration.

## Project layout

```
.
├── app.py              # Gradio Blocks entry, bootstrap, event handlers, CTA
├── backend.py          # ZImageStudioBackend; @spaces.GPU; duration estimator
├── modes.py            # call_t2i / call_controlnet / call_upscale pure handlers
├── models.py           # device autodetect, MODEL_CONFIGS, cache mirror + symlink
├── lora.py             # safetensors header sniff + apply/revert ctx
├── preprocessors.py    # Canny (cv2) + Depth (depth_midas) + Pose (openpose)
├── upscale.py          # RealESRGAN x4 wrapper + basicsr/torchvision shim
├── ui.py               # Per-tab Gradio component builders
├── theme.py            # Soft Dark Restraint palette + minimal CSS
├── tooltips.py         # Centralised info= strings
├── requirements.txt    # pinned deps
├── pyproject.toml      # ruff + pytest config (py311)
├── setup.sh            # venv bootstrap
└── tests/              # 70 passing (L1+L2 in CI); GPU smoke in -m gpu
```

## Tech stack

- **[Gradio 5.50](https://gradio.app/)** — UI shell, native components, `gr.Progress(track_tqdm=True)`
- **[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)** — Z-Image pipeline + model pool + VRAM management
- **[Z-Image / Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image)** by Tongyi-MAI
- **[Z-Image-Turbo-Fun-Controlnet-Union-2.1](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1)** by Alibaba PAI
- **[RealESRGAN](https://github.com/xinntao/Real-ESRGAN)** weights via [`lllyasviel/Annotators`](https://huggingface.co/lllyasviel/Annotators)
- **[controlnet_aux](https://github.com/huggingface/controlnet_aux)** for Depth (MiDaS) and Pose (OpenPose)
- **HF Spaces ZeroGPU** (A10G) — `@spaces.GPU(duration=…)` queue priority

## Design

Theme: **Soft Dark Restraint** — warm dark substrate `#1A1614`, cream ink `#F0E8DD`, one accent `#FFB02E` used sparingly (live radio dot, slider fill, primary button, progress fill, brand period). Inter throughout. No display fonts, no shadows, no gradients. The accent is rationed so the generated image stays the visual focus.

Disclosure patterns — controls appear when they're needed:

- `Use a LoRA` checkbox → file slot + strength slider appear inline
- Model = Base → Negative Prompt + CFG slider appear (Turbo runs cfg=1 so they'd be no-ops)
- `Advanced` accordion → Width / Height / Seed live inside, collapsed by default

Spec + plan + design rationale live under `docs/superpowers/`.

## Notes on running

- **First inference is slow.** Cold-start pipeline construction (~30 – 60 s on MPS, ~10 – 20 s on CUDA) is amortised across the whole session. Subsequent requests hit warm cache.
- **MPS Macs:** Z-Image-Turbo at 8 steps + 1024² produces an image in ~30 – 60 s. Base at 25 steps is closer to 2 min. Upscale on 1024² → 2048² adds ~30 s on the refinement pass.
- **ZeroGPU duration cap.** The estimator clamps at 180 s. If a generation aborts, the handler retries once at 2× duration. The duration field per call is the queue-priority signal, not a billing cap.

## License

MIT for the app code (see `LICENSE`). DiffSynth-Studio is Apache-2.0. Z-Image and Z-Image-Turbo retain their respective Tongyi-MAI licenses. RealESRGAN weights are BSD-3-Clause via the xinntao/Real-ESRGAN repository.

## Credits

Z-Image and Z-Image-Turbo by [Tongyi-MAI](https://github.com/Tongyi-MAI). DiffSynth-Studio by the [ModelScope](https://github.com/modelscope) team. ControlNet Union 2.1 by [Alibaba PAI](https://github.com/alibaba). Built by [@techfreakworm](https://huggingface.co/techfreakworm) — drop a ♥ on the [Space](https://huggingface.co/spaces/techfreakworm/z-image-studio) if it's useful.
