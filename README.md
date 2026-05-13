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

# z-image-studio

Gradio app for [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) and [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) wrapping three modes under a single, focused UI:

1. **Text → Image** — pick Base (25 steps, cfg=4) or Turbo (8 steps, cfg=1)
2. **ControlNet** — Z-Image-Turbo-Fun-Controlnet-Union-2.1 with Canny / Depth / Pose preprocessors
3. **Upscale** — RealESRGAN x4 + Z-Image-Turbo img2img refinement (effective 2× with detail restoration)

Each tab supports an optional LoRA upload + strength slider. Runs on Apple Silicon (MPS) or NVIDIA (CUDA) locally, deploys to Hugging Face Spaces (ZeroGPU H200).

## Local quickstart

Requires Python 3.11 and ~35 GB free disk for model weights.

```bash
git clone https://github.com/<your-handle>/z-image-studio
cd z-image-studio
bash setup.sh
source .venv/bin/activate
python app.py
```

First run downloads ~30 GB into `~/.cache/huggingface/hub` (one-time). Subsequent starts are fast.

## HF Spaces deployment

```bash
git remote add space https://huggingface.co/spaces/<your-handle>/z-image-studio
git push space main
```

The Space's `preload_from_hub` directive pre-downloads the weights at build time; the `_bootstrap()` in `app.py` mirrors them into a writable tree at runtime.

## License

MIT for the app code. DiffSynth-Studio (Apache-2.0), Z-Image, and RealESRGAN retain their respective licenses.
