"""Tooltip strings for every labeled UI component.

Kept separate from ``ui.py`` so copy edits don't touch component wiring. Every
key here MUST be referenced from a labeled component in ``ui.py`` (and vice versa).
"""
from __future__ import annotations

TOOLTIPS: dict[str, str] = {
    "prompt":                  "What to generate. Be specific: subject, style, lighting, camera angle.",
    "negative_prompt":         "What to avoid (Base only). e.g. 'blurry, low quality, distorted'.",
    "model":                   "Base = 25 steps, higher quality. Turbo = 8 steps, fast.",
    "lora":                    "Optional .safetensors LoRA file. Trained on Z-Image base or turbo.",
    "lora_strength":           "LoRA influence. 0.6–1.0 typical. Higher = more LoRA, less base model.",
    "steps":                   "Denoising steps. Turbo: 6–10. Base: 20–30. More = better detail, slower.",
    "cfg":                     "Classifier-free guidance. Turbo: locked at 1.0. Base: 3–5 typical.",
    "width":                   "Output width in pixels. Multiples of 64. Higher = more memory.",
    "height":                  "Output height in pixels. Multiples of 64.",
    "seed":                    "0 = random each run. Pin a number to reproduce an image exactly.",
    "controlnet_image":        "Control image — the structural reference for the output.",
    "controlnet_preprocessor": "Canny = edges, Depth = depth map, Pose = body pose, Pre-processed = use image as-is.",
    "controlnet_scale":        "How strongly the control image guides the output. 0.6–1.2 typical.",
    "upscale_image":           "Input image to upscale 2x.",
    "refine_steps":            "Steps for the Z-Image-Turbo refinement pass after RealESRGAN. 3–8 typical.",
    "refine_denoise":          "How much the refinement alters pixels. 0.2–0.4 typical. Higher = more detail change.",
    "output":                  "Generated image. Right-click to download full resolution.",
}
