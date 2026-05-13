# z-image-studio — Design Spec

| Field        | Value                                                                                  |
| ------------ | -------------------------------------------------------------------------------------- |
| Date         | 2026-05-13                                                                             |
| Status       | Draft for review                                                                       |
| Owner        | Mayank Gupta                                                                           |
| Brainstorm   | `.superpowers/brainstorm/2440-1778660739/` (UI mockups Plate/Console/Atelier → Onyx variants → Amber chosen) |

A single-process Gradio 5.x app that runs the Z-Image and Z-Image-Turbo diffusion models with three task tabs (Text → Image, ControlNet, Upscale) plus a per-tab LoRA loader. The same code runs locally (auto-detected MPS on Apple Silicon, CUDA on NVIDIA) and on Hugging Face Spaces (ZeroGPU H200). Backend is `DiffSynth-Studio`'s `ZImagePipeline` — no ComfyUI bundling, no JSON workflows, no PromptServer stubs.

---

## 1. Goals · Non-goals

**Goals (v1)**
- Public HF Space exposing Z-Image base + Turbo with a model selector
- Three tabs: T2I (dual-model), ControlNet (Turbo + Union 2.1, all preprocessor modes), Upscale (RealESRGAN + Z-Image-Turbo refinement)
- LoRA upload + strength slider in each tab
- Eager preload of both transformers so model-switching in T2I is near-instant
- Local development on Apple Silicon MPS that matches HF behavior bit-for-bit

**Non-goals (v1)** — list locked, do not re-litigate during implementation
- Video / motion modes (this is image-only)
- Persistent storage add-on
- Output history persistence across sessions
- Multi-prompt queueing
- Custom LoRA add/remove rows (single LoRA per tab)
- LoRA on the Upscale refinement pass (locked to vanilla Turbo refinement)
- ControlNet on Z-Image base (no released ControlNet weights for base; Turbo-only)
- Gradio sidebar / drawer layout (LTX-style is overkill for 3 modes)

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ app.py — Gradio Blocks                                       │
│   Header (gr.HTML) · Tabs · gr.Image / gr.JSON outputs       │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│ backend.py — ZImageStudioBackend                             │
│   @spaces.GPU(duration=_duration_for)  ← module-level        │
│   ├─ pipeline: diffsynth.ZImagePipeline                      │
│   ├─ load_lora(safetensors, strength)   — apply/revert ctx   │
│   └─ generate(mode, params) → image, meta                    │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│ modes.py · preprocessors.py · upscale.py · lora.py · models.py │
└──────────────────────────────────────────────────────────────┘
```

Single pipeline instance shared across modes. The transformer swap (Base ↔ Turbo) is the only model-pool change — both already loaded on CPU. DiffSynth's `vram_management` swaps modules CPU↔GPU per call.

`@spaces.GPU` is the only divergence between local and Spaces — applied via a runtime check that yields an identity decorator off-Spaces.

---

## 3. Modes — DiffSynth call mappings

All three modes go through one `ZImagePipeline.__call__`. Mode-specific code is just argument shaping.

| Mode | Model | DiffSynth call shape | Source ComfyUI workflow |
| --- | --- | --- | --- |
| **T2I (Base)** | `Tongyi-MAI/Z-Image` | `pipe(prompt, negative_prompt, cfg_scale=4.0, num_inference_steps=25, sigma_shift=3.0, height, width, seed)` | `image_z_image.json` |
| **T2I (Turbo)** | `Tongyi-MAI/Z-Image-Turbo` | `pipe(prompt, cfg_scale=1.0, num_inference_steps=8, sigma_shift=3.0, height, width, seed)` | `image_z_image_turbo.json` |
| **ControlNet** | Turbo + `PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1` | `pipe(prompt, controlnet_inputs=[ControlNetInput(image=preprocessed, scale)], cfg_scale=1.0, num_inference_steps=9, sigma_shift=3.0)` | `image_z_image_turbo_fun_union_controlnet.json` |
| **Upscale** | Turbo + RealESRGAN_x4plus | `RealESRGAN_x4(input) → PIL.resize 0.5 → pipe(prompt="masterpiece, 8k", input_image=upscaled, denoising_strength=0.33, num_inference_steps=5, cfg_scale=1.0, sigma_shift=3.0)` | `utility_z_image_turbo_2k_upscaler.json` |

**LoRA wiring:** validated `safetensors` file + `gr.Slider(0.0, 1.5, value=0.8)` strength. Applied via DiffSynth's `merge_lora` inside an apply/revert context manager so the cached GPU model returns to a clean state after each request. Safetensors header sniffed before `@spaces.GPU` fires to reject mismatched LoRAs with a clear error (no GPU slot wasted).

**ControlNet preprocessors:** dropdown `["Canny", "Depth", "Pose", "Pre-processed (no-op)"]`. Backed by `controlnet_aux` with lazy imports (only the chosen preprocessor's deps load).

---

## 4. UI — Onyx Amber

Aesthetic locked from the brainstorm: warm dark, golden amber accent, Geist family. Reads as "studio at midnight" — not corporate, not gamer.

### 4.1 Color tokens

| Token              | Value      | Used for                                  |
| ------------------ | ---------- | ----------------------------------------- |
| `body_bg`          | `#0F0C08`  | App body                                  |
| `panel_bg`         | `#0F0C08`  | Panel background (same as body)           |
| `input_bg`         | `#0F0C08`  | Text inputs, sliders                      |
| `canvas_bg`        | `#110D08`  | Image preview block background            |
| `border`           | `#2A2218`  | All hairline borders                      |
| `text`             | `#FAF1E3`  | Primary text                              |
| `text_dim`         | `#A89478`  | Secondary text, labels                    |
| `accent`           | `#FFB02E`  | Active tab underline, primary button bg, radio-on bg, slider fill, LoRA tag |
| `accent_text`      | `#1A1208`  | Text on accent fills                      |
| `radius`           | `8px`      | Block radius, button radius               |
| `radius_sm`        | `6px`      | Radio pill radius                         |

### 4.2 Typography

- Display + body: **Geist** (variable, free; loaded via `@import url(https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&display=swap)`)
- Mono: **Geist Mono** (labels, numeric readouts, status line, LoRA file metadata)
- Wired through `gr.themes.Base(font=[gr.themes.GoogleFont("Geist"), "sans-serif"], font_mono=[gr.themes.GoogleFont("Geist Mono"), "monospace"])`

### 4.3 Decoration

- Subtle warm vignette at top of body: `background-image: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(255,176,46,0.06), transparent 70%)`
- Primary button glow: `box-shadow: 0 0 0 1px rgba(255,176,46,0.4), 0 8px 24px -8px rgba(255,176,46,0.35)`
- Image preview placeholder gradient (visible only before first generation): `radial-gradient(circle at 35% 25%, rgba(255,176,46,0.32), transparent 55%), linear-gradient(135deg, #2a1f0e, #573b0f, #FFB02E 115%)`

### 4.4 Component patterns (all Gradio-native shapes)

- **Tab strip** — top horizontal `gr.Tabs` with underline indicator in amber on the active tab.
- **Prompt** — `gr.Textbox(lines=4, label="Prompt")` — full width.
- **Model selector** — `gr.Radio(["Base", "Turbo"], value="Turbo", label="Model")` — T2I tab only.
- **LoRA loader** — `gr.File(label="LoRA", file_types=[".safetensors"])` + `gr.Slider(0.0, 1.5, value=0.8, label="LoRA strength", step=0.05)`. On phone they stack; on tablet+ they share a row.
- **Parameter sliders** — Steps, CFG (T2I-base only), Width, Height, Seed.
- **ControlNet-only** — additional `gr.Image(label="Control image")` + `gr.Dropdown(["Canny","Depth","Pose","Pre-processed"], value="Canny")` + `gr.Slider(0.0, 2.0, value=1.0, label="ControlNet scale")`.
- **Upscale-only** — additional `gr.Image(label="Input image")`; no model selector; the upscale prompt defaults to `"masterpiece, 8k"` and is editable.
- **Generate button** — `gr.Button("Generate", variant="primary")` — amber fill with glow.
- **Output** — `gr.Image(label="Output", show_download_button=True)` + `gr.JSON(label="Meta", value={...seed, steps, model, lora})`.
- **Status line** — `gr.Markdown` updated on generation start/end. Mono font, dim text.

### 4.5 Responsive behavior

Phone (< 600 px): single column. Controls then output. LoRA row stacks (file then strength). Tab labels truncate to `Text`, `ControlNet`, `Upscale`.

Tablet (600 – 1024 px): two-column grid below the tab strip. Controls on left, output on right. LoRA row goes side-by-side (file widget left, strength slider right).

Desktop (> 1024 px): same as tablet, but generous margins and a max-width on the controls column. No sidebar; tab strip remains horizontal.

Gradio's default `gr.Tabs` + `gr.Row`/`gr.Column` machinery already collapses on small viewports; we just need the CSS media query to swap from grid to stacked at < 600 px.

---

## 5. File layout

```
llm/z-image-studio/
├── app.py              # Gradio Blocks: tabs, header, launch
├── backend.py          # ZImageStudioBackend; @spaces.GPU decoration
├── modes.py            # 3 mode handlers (T2I, ControlNet, Upscale) — pure functions
├── models.py           # ModelConfig registry, preload helpers
├── preprocessors.py    # Canny / Depth / Pose via controlnet_aux (lazy)
├── upscale.py          # RealESRGAN x4 wrapper + 0.5-resize bridge
├── lora.py             # LoRA safetensors header sniffer + apply/revert ctx
├── ui.py               # Per-tab Gradio component builders
├── theme.py            # Amber tokens + gr.themes.Base subclass + CSS string
├── pyproject.toml      # ruff config; py311
├── requirements.txt    # diffsynth-studio, gradio==5.x, spaces, controlnet-aux, realesrgan, ...
├── README.md           # HF Space YAML frontmatter (preload_from_hub) + user docs
├── LICENSE             # MIT
├── CLAUDE.md           # Mirror sole-author rule + venv/no-conda + hf CLI
├── setup.sh            # python3.11 -m venv .venv; pip install
├── tests/
│   ├── conftest.py
│   ├── test_modes.py
│   ├── test_models.py
│   ├── test_preprocessors.py
│   └── test_lora.py
├── assets/             # small seed images for smoke tests
└── docs/superpowers/
    ├── specs/2026-05-13-z-image-studio-design.md   # this file
    └── plans/                                       # plan from writing-plans (next)
```

Flat top-level layout (no `src/`, no nested packages). Each module owns one responsibility. Same convention as LTX2.3-AIO-Generator.

---

## 6. Models, preload, cache mirroring

### 6.1 HF Space YAML frontmatter (`README.md` top)

```yaml
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
  - Tongyi-MAI/Z-Image transformer/diffusion_pytorch_model.safetensors,text_encoder/*.safetensors,vae/diffusion_pytorch_model.safetensors,tokenizer/*
  - Tongyi-MAI/Z-Image-Turbo transformer/diffusion_pytorch_model.safetensors
  - PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1 Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors
  - xinntao/Real-ESRGAN RealESRGAN_x4plus.pth
---
```

Total preload ≈ 30 – 35 GB. Both transformers share the same text encoder + VAE + tokenizer (downloaded once via the Z-Image entry). Well under the 150 GB ephemeral storage cap and well under the 10-entry preload list cap.

### 6.2 Runtime cache mirror

`app.py:_bootstrap()` runs once at module import on HF Spaces:

1. Walk `~/.cache/huggingface/hub` (preload tree, owned by the build user, read-only at runtime).
2. Build a parallel writable tree at `~/hf-cache-rw/`:
   - `blobs/<sha>` files → hardlinked (zero copy, shared inode).
   - `snapshots/<commit>/...` symlinks → preserved as-is.
   - `refs/<branch>` → byte-copied (HF lib overwrites these on etag check).
   - All dirs → `mkdir` (we own them).
3. Set `HF_HOME=~/hf-cache-rw` and `HF_HUB_CACHE=~/hf-cache-rw/hub`.

After mirror: preloaded reads are instant cache hits, and lazy downloads (LoRAs uploaded by users at runtime) write to dirs the runtime user owns. Same mechanism as LTX (`_mirror_preload_hf_cache` in their `app.py`). Falls back to symlink if `os.link()` returns EXDEV.

### 6.3 Pipeline construction at boot

```python
pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=_auto_device(),  # "cuda" | "mps" | "cpu"
    model_configs=[
        # Both transformers — model_pool keeps them on CPU and swaps on demand
        ModelConfig(model_id="Tongyi-MAI/Z-Image",       origin_file_pattern="transformer/*.safetensors", **vram_cfg),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/*.safetensors", **vram_cfg),
        # Shared text encoder + VAE
        ModelConfig(model_id="Tongyi-MAI/Z-Image", origin_file_pattern="text_encoder/*.safetensors", **vram_cfg),
        ModelConfig(model_id="Tongyi-MAI/Z-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_cfg),
        # ControlNet — eager preload at boot to avoid first-ControlNet-call wait.
        # If startup RAM becomes tight on Spaces, move this to a lazy-load on first ControlNet request.
        ModelConfig(model_id="PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
                    origin_file_pattern="Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors", **vram_cfg),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image", origin_file_pattern="tokenizer/"),
    vram_limit=_vram_limit_for_device(),
)
```

Model selection (Base ↔ Turbo) inside the request handler just calls `pipe.dit = pipe.model_pool.fetch_model("z_image_dit", variant="base"|"turbo")` before `pipe(...)`. Per-call overhead: negligible — both are already on CPU.

---

## 7. ZeroGPU integration

### 7.1 Module-level decoration

```python
def _identity(fn): return fn

try:
    import spaces  # type: ignore
except ImportError:
    spaces = None

_ON_SPACES = bool(os.environ.get("SPACES_ZERO_GPU"))
_GPU = spaces.GPU(duration=_duration_for) if (spaces and _ON_SPACES) else _identity

@_GPU
def _generate(pipeline, mode, params, progress=None) -> tuple[Image, dict]:
    ...
```

Module-level so ZeroGPU's startup analyzer detects it. Identity off-Spaces.

### 7.2 Duration estimator

```python
_BASE_DURATION_S = {"t2i_base": 30, "t2i_turbo": 12, "controlnet": 35, "upscale": 45}
_PER_STEP_S = {"t2i_base": 2.0, "t2i_turbo": 1.5, "controlnet": 2.0, "upscale": 1.5}

def _duration_for(pipeline, mode, params, progress=None, multiplier=1.0) -> int:
    base = _BASE_DURATION_S[mode]
    per_step = _PER_STEP_S[mode] * params.get("steps", 8)
    image_size_factor = (params.get("width", 1024) * params.get("height", 1024)) / (1024 * 1024)
    cold_buffer = 20  # CPU→GPU copy on first call
    est = int((base + per_step + cold_buffer) * image_size_factor * multiplier)
    return max(60, min(est, 180))
```

Clamped to `[60, 180]` — well within ZeroGPU per-call caps. Light Turbo T2I gets queue priority; heavy upscale reserves real headroom.

### 7.3 Auto-retry on timeout

If the first attempt raises `gradio.exceptions.Error('GPU task aborted')`, classified as `category='gpu_timeout'`, the handler re-submits once with `multiplier=2.0`. UI shows a banner. One retry only.

### 7.4 Context propagation

`contextvars.copy_context()` is passed into any worker thread that calls into the GPU path — same fix as LTX. Without it, ZeroGPU's request-identity lookup fails and the call hits the unlogged-user path.

---

## 8. Errors & edge cases

| Category | Detection | Recovery |
| --- | --- | --- |
| OOM (CUDA / MPS) | Exception class `OutOfMemoryError` / message match | Re-call with reduced `vram_limit` (90% of free memory). If still OOM, surface "Try smaller size / use Turbo" error in UI. |
| ZeroGPU timeout | `'gpu task aborted'` in error message | Auto-retry once with `multiplier=2.0`. Banner shown. |
| Bad LoRA file | Safetensors header sniff fails or unexpected key prefix | Reject before `@spaces.GPU` fires. Error message names the unexpected key. |
| Preprocessor failure | controlnet_aux raises (e.g. canny on solid-color image) | Warn; fall back to raw input image. |
| Missing input on ControlNet/Upscale | gr.Image.value is None | Generate button is `interactive=False` until image uploaded. Caught also server-side. |
| Quota exceeded | `'exceeded your gpu'` in error | Surface ZeroGPU quota message + link to user's profile. |
| Expired token | `'expired zerogpu proxy token'` | Tell user to refresh the page. |

---

## 9. Testing strategy

Four tiers, mirroring LTX:

- **L1 — no GPU, CI** (`pytest`): ruff format/check, mode parameter shaping (`modes.py`), LoRA safetensors header validation, preprocessor type contracts, model config registry.
- **L2 — no GPU, CI** (`pytest`): monkeypatch `ZImagePipeline.__call__` to capture args; assert each mode handler builds the right call. Mock `controlnet_aux` to capture preprocessor invocations.
- **L3 — GPU smoke, manual** (`pytest --gpu`): one image per mode at 384 × 384 (small to keep duration short). Verifies real pipeline produces non-blank output.
- **L4 — HF Space smoke, manual**: push to a private staging Space, run all 3 modes, verify model selector hot-swap doesn't OOM and LoRA upload + revert keeps cached model clean.

No mocks for `ZImagePipeline` internals — only for its `__call__` boundary. Tests use fixed-content `assets/seed_inputs/` images.

---

## 10. Repo, license, naming, conventions

| Item                | Value                                                  |
| ------------------- | ------------------------------------------------------ |
| Local path          | `/Users/techfreakworm/Projects/llm/z-image-studio`      |
| GitHub repo (default) | `techfreakworm/z-image-studio` (confirm during plan)   |
| HF Space (default)  | `techfreakworm/z-image-studio` (confirm during plan)   |
| License             | **MIT** — DiffSynth (Apache-2.0) listed as dependency, not redistributed |
| Python              | 3.11 (matches LTX, HF base image)                      |
| venv                | `python3.11 -m venv .venv` — no conda                  |
| Model fetches       | `hf` CLI (not `huggingface-cli`), HF cache             |
| Linter              | `ruff format` + `ruff check`                           |
| Commits             | Conventional Commits style (`feat(ui): ...`, `fix(backend): ...`, etc.) — sole author = Mayank Gupta — no Claude co-author trailer, no "Generated with…" footer |

---

## 11. Decisions implicit in this spec (confirm in review)

1. **No t2v-style workflow JSON files** — DiffSynth's Python call surface IS the API, so we hand-code the four call shapes in `modes.py` (vs LTX which parameterizes JSONs).
2. **One pipeline instance** shared across modes. Transformer swap = the only model-pool change.
3. **No persistent storage add-on** needed — Pro Space + ephemeral storage is enough.
4. **License MIT** — say otherwise during review if you'd prefer Apache-2.0.
5. **ControlNet preload is in the YAML** — accept the larger startup at the gain of zero first-ControlNet-call wait. If RAM is tight at boot we'll move it to lazy.

---

## 12. Open questions (none blocking implementation)

- Final GitHub + HF namespace strings (defaults above are placeholders).

Everything else is settled. Ready to hand off to `writing-plans` for the implementation plan.
