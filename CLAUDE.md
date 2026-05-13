# Project Guidelines — Z-Image Studio

Working notes for AI assistants editing this repo. This file is the *what & why* — the locked architecture, the gotchas, the sole-author rule. Companion to `SKILLS.md` (the *how* — process, debugging, deployment workflow) and `AGENTS.md` (tool-agnostic version of this file).

---

## ⚠ Sole-author rule (non-negotiable)

**Mayank Gupta is the sole author on every commit in this repo.** No exceptions.

When committing:

- **NO** `Co-Authored-By: Claude…` (or any agent name) trailer.
- **NO** "Generated with Claude Code" / "🤖 Generated with…" footers.
- **NO** `--author=…` flag — let git use the user's configured identity.
- **NO** attribution in PR descriptions.

If asked to amend, re-commit, or rebase, strip any prior agent attribution from the commit message. Treat any tooling that suggests adding a Claude trailer as a bug to ignore.

---

## Architecture facts (locked — do not relitigate)

Spec: `docs/superpowers/specs/2026-05-13-z-image-studio-design.md`
Plan: `docs/superpowers/plans/2026-05-13-z-image-studio.md`

1. **Backend is DiffSynth-Studio's `ZImagePipeline`** — not ComfyUI. Installed from git (the package isn't on PyPI). The repo lives at `/Users/techfreakworm/Projects/llm/lora-training-zimage-base/DiffSynth-Studio/` for local development and is `git+https://github.com/modelscope/DiffSynth-Studio.git` in `requirements.txt`.
2. **Three tabs.** T2I has the Base/Turbo radio; ControlNet and Upscale are hard-locked to Turbo.
3. **One pipeline instance, two transformers in the pool.** `backend._build_pipeline` does NOT call `ZImagePipeline.from_pretrained` (which discards its `ModelPool` locally). Instead it instantiates the pipeline manually, runs `download_and_load_models`, attaches the pool to `pipe._zis_pool`, and indexes the two `z_image_dit` entries by load order (Base = `pool.model[0]`, Turbo = `pool.model[1]`). Swap is `pipe.dit = dits[idx]` in `modes._swap_transformer`.
4. **`@spaces.GPU` is applied at module load time.** Identity decorator off Spaces. The decorator's `duration=` parameter takes a callable that estimates per-call timeout from `(mode, params, multiplier)`. Estimator clamps at `[60, 180] s`.
5. **DiffSynth handles VRAM management.** Do **not** sprinkle `empty_cache()` calls. The only place we touch this is `models.vram_limit_for()` which returns `None` for MPS (CUDA-only `mem_get_info` API would crash otherwise) and a numeric cap for CUDA.
6. **HF cache → DiffSynth `./models/<repo>/` symlink.** DiffSynth's `ModelConfig.download()` looks for files at `local_model_path/<model_id>/...`, NOT in `~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<sha>/`. `app._bootstrap()` symlinks every cached snapshot into `./models/<org>/<repo>/` so the preload weights are findable. On Spaces, the build-user-owned `~/.cache/huggingface/hub` is mirrored to runtime-writable `~/hf-cache-rw/` first, then symlinked.
7. **One Gradio process. Lazy backend singleton.** `get_backend()` constructs the pipeline on the first request (~30 – 60 s warm-up). Module import is fast.

---

## Gotchas we already paid for (don't re-discover)

Each of these cost a debug cycle. Read once.

### Model selector swap

- `pipe.model_pool` does NOT exist after `ZImagePipeline.from_pretrained` — DiffSynth builds the pool locally and discards it. **Fix:** we keep our own reference on `pipe._zis_pool`. See architecture fact #3.
- A hidden `gr.Textbox(visible=False)` is removed from the DOM entirely in Gradio 5, so a JS shim can't write to it. We use `elem_classes=["zis-hidden"]` + CSS `display:none` when we need an off-screen value carrier. As of the v2 redesign we use `gr.Radio` directly and don't need a carrier textbox.

### MPS / Apple Silicon

- `torch.mps` has no `mem_get_info`. DiffSynth's `AutoWrappedModule.check_free_vram` calls that method and raises AttributeError when `vram_limit` is set. **Fix:** `vram_limit_for("mps")` returns `None` so the gate short-circuits.
- Several DiffSynth ops aren't implemented on the MPS backend (SDPA variants, some index ops). `app.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so they degrade to CPU instead of crashing.

### Dependency footguns

- `diffsynth-studio` (kebab) is NOT a PyPI package. The pip-installable name is `diffsynth` and only via `git+https://github.com/modelscope/DiffSynth-Studio.git`.
- `transformers >= 5` removes `SiglipVisionTransformer` from `transformers.models.siglip.modeling_siglip`. DiffSynth 2.0.7 imports it. **Pin:** `transformers>=4.45,<5.0`.
- DiffSynth blanket-imports `torchaudio` in `diffsynth.core.data.operators`. Add `torchaudio>=2.4` to requirements even though we don't use audio.
- `basicsr` (a `realesrgan` dep) imports `torchvision.transforms.functional_tensor`, removed in `torchvision >= 0.17`. **Fix:** `upscale.py` aliases `torchvision.transforms.functional` into `sys.modules["torchvision.transforms.functional_tensor"]` BEFORE the basicsr import.

### Model name slugs

- `PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1` is the **ModelScope** slug. On HuggingFace the same model is at `alibaba-pai/...`. We use the HF slug + `DIFFSYNTH_DOWNLOAD_SOURCE=huggingface` env var.
- `xinntao/Real-ESRGAN` doesn't exist on HF (returns 401). We use `lllyasviel/Annotators` which mirrors `RealESRGAN_x4plus.pth`.
- `controlnet_aux.Processor` registers depth as `depth_midas`, **not** `midas`. The plain name raises KeyError.

### Gradio 5 quirks

- Don't put `<script>` tags inside `gr.HTML` blocks — they get stripped. JS goes in `gr.Blocks(head=…)`.
- `gr.File`'s default drop zone is ~400 px tall. CSS in `theme.py` (`.zis-lora-file .upload-container`) tightens it to 56 px.
- The Gradio 6.0 deprecation warnings about `theme=` / `css=` / `head=` on `Blocks` are benign on 5.50. Ignore until upgrade.

### HF Spaces deployment

- `preload_from_hub` is build-time only. Runtime falls back to network if any required file isn't preloaded. Use broad globs (`transformer/*` not `transformer/*.safetensors`) so configs + index.json files come along. Our current preload totals ~47 GB (cap is 150 GB).
- ZeroGPU build injects `spaces==0.50.0`. If `requirements.txt` pins `spaces==0.30.0`, pip resolution fails. **Don't pin `spaces` at all** — let HF provide it.
- The `@spaces.GPU` decorator must be applied at module load. Runtime decoration isn't detected by ZeroGPU's startup analyzer.
- Per-call `duration=` is a queue-priority signal AND a hard cap. Auto-retry once at 2× on `"GPU task aborted"`.

### Brand vs filename casing

- Repo / directory / Python package: `z-image-studio` (kebab-case).
- User-visible brand: `Z-Image Studio` (title-case) — header, browser tab, README title. Do not propagate the kebab into UI strings.

---

## Coding conventions

- **Python 3.11.** HF Spaces base image is 3.11; older syntax (like no `match`) is fine.
- **Flat top-level layout.** No `src/`, no nested packages. One `.py` per responsibility.
- **No conda.** `python3.11 -m venv .venv`; `brew` for system binaries.
- **No emojis** in code or commits unless explicitly requested. UI strings (CTA banner, button labels) are OK because they're user-facing copy, not code.
- **Type hints on public functions.** Internal helpers can skip them when obvious.
- **Imports at the top of the file.** Inline imports only to break circular deps OR to defer heavy modules (DiffSynth, torch, basicsr) for fast CI startup.
- **`ruff format` + `ruff check`** both pass in CI. No exceptions.

---

## Commits

- **Conventional Commits:** `<type>(<scope>): <subject>` — types: `feat`, `fix`, `chore`, `docs`, `test`, `refactor`, `ci`, `perf`.
- Subject is **imperative**, lowercase, no trailing period.
- Body explains **why** when not obvious. Reference the spec / plan section if relevant.
- Frequent small commits — one logical change per commit.
- **NO Claude trailer.** See top of file.

---

## Testing

- **TDD per the plan.** Each implementation task has the failing test first.
- **L1 + L2 in CI** (no GPU): module structure, mocked pipeline call boundaries, ruff. `tests/test_smoke_gpu.py` is the GPU smoke; it's marked with `@pytest.mark.gpu` and skipped by default (pyproject `addopts = -m 'not gpu'`).
- **No mocks for DiffSynth internals.** Mock only the `pipe(...)` call boundary so the mode-handler logic is verified at the boundary.
- **Use `pytest -m gpu`** to opt into the GPU smoke (~30 GB download on a cold cache; runs full t2i base/turbo + controlnet + upscale at 384²).

---

## Out of scope for v1 (don't add without asking)

- Multi-prompt queueing
- Output history persistence across sessions
- Telemetry / duration estimator that learns from logs
- Persistent storage add-on integration
- Custom LoRA add/remove rows (single LoRA per tab is the v1 cap)
- LoRA on the Upscale refinement pass (locked to vanilla Turbo refinement)
- ControlNet on Z-Image base (no released ControlNet weights for base)
- Z-Image-Edit and Z-Image-Omni-Base (placeholders link to GitHub Model Zoo)
- Display-font customization beyond Inter (locked by Soft Dark Restraint)
- Visual regression tests for the Gradio UI

If a task feels like it needs one of these, stop and ask the user.

---

## When in doubt

1. Read the spec + plan. Fifteen minutes of reading vs a day of wrong implementation.
2. Read `SKILLS.md` for the process side — debugging, deployment, when to commit, when to verify.
3. `git log --oneline` — most non-obvious decisions have a fix-commit explaining the reasoning.
4. **Ask the user** before changing architectural shape or adding scope outside the v1 list.
