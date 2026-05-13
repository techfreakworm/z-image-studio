# AGENTS.md

Tool-agnostic agent guidance for the Z-Image Studio repo. If you're driving Claude Code, Cursor, Aider, Codex, or anything else with file-edit + shell access, **start here**.

This file is the authoritative project rulebook. `CLAUDE.md` is Claude-specific extensions; `SKILLS.md` is workflow rules. README.md is the public-facing project intro — different audience.

---

## TL;DR — the five rules

1. **Mayank Gupta is sole author on every commit.** No agent co-author trailers. No "generated with…" footers. No `--author=` flag. Strip any tool-suggested attribution.
2. **Backend = DiffSynth-Studio, not ComfyUI.** Don't add a ComfyUI dependency under any guise.
3. **Both transformers live in one pool.** `pipe._zis_pool` is our handle; `pipe.dit` swaps by index. Don't refactor to one-pipeline-per-model — it doubles memory and breaks LoRA-revert.
4. **Don't pin `spaces` in `requirements.txt`.** HF Spaces' ZeroGPU build injects its own version. A pin causes pip-resolve failure.
5. **Locally is the source of truth.** All changes restart `python app.py` and verify on http://127.0.0.1:7860 BEFORE pushing to HF. The Space rebuild is ~5–10 min; iterate locally.

If you can't satisfy these without changing architectural shape, **ask the user before proceeding**.

---

## Project shape

Single-process Gradio 5.50 app, flat top-level Python layout, ~2.7k LOC including tests.

```
app.py            Gradio Blocks entry + bootstrap + event handlers + CTA banner
backend.py        ZImageStudioBackend; @spaces.GPU; duration_for; generate_with_retry
modes.py          call_t2i / call_controlnet / call_upscale (pure handlers)
models.py         auto_device, MODEL_CONFIGS, vram_limit_for, HF→DiffSynth symlink helper
lora.py           safetensors header sniff + applied_lora context manager
preprocessors.py  Canny (cv2), Depth (controlnet_aux "depth_midas"), Pose ("openpose")
upscale.py        RealESRGAN x4 + 0.5 resize bridge (with basicsr→torchvision shim)
ui.py             Three per-tab builders, gr.Radio model selector, soon-row links
theme.py          Soft Dark Restraint palette + minimal CSS (~175 lines)
tooltips.py       Centralised `info=` strings — single source of truth
tests/            70 passing tests + 4 GPU-deselected smoke
docs/superpowers/ spec + plan + brainstorm artifacts
```

Same code path locally (MPS / CUDA) and on HF Spaces. The only branching is whether `_bootstrap()` does the cache-mirror dance (Spaces) or just the symlink step (local).

---

## Locked architecture decisions

These came out of brainstorming + 40+ commits of iteration. Do not relitigate.

| Decision | Why | Code reference |
|---|---|---|
| One `ZImagePipeline` instance, both transformers preloaded | Avoids ~30 s pipeline rebuild per model swap; LoRA revert is cleaner | `backend._build_pipeline` |
| Transformer swap = `pipe.dit = pool.model[idx]` | DiffSynth's `fetch_model("z_image_dit")` returns the first match; both base + turbo register under the same name. Index by load order. | `modes._swap_transformer` |
| MPS `vram_limit = None` | `torch.mps` has no `mem_get_info`; DiffSynth's `check_free_vram` raises AttributeError otherwise | `models.vram_limit_for` |
| `PYTORCH_ENABLE_MPS_FALLBACK=1` set at app import | A few MPS-unsupported ops crash mid-pipeline without it | `app.py` top-of-file |
| HF cache → `./models/<repo>/` symlink at boot | DiffSynth's `ModelConfig.download` looks at `local_model_path/<model_id>/`, NOT in the HF cache `models--<org>--<repo>/snapshots/<sha>/` layout | `app._bootstrap` + `models.symlink_hf_cache_to_diffsynth_layout` |
| Native `gr.Radio` for model selector (not a custom HTML card grid) | Gradio reactivity + accessibility free; nothing to debug | `ui.build_t2i_tab` |
| Native `gr.Progress(track_tqdm=True)` for progress bar | DiffSynth + RealESRGAN both use `tqdm`; one parameter auto-captures both | `app.on_*_generate` signatures |
| Soft Dark Restraint theme | Locked from brainstorming round 2 (round 1 was over-designed) | `theme.py` |
| Single output meta block under the image | The first redesign duplicated meta in Advanced; users flagged it | `ui.build_*_tab` |

---

## Commit rules

- **Conventional Commits:** `<type>(<scope>): <subject>`
  - types: `feat`, `fix`, `chore`, `docs`, `test`, `refactor`, `ci`, `perf`
- Subject is imperative, lowercase, **no trailing period**.
- Body explains **why** when not obvious. Reference plan task IDs (Task 7, Task A, etc.) when the change implements a specific plan step.
- Frequent small commits; one logical change per commit.
- **No agent attribution** in commit message or body. See rule 1.
- Don't `git push --force` to `main` unless the user explicitly says so. Force-push to a feature branch is fine; the seed commits + spec doc are on `main` and protected by convention only.

---

## Verification rules

- **Tests must pass before committing.** `python -m pytest tests/ -q` from the project root. Target: 70/70 + 4 deselected.
- **Ruff must be clean.** `ruff check . && ruff format --check .`
- **The local app must boot.** `python app.py` → http://127.0.0.1:7860 reachable, no import error in `/tmp/zimage-studio.log`.
- **For UI changes:** open the URL in a browser (or Playwright eval) and verify the change is rendered. Don't trust a clean test run + clean ruff as proof that the UI works.
- **For deployment changes:** push to HF Space, watch the build, verify the runtime stage transitions to `RUNNING` before claiming success.

If a change requires breaking these rules, write the reason in the commit body.

---

## Testing conventions

- **TDD per the plan.** Failing test first, then implementation.
- **L1 + L2 in CI** (no GPU). The mode handlers are tested with a mocked pipeline — patches on `preprocessors.run`, `upscale.realesrgan_2x`, and direct injection of fake dits into `pipe._zis_pool.model`. We do NOT mock DiffSynth internals.
- **L3 GPU smoke** is opt-in (`pytest -m gpu`). Lives in `tests/test_smoke_gpu.py`. Loads the real pipeline (~30 GB cache hit on a warm machine).
- **L4 HF Space smoke** is manual. Push, wait, click each tab, verify the image renders.

`pyproject.toml` has `addopts = -m 'not gpu'` so the default `pytest` invocation skips GPU. Add the marker before any test that touches DiffSynth weights.

---

## Out of scope (v1 cap — don't add without asking)

The spec lists these as deferred. If you find yourself "while I'm here"-ing into one of them, stop.

- Multi-prompt queueing
- Output history across sessions
- Telemetry-driven duration estimator
- Persistent storage add-on
- Custom LoRA add/remove rows (single LoRA per tab is the cap)
- LoRA on the Upscale refinement pass
- ControlNet on Z-Image base
- Z-Image-Edit and Z-Image-Omni-Base (intentionally placeholders linking to GitHub Model Zoo)
- Display-font customization beyond Inter
- Visual regression tests
- Property-based / fuzz testing of generation params

If a feature you're adding requires one of these as a sub-step, **ask the user** before proceeding.

---

## When you're not sure

1. Read `docs/superpowers/specs/2026-05-13-z-image-studio-design.md` — that's the architectural source of truth.
2. Read `docs/superpowers/plans/2026-05-13-z-image-studio.md` — the task-by-task breakdown.
3. Read `SKILLS.md` — process rules, debugging patterns, deployment workflow.
4. `git log --oneline` — every non-obvious decision has a fix-commit explaining the reasoning.
5. **Ask the user.** A clarifying question costs the user ten seconds. A wrong implementation costs everyone an hour.
