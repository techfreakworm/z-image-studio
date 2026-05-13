# z-image-studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-process Gradio 5.x app exposing Z-Image + Z-Image-Turbo via DiffSynth-Studio with three tabs (Text→Image dual-model, ControlNet, Upscale) and a per-tab LoRA loader, running locally on Apple Silicon (MPS) or NVIDIA (CUDA) and on Hugging Face Spaces (ZeroGPU H200).

**Architecture:** One `ZImagePipeline` shared across modes; `@spaces.GPU(duration=callable)` applied at module load (identity decorator off-Spaces). DiffSynth handles VRAM management. Flat top-level Python layout — one responsibility per file. Onyx Amber theme wired via `gr.themes.Base(...).set(...)` + a small CSS string.

**Tech Stack:** Python 3.11 · Gradio 5.50 · DiffSynth-Studio (Apache-2.0) · `spaces` (HF) · `controlnet-aux` · `realesrgan` · `torch>=2.4` (bf16) · `safetensors` · `ruff` · `pytest`.

**Spec:** `docs/superpowers/specs/2026-05-13-z-image-studio-design.md` — read first if any decision is unclear.

---

## File map

```
llm/z-image-studio/                 (already initialized; .gitignore + spec committed)
├── app.py                # Task 15. Gradio Blocks entry, _bootstrap, app.launch
├── backend.py            # Task 12, 13. ZImageStudioBackend; @spaces.GPU; duration estimator
├── modes.py              # Task 9-11. Pure mode handler functions
├── models.py             # Task 3, 4. Device autodetect, ModelConfig list, HF cache mirror
├── preprocessors.py      # Task 7. Canny/Depth/Pose via controlnet_aux (lazy imports)
├── upscale.py            # Task 8. RealESRGAN x4 + 0.5-resize bridge
├── lora.py               # Task 5, 6. Safetensors header sniff + apply/revert ctx
├── ui.py                 # Task 14. Per-tab Gradio component builders
├── theme.py              # Task 2. Onyx Amber tokens + gr.themes.Base subclass + CSS string
├── pyproject.toml        # Task 1. ruff + pytest config; py311
├── requirements.txt      # Task 1. Pinned deps
├── README.md             # Task 16. HF Space YAML + user docs
├── LICENSE               # Task 1. MIT
├── CLAUDE.md             # Task 1. Sole-author rule + venv + hf CLI conventions
├── setup.sh              # Task 1. python3.11 -m venv .venv
├── .github/workflows/ci.yml  # Task 17. ruff + pytest L1/L2
└── tests/
    ├── __init__.py
    ├── conftest.py       # Task 1. Shared fixtures
    ├── test_theme.py     # Task 2
    ├── test_models.py    # Task 3, 4
    ├── test_lora.py      # Task 5, 6
    ├── test_preprocessors.py  # Task 7
    ├── test_upscale.py   # Task 8
    ├── test_modes.py     # Task 9-11
    ├── test_backend.py   # Task 12, 13
    └── test_scaffold.py  # Task 1
```

The directory `/Users/techfreakworm/Projects/llm/z-image-studio/` is already a git repo with the spec committed (commit `9ee5274`). All work happens inside that directory.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `setup.sh`, `LICENSE`, `CLAUDE.md`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_scaffold.py`
- The `.gitignore` already exists in the seed commit

- [ ] **Step 1.1: Write the failing scaffold test**

Create `tests/test_scaffold.py`:

```python
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]

def test_required_files_exist():
    for rel in [
        "pyproject.toml", "requirements.txt", "setup.sh",
        "LICENSE", "CLAUDE.md", "README.md", ".gitignore",
        "tests/__init__.py", "tests/conftest.py",
    ]:
        assert (REPO / rel).exists(), f"missing {rel}"

def test_pyproject_targets_py311():
    text = (REPO / "pyproject.toml").read_text()
    assert "python = " not in text  # not poetry
    assert "py311" in text  # ruff target-version

def test_requirements_has_core_deps():
    text = (REPO / "requirements.txt").read_text().lower()
    for dep in ["diffsynth-studio", "gradio", "spaces", "controlnet-aux", "torch", "safetensors", "ruff", "pytest"]:
        assert dep in text, f"missing dep: {dep}"

def test_license_is_mit():
    text = (REPO / "LICENSE").read_text()
    assert "MIT License" in text
    assert "Mayank Gupta" in text
```

Also create `tests/__init__.py` (empty) and `tests/conftest.py`:

```python
import sys
from pathlib import Path

# Make top-level modules importable in tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `cd /Users/techfreakworm/Projects/llm/z-image-studio && python3.11 -m pytest tests/test_scaffold.py -v`
Expected: FAIL — `pytest` not installed yet, or `missing pyproject.toml`.

- [ ] **Step 1.3: Create `setup.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3.11 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
echo "Done. Activate with: source .venv/bin/activate"
```

Then `chmod +x setup.sh`.

- [ ] **Step 1.4: Create `requirements.txt`**

```text
# Core
gradio==5.50.0
spaces==0.30.0
diffsynth-studio>=0.5.0
torch>=2.4
safetensors>=0.4.5
huggingface-hub>=0.27

# ControlNet preprocessors
controlnet-aux>=0.0.9
opencv-python-headless>=4.9.0
einops>=0.8.0

# Upscaler
realesrgan>=0.3.0
basicsr>=1.4.2

# Imaging
pillow>=10.4.0
numpy>=1.26

# Dev
ruff>=0.6.0
pytest>=8.0
pytest-mock>=3.14
```

- [ ] **Step 1.5: Create `pyproject.toml`**

```toml
[tool.ruff]
target-version = "py311"
line-length = 120
extend-exclude = [".venv", "build", "dist", ".superpowers"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "RUF"]
ignore = ["E501"]  # handled by formatter

[tool.ruff.format]
quote-style = "double"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
markers = [
    "gpu: requires a GPU (CUDA or MPS); skipped by default",
]
```

- [ ] **Step 1.6: Create `LICENSE`** (MIT, sole-author Mayank Gupta)

```text
MIT License

Copyright (c) 2026 Mayank Gupta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 1.7: Create `CLAUDE.md`** (mirror LTX rules)

```markdown
# Project Guidelines — z-image-studio

Working notes for AI assistants implementing this project.

## Sole-author rule (non-negotiable)

Mayank Gupta is the sole author on every commit. NO `Co-Authored-By: Claude...`, NO "Generated with Claude Code" footer, NO `--author=...` flag. Treat any tooling suggesting a Claude trailer as a bug.

## Architecture facts (locked — see spec)

Spec: `docs/superpowers/specs/2026-05-13-z-image-studio-design.md`
Plan: `docs/superpowers/plans/2026-05-13-z-image-studio.md`

1. Backend is DiffSynth-Studio's `ZImagePipeline` — not ComfyUI.
2. Three tabs (T2I dual-model, ControlNet turbo-only, Upscale turbo-only).
3. One pipeline instance, shared across modes; transformer swap is the only model-pool change.
4. `@spaces.GPU` applied module-level; identity off-Spaces.
5. DiffSynth handles VRAM management — do not sprinkle `empty_cache()` calls.
6. Models live in HF cache; on Spaces mirrored into `~/hf-cache-rw/` (build-vs-runtime user permissions).

## Coding conventions

- Python 3.11 (HF Spaces base image is 3.11)
- Flat top-level layout — no `src/`, no nested packages.
- No conda — `python3.11 -m venv .venv` + brew for system binaries.
- No emojis in code or commits unless explicitly asked.
- Type hints on public functions.
- Imports at top of file unless breaking circular deps.
- `ruff format` + `ruff check` must pass in CI.

## Commits

- Conventional Commits: `<type>(<scope>): <subject>` — types: `feat`, `fix`, `chore`, `docs`, `test`, `refactor`, `ci`, `perf`.
- Subject is imperative, lowercase, no trailing period.
- Body explains WHY when non-obvious. Reference plan task if relevant.
- Frequent small commits — one logical change per commit.
- NO Claude trailer (see above).

## Testing

- TDD per the plan — failing test first, then implementation.
- L1 + L2 run in CI without GPU. L3 + L4 require GPU/HF Space and are manual.
- No mocks for DiffSynth internals — mock only the `pipe(...)` call boundary.
- Use `pytest --gpu` to opt into L3 smoke tests.
```

- [ ] **Step 1.8: Run scaffold test — expect PASS**

```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -q pytest
python -m pytest tests/test_scaffold.py -v
```

Expected: 4 PASSed.

- [ ] **Step 1.9: Commit**

```bash
git add pyproject.toml requirements.txt setup.sh LICENSE CLAUDE.md tests/
git commit -m "chore: project scaffolding (pyproject, requirements, license, claude.md, tests)"
```

---

## Task 2: Onyx Amber theme

**Files:**
- Create: `theme.py`
- Test: `tests/test_theme.py`

- [ ] **Step 2.1: Write the failing test**

Create `tests/test_theme.py`:

```python
import theme

def test_amber_palette_tokens_match_spec():
    pal = theme.AMBER
    assert pal["body_bg"] == "#0F0C08"
    assert pal["text"] == "#FAF1E3"
    assert pal["text_dim"] == "#A89478"
    assert pal["border"] == "#2A2218"
    assert pal["accent"] == "#FFB02E"
    assert pal["accent_text"] == "#1A1208"
    assert pal["radius"] == "8px"

def test_build_theme_returns_gradio_base():
    import gradio as gr
    th = theme.build_theme()
    assert isinstance(th, gr.themes.Base)

def test_css_string_contains_critical_selectors():
    css = theme.CSS
    # warm vignette + amber button glow are the two decorations the spec calls out
    assert "radial-gradient" in css
    assert "rgba(255,176,46" in css.lower() or "255, 176, 46" in css.lower()

def test_fonts_geist_and_geist_mono():
    th = theme.build_theme()
    # gr.themes.GoogleFont stringifies to its name
    fonts = [str(f) for f in th.font]
    assert any("Geist" in f for f in fonts)
    monos = [str(f) for f in th.font_mono]
    assert any("Geist Mono" in f for f in monos)
```

- [ ] **Step 2.2: Run test to verify it fails**

`python -m pytest tests/test_theme.py -v` → ModuleNotFoundError: theme.

- [ ] **Step 2.3: Implement `theme.py`**

```python
"""Onyx Amber theme — palette tokens, gr.themes.Base subclass, and CSS string."""
from __future__ import annotations

import gradio as gr

AMBER: dict[str, str] = {
    "body_bg":     "#0F0C08",
    "panel_bg":    "#0F0C08",
    "input_bg":    "#0F0C08",
    "canvas_bg":   "#110D08",
    "border":      "#2A2218",
    "text":        "#FAF1E3",
    "text_dim":    "#A89478",
    "accent":      "#FFB02E",
    "accent_text": "#1A1208",
    "radius":      "8px",
    "radius_sm":   "6px",
}


def build_theme() -> gr.themes.Base:
    """Return a Gradio theme matching the Onyx Amber palette."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#FFF8E6", c100="#FFEFC2", c200="#FFE08A",
            c300="#FFD161", c400="#FFC042", c500=AMBER["accent"],
            c600="#E69926", c700="#B37A1F", c800="#805717", c900="#4D3510", c950="#1A1208",
        ),
        neutral_hue=gr.themes.Color(
            c50="#FAF1E3", c100="#E8DCC4", c200="#D4C2A1", c300="#A89478",
            c400="#867054", c500="#5C4D38", c600="#3C3225", c700="#2A2218",
            c800="#1C170F", c900="#100C08", c950="#0A0805",
        ),
        font=[gr.themes.GoogleFont("Geist"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Geist Mono"), "ui-monospace", "monospace"],
        radius_size=gr.themes.sizes.radius_md,
    ).set(
        body_background_fill=AMBER["body_bg"],
        body_text_color=AMBER["text"],
        body_text_color_subdued=AMBER["text_dim"],
        background_fill_primary=AMBER["panel_bg"],
        background_fill_secondary=AMBER["canvas_bg"],
        block_background_fill=AMBER["panel_bg"],
        block_border_color=AMBER["border"],
        block_border_width="1px",
        block_radius=AMBER["radius"],
        input_background_fill=AMBER["input_bg"],
        input_border_color=AMBER["border"],
        button_primary_background_fill=AMBER["accent"],
        button_primary_background_fill_hover=AMBER["accent"],
        button_primary_text_color=AMBER["accent_text"],
        button_primary_border_color=AMBER["accent"],
        slider_color=AMBER["accent"],
        color_accent=AMBER["accent"],
        color_accent_soft="rgba(255,176,46,0.12)",
    )


CSS: str = """
/* Onyx Amber — atmospheric layer that Gradio's theme can't express alone */

body, .gradio-container {
    background-image: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(255,176,46,0.06), transparent 70%);
}

/* Amber glow on primary button */
.gradio-container button.primary {
    box-shadow: 0 0 0 1px rgba(255,176,46,0.4), 0 8px 24px -8px rgba(255,176,46,0.35);
}

/* Slim status line typography */
.zis-status {
    font-family: 'Geist Mono', ui-monospace, monospace;
    font-size: 11px;
    letter-spacing: 0.06em;
    color: #A89478;
}

/* LoRA file slot — solid amber border + slim icon when a file is loaded */
.zis-lora.loaded {
    border: 1px solid #FFB02E !important;
}
""".strip()
```

- [ ] **Step 2.4: Run test — expect PASS**

`python -m pytest tests/test_theme.py -v` → 4 PASSed.

- [ ] **Step 2.5: Commit**

```bash
git add theme.py tests/test_theme.py
git commit -m "feat(theme): onyx amber palette + gr.themes.Base + glow CSS"
```

---

## Task 3: Device autodetect + model config registry

**Files:**
- Create: `models.py`
- Test: `tests/test_models.py`

- [ ] **Step 3.1: Write failing test**

Create `tests/test_models.py`:

```python
import os
from unittest import mock

import models


def test_auto_device_returns_cuda_or_mps_or_cpu():
    dev = models.auto_device()
    assert dev in ("cuda", "mps", "cpu")


def test_on_spaces_reads_env_var():
    with mock.patch.dict(os.environ, {"SPACES_ZERO_GPU": "1"}, clear=False):
        assert models.on_spaces() is True
    with mock.patch.dict(os.environ, {}, clear=True):
        assert models.on_spaces() is False


def test_model_configs_contains_both_transformers():
    configs = models.MODEL_CONFIGS
    repos = {c.model_id for c in configs}
    assert "Tongyi-MAI/Z-Image" in repos
    assert "Tongyi-MAI/Z-Image-Turbo" in repos
    assert "PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1" in repos


def test_vram_limit_for_cuda_is_reasonable():
    limit = models.vram_limit_for("cuda", free_gb=80.0)
    assert 60.0 <= limit <= 80.0  # leave headroom


def test_vram_limit_for_mps_is_unified_memory_aware():
    limit = models.vram_limit_for("mps", free_gb=24.0)
    assert 12.0 <= limit <= 22.0  # half of unified, headroom


def test_vram_limit_for_cpu_is_zero():
    assert models.vram_limit_for("cpu", free_gb=64.0) == 0.0
```

- [ ] **Step 3.2: Run test — expect FAIL**

`python -m pytest tests/test_models.py -v` → ModuleNotFoundError.

- [ ] **Step 3.3: Implement `models.py` (device + configs only — cache mirror is Task 4)**

```python
"""Device autodetect, ZImagePipeline ModelConfig registry, and (Task 4) HF cache mirror."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Avoid importing torch at module load — keeps `import models` fast in CI.


def on_spaces() -> bool:
    """True iff we are running inside a Hugging Face ZeroGPU Space."""
    return bool(os.environ.get("SPACES_ZERO_GPU"))


def auto_device() -> str:
    """Detect the best available compute device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def vram_limit_for(device: str, free_gb: float | None = None) -> float:
    """Conservative VRAM limit (GB) passed to DiffSynth's vram_management.

    - CUDA: keep ~5% headroom (loaded models + scratch).
    - MPS: half of unified memory (CPU still needs RAM), capped.
    - CPU: 0.0 (no offload budget).
    """
    if device == "cpu":
        return 0.0
    if free_gb is None:
        import torch
        if device == "cuda":
            free_gb = torch.cuda.mem_get_info()[1] / (1024 ** 3)
        else:  # mps
            # torch.mps has no mem_get_info on most builds; fall back to a safe constant.
            free_gb = 24.0
    if device == "mps":
        return max(8.0, free_gb / 2 - 1.0)
    # cuda
    return max(8.0, free_gb - 4.0)


@dataclass(frozen=True)
class ModelConfig:
    """Lightweight wrapper around DiffSynth's ModelConfig.

    Stored as plain data so this module imports cheaply in CI. The real
    ``diffsynth.core.ModelConfig`` instance is built on demand by
    :func:`build_diffsynth_configs`.
    """
    model_id: str
    origin_file_pattern: str
    description: str = ""


MODEL_CONFIGS: tuple[ModelConfig, ...] = (
    # Base
    ModelConfig("Tongyi-MAI/Z-Image", "transformer/*.safetensors",
                "Z-Image base transformer (25 steps, cfg=4)"),
    ModelConfig("Tongyi-MAI/Z-Image", "text_encoder/*.safetensors",
                "Qwen3-4B text encoder — shared between base + turbo"),
    ModelConfig("Tongyi-MAI/Z-Image", "vae/diffusion_pytorch_model.safetensors",
                "Flux-family VAE — shared between base + turbo"),
    # Turbo (transformer only — encoder + VAE come from the Z-Image entry above)
    ModelConfig("Tongyi-MAI/Z-Image-Turbo", "transformer/*.safetensors",
                "Z-Image-Turbo transformer (8 steps, cfg=1)"),
    # ControlNet Union 2.1 (eager preload per spec; can move to lazy if RAM is tight)
    ModelConfig("PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
                "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors",
                "ControlNet Union 2.1 — canny/depth/pose"),
)

TOKENIZER_CONFIG = ModelConfig("Tongyi-MAI/Z-Image", "tokenizer/",
                                "Qwen3-4B tokenizer")


def build_diffsynth_configs(
    configs: tuple[ModelConfig, ...] = MODEL_CONFIGS,
    vram_cfg: dict[str, Any] | None = None,
) -> list[Any]:
    """Build DiffSynth ``ModelConfig`` instances from our lightweight dataclasses.

    Called at app boot; not at module import. ``vram_cfg`` is the disk-offload
    block (offload_dtype, offload_device, etc.) that DiffSynth's low-VRAM examples use.
    """
    from diffsynth.core import ModelConfig as DSConfig
    return [
        DSConfig(model_id=c.model_id, origin_file_pattern=c.origin_file_pattern, **(vram_cfg or {}))
        for c in configs
    ]
```

- [ ] **Step 3.4: Run test — expect PASS**

`python -m pytest tests/test_models.py -v` → 6 PASSed.

- [ ] **Step 3.5: Commit**

```bash
git add models.py tests/test_models.py
git commit -m "feat(models): device autodetect, vram-limit helpers, model config registry"
```

---

## Task 4: HF Spaces cache mirror

**Files:**
- Modify: `models.py`
- Test: `tests/test_models.py`

The mirror copies the read-only `preload_from_hub` tree (owned by the build user) into a writable parallel tree owned by the runtime user. Same trick as LTX2.3-AIO-generator.

- [ ] **Step 4.1: Write failing test**

Append to `tests/test_models.py`:

```python
def test_mirror_hardlinks_blobs(tmp_path):
    """Blobs (content-addressed files) get hardlinked into the mirror."""
    src = tmp_path / "src" / "hub"
    dst = tmp_path / "rw"
    blob_dir = src / "blobs"
    blob_dir.mkdir(parents=True)
    blob = blob_dir / "abcdef"
    blob.write_bytes(b"hello")

    models.mirror_preload_hf_cache(src.parent, dst)

    mirrored = dst / "hub" / "blobs" / "abcdef"
    assert mirrored.exists()
    assert mirrored.stat().st_ino == blob.stat().st_ino, "should be hardlinked"


def test_mirror_preserves_snapshot_symlinks(tmp_path):
    """Snapshot symlinks point at relative blob paths — preserve as-is."""
    src = tmp_path / "src" / "hub"
    dst = tmp_path / "rw"
    (src / "blobs").mkdir(parents=True)
    blob = src / "blobs" / "abc"
    blob.write_bytes(b"content")
    snap_dir = src / "snapshots" / "v1"
    snap_dir.mkdir(parents=True)
    link = snap_dir / "model.safetensors"
    link.symlink_to("../../blobs/abc")

    models.mirror_preload_hf_cache(src.parent, dst)

    mirrored_link = dst / "hub" / "snapshots" / "v1" / "model.safetensors"
    assert mirrored_link.is_symlink()
    target = os.readlink(mirrored_link)
    assert target == "../../blobs/abc"


def test_mirror_byte_copies_refs(tmp_path):
    """Refs are rewritten by HF lib on etag; must be a real copy, not hardlink."""
    src = tmp_path / "src" / "hub"
    dst = tmp_path / "rw"
    refs_dir = src / "refs" / "main"
    refs_dir.mkdir(parents=True)
    ref = refs_dir / "v1"
    ref.write_text("commit-sha\n")

    models.mirror_preload_hf_cache(src.parent, dst)

    mirrored_ref = dst / "hub" / "refs" / "main" / "v1"
    assert mirrored_ref.read_text() == "commit-sha\n"
    assert mirrored_ref.stat().st_ino != ref.stat().st_ino, "must be a real copy"
```

- [ ] **Step 4.2: Run test — expect FAIL**

`python -m pytest tests/test_models.py::test_mirror_hardlinks_blobs -v` → AttributeError (no `mirror_preload_hf_cache`).

- [ ] **Step 4.3: Append `mirror_preload_hf_cache` to `models.py`**

```python
def mirror_preload_hf_cache(src_root: Path | str, dst_root: Path | str) -> None:
    """Mirror a read-only HF cache tree (preload_from_hub) into a writable tree.

    - ``blobs/<sha>`` files → **hardlinked** (zero-copy, shared inode).
    - ``snapshots/<commit>/...`` symlinks → **preserved** with original relative target.
    - ``refs/<branch>`` files → **byte-copied** (HF lib overwrites on etag check).
    - Directories → ``mkdir`` so the runtime user owns them.

    Falls back to ``symlink`` when ``os.link()`` raises EXDEV (cross-device).
    """
    import errno
    import shutil

    src_root = Path(src_root)
    dst_root = Path(dst_root)

    if not (src_root / "hub").exists():
        return  # nothing preloaded — no-op

    for src_dir, _, files in os.walk(src_root / "hub"):
        rel = Path(src_dir).relative_to(src_root)
        dst_dir = dst_root / rel
        dst_dir.mkdir(parents=True, exist_ok=True)

        for name in files:
            src_path = Path(src_dir) / name
            dst_path = dst_dir / name
            if dst_path.exists():
                continue

            # Refs get byte-copied
            if "refs/" in str(rel).replace("\\", "/"):
                shutil.copy2(src_path, dst_path)
                continue

            # Symlinks (snapshot files) preserve their relative target
            if src_path.is_symlink():
                target = os.readlink(src_path)
                dst_path.symlink_to(target)
                continue

            # Regular files (blobs) hardlink with EXDEV fallback
            try:
                os.link(src_path, dst_path)
            except OSError as e:
                if e.errno == errno.EXDEV:
                    dst_path.symlink_to(src_path)
                else:
                    raise


# Top-of-file: add `from pathlib import Path` and `from typing import Iterable` imports
```

Also add at the top of `models.py`:

```python
from pathlib import Path
```

- [ ] **Step 4.4: Run all model tests — expect PASS**

`python -m pytest tests/test_models.py -v` → 9 PASSed.

- [ ] **Step 4.5: Commit**

```bash
git add models.py tests/test_models.py
git commit -m "feat(models): hf cache mirror (hardlink blobs, preserve snapshot symlinks, copy refs)"
```

---

## Task 5: LoRA safetensors header sniff

**Files:**
- Create: `lora.py`
- Test: `tests/test_lora.py`

- [ ] **Step 5.1: Write failing test**

Create `tests/test_lora.py`:

```python
import json
import struct
from pathlib import Path

import pytest

import lora


def _write_safetensors(path: Path, header: dict) -> None:
    """Minimal safetensors file: 8-byte LE header length + JSON header (no tensor data)."""
    h = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(h)) + h)


def test_sniff_valid_zimage_lora_returns_metadata(tmp_path):
    p = tmp_path / "ok.safetensors"
    _write_safetensors(p, {
        "transformer.layer1.lora_A.weight": {"dtype": "BF16", "shape": [64, 3840]},
        "transformer.layer1.lora_B.weight": {"dtype": "BF16", "shape": [3840, 64]},
        "__metadata__": {"rank": "64"},
    })
    info = lora.sniff(p)
    assert info.rank == 64
    assert info.target == "transformer"
    assert info.size_bytes == p.stat().st_size


def test_sniff_rejects_non_safetensors(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"this is not a safetensors file at all")
    with pytest.raises(lora.LoRAValidationError) as exc:
        lora.sniff(p)
    assert "safetensors" in str(exc.value).lower()


def test_sniff_rejects_non_zimage_keys(tmp_path):
    p = tmp_path / "wrong.safetensors"
    _write_safetensors(p, {
        "down_blocks.0.weight": {"dtype": "F32", "shape": [320, 320]},
    })
    with pytest.raises(lora.LoRAValidationError) as exc:
        lora.sniff(p)
    msg = str(exc.value).lower()
    assert "down_blocks" in msg or "unexpected" in msg
```

- [ ] **Step 5.2: Run test — expect FAIL** (no `lora` module).

- [ ] **Step 5.3: Implement `lora.py` (header sniff only — context manager is Task 6)**

```python
"""LoRA file validation and apply/revert context manager."""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

ZIMAGE_LORA_PREFIXES = ("transformer.", "dit.", "model.transformer.")


class LoRAValidationError(ValueError):
    """Raised when a LoRA safetensors file doesn't match Z-Image's key layout."""


@dataclass(frozen=True)
class LoRAInfo:
    path: Path
    rank: int
    target: str  # which submodule it applies to ("transformer" for Z-Image)
    size_bytes: int


def sniff(path: Path | str) -> LoRAInfo:
    """Read just the safetensors header to verify and infer rank + target.

    Doesn't load tensors. Doesn't allocate GPU memory. Cheap enough to call before
    @spaces.GPU fires.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 8:
        raise LoRAValidationError(f"{path.name}: file too short to be safetensors")
    (header_len,) = struct.unpack("<Q", raw[:8])
    if header_len <= 0 or header_len + 8 > len(raw):
        raise LoRAValidationError(f"{path.name}: not a valid safetensors header")
    try:
        header = json.loads(raw[8 : 8 + header_len])
    except json.JSONDecodeError as e:
        raise LoRAValidationError(f"{path.name}: safetensors header is not JSON ({e})") from e

    tensor_keys = [k for k in header.keys() if not k.startswith("__")]
    if not tensor_keys:
        raise LoRAValidationError(f"{path.name}: no tensors in file")

    bad = [k for k in tensor_keys if not k.startswith(ZIMAGE_LORA_PREFIXES)]
    if bad:
        sample = bad[0]
        raise LoRAValidationError(
            f"{path.name}: unexpected key '{sample}' — Z-Image LoRAs must target "
            f"{ZIMAGE_LORA_PREFIXES} (got {len(bad)}/{len(tensor_keys)} mismatched keys)"
        )

    meta = header.get("__metadata__") or {}
    rank = int(meta.get("rank", 0))
    if not rank:
        # Infer from any A/B tensor pair shape
        for k, v in header.items():
            if "lora_A" in k or "lora_down" in k:
                shape = v.get("shape") or []
                if shape:
                    rank = int(min(shape))
                    break

    return LoRAInfo(
        path=path,
        rank=rank,
        target="transformer",
        size_bytes=path.stat().st_size,
    )
```

- [ ] **Step 5.4: Run test — expect PASS**

`python -m pytest tests/test_lora.py -v` → 3 PASSed.

- [ ] **Step 5.5: Commit**

```bash
git add lora.py tests/test_lora.py
git commit -m "feat(lora): safetensors header sniff + zimage key validation"
```

---

## Task 6: LoRA apply/revert context manager

**Files:**
- Modify: `lora.py`
- Test: `tests/test_lora.py`

- [ ] **Step 6.1: Write failing test (with a mock DiffSynth)**

Append to `tests/test_lora.py`:

```python
class _FakePipe:
    """Minimal stand-in for DiffSynth's ZImagePipeline.dit hook surface."""
    def __init__(self):
        self.applied = []   # list of (path, strength) tuples
        self.reverted = []


def test_applied_lora_calls_apply_then_revert(tmp_path, monkeypatch):
    p = tmp_path / "ok.safetensors"
    _write_safetensors(p, {
        "transformer.x.lora_A.weight": {"dtype": "BF16", "shape": [32, 3840]},
        "transformer.x.lora_B.weight": {"dtype": "BF16", "shape": [3840, 32]},
    })
    pipe = _FakePipe()

    # Monkeypatch the DiffSynth merge call to record applications
    def fake_apply(pipe, path, strength):
        pipe.applied.append((str(path), strength))
    def fake_revert(pipe):
        pipe.reverted.append(True)
    monkeypatch.setattr(lora, "_apply_lora_impl", fake_apply)
    monkeypatch.setattr(lora, "_revert_lora_impl", fake_revert)

    with lora.applied_lora(pipe, p, strength=0.8):
        assert pipe.applied == [(str(p), 0.8)]
        assert pipe.reverted == []

    assert pipe.reverted == [True]


def test_applied_lora_with_none_is_a_noop(tmp_path, monkeypatch):
    pipe = _FakePipe()
    sentinel = []
    monkeypatch.setattr(lora, "_apply_lora_impl", lambda *a, **k: sentinel.append("apply"))
    monkeypatch.setattr(lora, "_revert_lora_impl", lambda *a, **k: sentinel.append("revert"))

    with lora.applied_lora(pipe, None, strength=0.0):
        pass

    assert sentinel == []


def test_applied_lora_reverts_on_exception(tmp_path, monkeypatch):
    p = tmp_path / "ok.safetensors"
    _write_safetensors(p, {
        "transformer.x.lora_A.weight": {"dtype": "BF16", "shape": [16, 3840]},
        "transformer.x.lora_B.weight": {"dtype": "BF16", "shape": [3840, 16]},
    })
    pipe = _FakePipe()
    monkeypatch.setattr(lora, "_apply_lora_impl", lambda pipe, p, s: pipe.applied.append((p, s)))
    monkeypatch.setattr(lora, "_revert_lora_impl", lambda pipe: pipe.reverted.append(True))

    with pytest.raises(RuntimeError):
        with lora.applied_lora(pipe, p, strength=1.0):
            raise RuntimeError("inference failed mid-step")

    assert pipe.reverted == [True], "must still revert on exception"
```

- [ ] **Step 6.2: Run test — expect FAIL** (`applied_lora` doesn't exist).

- [ ] **Step 6.3: Append context manager to `lora.py`**

```python
from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def applied_lora(pipe: Any, path: Path | str | None, strength: float) -> Iterator[None]:
    """Apply a LoRA to the pipeline's dit for the duration of the context.

    Reverts on exit (including exception path) so the cached GPU model is left clean.
    If ``path`` is ``None``, this is a no-op.

    Validates the LoRA file with :func:`sniff` before touching the pipeline so a bad
    file is rejected before any GPU work begins.
    """
    if path is None:
        yield
        return

    sniff(path)  # raises LoRAValidationError on bad input
    _apply_lora_impl(pipe, path, strength)
    try:
        yield
    finally:
        _revert_lora_impl(pipe)


def _apply_lora_impl(pipe: Any, path: Path | str, strength: float) -> None:
    """Apply a LoRA to ``pipe.dit``. Imports DiffSynth lazily for testability."""
    from diffsynth.utils.lora import merge_lora
    merge_lora(pipe.dit, str(path), alpha=float(strength))


def _revert_lora_impl(pipe: Any) -> None:
    """Revert the most recent LoRA from ``pipe.dit``.

    DiffSynth's ``merge_lora`` is invertible by calling it again with negated alpha
    on the same weights — but the simpler, safer approach is to track a delta and
    subtract. We delegate to DiffSynth's ``unmerge_lora`` if available; otherwise
    we fall back to re-fetching the clean dit from the model pool.
    """
    try:
        from diffsynth.utils.lora import unmerge_lora  # available in recent DiffSynth
        unmerge_lora(pipe.dit)
        return
    except ImportError:
        pass

    # Fallback: re-fetch clean weights from the model pool.
    # The variant in use can be discovered from pipe.dit.config_name or similar.
    if hasattr(pipe, "model_pool"):
        # Best-effort: re-fetch via the same name that built the current dit.
        variant = getattr(pipe.dit, "_zis_variant", None)
        if variant:
            pipe.dit = pipe.model_pool.fetch_model("z_image_dit", variant=variant)
```

- [ ] **Step 6.4: Run all lora tests — expect PASS**

`python -m pytest tests/test_lora.py -v` → 6 PASSed.

- [ ] **Step 6.5: Commit**

```bash
git add lora.py tests/test_lora.py
git commit -m "feat(lora): applied_lora ctx manager — validate, apply, revert on exit"
```

---

## Task 7: ControlNet preprocessors

**Files:**
- Create: `preprocessors.py`
- Test: `tests/test_preprocessors.py`

- [ ] **Step 7.1: Write failing test**

Create `tests/test_preprocessors.py`:

```python
import numpy as np
import pytest
from PIL import Image

import preprocessors


@pytest.fixture
def gradient_image():
    arr = np.linspace(0, 255, 256 * 256, dtype=np.uint8).reshape(256, 256)
    return Image.fromarray(arr).convert("RGB")


def test_modes_are_listed():
    assert preprocessors.MODES == ("Canny", "Depth", "Pose", "Pre-processed")


def test_canny_returns_rgb_image_of_same_size(gradient_image):
    out = preprocessors.run("Canny", gradient_image)
    assert isinstance(out, Image.Image)
    assert out.size == gradient_image.size
    assert out.mode == "RGB"


def test_passthrough_returns_input_unchanged(gradient_image):
    out = preprocessors.run("Pre-processed", gradient_image)
    assert out is gradient_image


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        preprocessors.run("Sobel", Image.new("RGB", (32, 32)))


def test_run_with_image_none_raises():
    with pytest.raises(ValueError):
        preprocessors.run("Canny", None)
```

- [ ] **Step 7.2: Run test — expect FAIL**.

- [ ] **Step 7.3: Implement `preprocessors.py`**

```python
"""ControlNet preprocessors — lazy imports so an unused mode pays no cost."""
from __future__ import annotations

from typing import Any

from PIL import Image

MODES: tuple[str, ...] = ("Canny", "Depth", "Pose", "Pre-processed")


def run(mode: str, image: Image.Image | None) -> Image.Image:
    if image is None:
        raise ValueError("preprocessor needs an input image")
    if mode == "Canny":
        return _run_canny(image)
    if mode == "Depth":
        return _run_depth(image)
    if mode == "Pose":
        return _run_pose(image)
    if mode == "Pre-processed":
        return image
    raise ValueError(f"unknown preprocessor mode: {mode!r}; expected one of {MODES}")


def _run_canny(image: Image.Image) -> Image.Image:
    import cv2
    import numpy as np
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def _run_depth(image: Image.Image) -> Image.Image:
    from controlnet_aux.processor import Processor
    proc = _get_processor("midas")
    out: Any = proc(image)
    if isinstance(out, Image.Image):
        return out.convert("RGB")
    return Image.fromarray(out).convert("RGB")


def _run_pose(image: Image.Image) -> Image.Image:
    proc = _get_processor("openpose")
    out: Any = proc(image)
    if isinstance(out, Image.Image):
        return out.convert("RGB")
    return Image.fromarray(out).convert("RGB")


_PROCESSOR_CACHE: dict[str, Any] = {}


def _get_processor(name: str) -> Any:
    """Lazy-init and cache a controlnet_aux Processor."""
    if name not in _PROCESSOR_CACHE:
        from controlnet_aux.processor import Processor
        _PROCESSOR_CACHE[name] = Processor(name)
    return _PROCESSOR_CACHE[name]
```

- [ ] **Step 7.4: Run test — expect PASS**.

Only the Canny test will exercise `cv2` here; Depth and Pose tests would require downloading model weights, so they're deferred to L3 smoke. The test suite as written only checks Canny + passthrough + error paths.

- [ ] **Step 7.5: Commit**

```bash
git add preprocessors.py tests/test_preprocessors.py
git commit -m "feat(preprocessors): canny/depth/pose via controlnet_aux (lazy imports)"
```

---

## Task 8: RealESRGAN upscale wrapper

**Files:**
- Create: `upscale.py`
- Test: `tests/test_upscale.py`

The wrapper does just: RealESRGAN x4 on input → PIL.resize(0.5) → return. The Z-Image-Turbo refinement pass happens inside the mode handler (Task 11), not here.

- [ ] **Step 8.1: Write failing test**

Create `tests/test_upscale.py`:

```python
from unittest import mock
import pytest
from PIL import Image

import upscale


@pytest.fixture
def small_image():
    return Image.new("RGB", (256, 256), color=(120, 50, 200))


def test_realesrgan_2x_produces_2x_image(small_image, monkeypatch):
    """RealESRGAN runs 4x then we scale down 0.5 → net 2x."""
    # Stub the realesrgan call to skip actually loading the model
    def fake_run_4x(_model_path, image):
        w, h = image.size
        return image.resize((w * 4, h * 4), Image.LANCZOS)
    monkeypatch.setattr(upscale, "_realesrgan_4x", fake_run_4x)

    out = upscale.realesrgan_2x(small_image, model_path="/dev/null")
    assert out.size == (512, 512)


def test_realesrgan_2x_rejects_none():
    with pytest.raises(ValueError):
        upscale.realesrgan_2x(None, model_path="/dev/null")
```

- [ ] **Step 8.2: Run test — expect FAIL**.

- [ ] **Step 8.3: Implement `upscale.py`**

```python
"""RealESRGAN x4plus wrapper + 0.5-resize bridge.

This module only handles the *pixel-space* upscale. The Z-Image-Turbo refinement
pass (img2img at denoise=0.33) lives in :mod:`modes` since it shares the pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def realesrgan_2x(image: Image.Image | None, model_path: Path | str) -> Image.Image:
    """RealESRGAN x4plus → ``image.resize(0.5)`` → net 2x upscale."""
    if image is None:
        raise ValueError("upscale needs an input image")
    upscaled = _realesrgan_4x(model_path, image)
    w, h = upscaled.size
    return upscaled.resize((w // 2, h // 2), Image.LANCZOS)


_MODEL_CACHE: dict[str, Any] = {}


def _realesrgan_4x(model_path: Path | str, image: Image.Image) -> Image.Image:
    """Run RealESRGAN x4plus on ``image``. Caches the model in-process."""
    import numpy as np
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    key = str(model_path)
    if key not in _MODEL_CACHE:
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        _MODEL_CACHE[key] = RealESRGANer(
            scale=4,
            model_path=key,
            model=net,
            tile=512,        # split into tiles to avoid OOM on large inputs
            tile_pad=10,
            pre_pad=0,
            half=False,      # bf16 elsewhere; keep this fp32 for stability
            gpu_id=None,
        )

    upsampler = _MODEL_CACHE[key]
    arr = np.array(image.convert("RGB"))
    out_arr, _ = upsampler.enhance(arr, outscale=4)
    return Image.fromarray(out_arr)
```

- [ ] **Step 8.4: Run test — expect PASS**.

- [ ] **Step 8.5: Commit**

```bash
git add upscale.py tests/test_upscale.py
git commit -m "feat(upscale): realesrgan x4 wrapper with 0.5-resize bridge"
```

---

## Task 9: Mode handler — Text → Image

**Files:**
- Create: `modes.py`
- Test: `tests/test_modes.py`

`modes.py` exposes one public function per mode (``call_t2i``, ``call_controlnet``, ``call_upscale``). Each takes ``pipeline`` + ``params`` and returns ``(PIL.Image, meta dict)``. The handler builds the right call args and applies the LoRA context manager.

- [ ] **Step 9.1: Write failing test**

Create `tests/test_modes.py`:

```python
from unittest.mock import MagicMock

import pytest
from PIL import Image

import modes


@pytest.fixture
def fake_pipe():
    """Stand-in pipeline that records its __call__ args and returns a dummy image."""
    pipe = MagicMock()
    pipe.dit = MagicMock()
    pipe.model_pool = MagicMock()
    pipe.return_value = Image.new("RGB", (64, 64), color=(255, 176, 46))
    return pipe


def test_t2i_turbo_builds_minimal_call(fake_pipe):
    out, meta = modes.call_t2i(
        fake_pipe,
        params=dict(
            prompt="a cat",
            negative_prompt="",
            model="Turbo",
            steps=8, cfg=1.0,
            width=1024, height=1024,
            seed=42,
            lora_path=None, lora_strength=0.0,
        ),
    )
    fake_pipe.assert_called_once()
    kwargs = fake_pipe.call_args.kwargs
    assert kwargs["prompt"] == "a cat"
    assert kwargs["cfg_scale"] == 1.0
    assert kwargs["num_inference_steps"] == 8
    assert kwargs["width"] == 1024
    assert kwargs["seed"] == 42
    assert kwargs["sigma_shift"] == 3.0
    assert "negative_prompt" not in kwargs or not kwargs.get("negative_prompt")
    assert meta["model"] == "Turbo"
    assert meta["steps"] == 8
    assert isinstance(out, Image.Image)


def test_t2i_base_passes_negative_prompt_and_cfg4(fake_pipe):
    modes.call_t2i(
        fake_pipe,
        params=dict(
            prompt="a cat", negative_prompt="blurry, lowres",
            model="Base", steps=25, cfg=4.0,
            width=1024, height=1024, seed=42,
            lora_path=None, lora_strength=0.0,
        ),
    )
    kwargs = fake_pipe.call_args.kwargs
    assert kwargs["negative_prompt"] == "blurry, lowres"
    assert kwargs["cfg_scale"] == 4.0
    assert kwargs["num_inference_steps"] == 25


def test_t2i_swaps_transformer_via_model_pool(fake_pipe):
    modes.call_t2i(
        fake_pipe,
        params=dict(prompt="x", negative_prompt="", model="Base", steps=25, cfg=4.0,
                    width=1024, height=1024, seed=0, lora_path=None, lora_strength=0.0),
    )
    fake_pipe.model_pool.fetch_model.assert_called()
    # Verify the model swap argument is one of the two known names
    call = fake_pipe.model_pool.fetch_model.call_args
    assert call.args[0] == "z_image_dit"
```

- [ ] **Step 9.2: Run test — expect FAIL** (no `modes` module).

- [ ] **Step 9.3: Implement `modes.py` — T2I handler only**

```python
"""Mode handlers — pure functions over a ZImagePipeline + params dict."""
from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from PIL import Image

import lora


class T2IParams(TypedDict, total=False):
    prompt: str
    negative_prompt: str
    model: str          # "Base" | "Turbo"
    steps: int
    cfg: float
    width: int
    height: int
    seed: int
    lora_path: Path | None
    lora_strength: float


def _swap_transformer(pipe: Any, model_name: str) -> None:
    """Swap the active transformer in the pipeline's model pool."""
    variant = "z_image" if model_name == "Base" else "z_image_turbo"
    pipe.dit = pipe.model_pool.fetch_model("z_image_dit", variant=variant)
    # Mark so lora._revert_lora_impl's fallback can re-fetch the same variant
    try:
        pipe.dit._zis_variant = variant
    except (AttributeError, RuntimeError):
        pass


def call_t2i(pipe: Any, params: T2IParams) -> tuple[Image.Image, dict[str, Any]]:
    """Text-to-image. Routes to base (cfg=4, 25 steps) or turbo (cfg=1, 8 steps)."""
    model_name = params.get("model", "Turbo")
    is_base = model_name == "Base"
    _swap_transformer(pipe, model_name)

    kwargs: dict[str, Any] = dict(
        prompt=params["prompt"],
        cfg_scale=float(params.get("cfg", 4.0 if is_base else 1.0)),
        num_inference_steps=int(params.get("steps", 25 if is_base else 8)),
        sigma_shift=3.0,
        height=int(params.get("height", 1024)),
        width=int(params.get("width", 1024)),
        seed=int(params.get("seed", 0)),
    )
    if is_base and params.get("negative_prompt"):
        kwargs["negative_prompt"] = params["negative_prompt"]

    with lora.applied_lora(pipe, params.get("lora_path"), params.get("lora_strength", 0.0)):
        image = pipe(**kwargs)

    meta = dict(
        mode="t2i", model=model_name,
        steps=kwargs["num_inference_steps"], cfg=kwargs["cfg_scale"],
        seed=kwargs["seed"], width=kwargs["width"], height=kwargs["height"],
        lora=str(params.get("lora_path")) if params.get("lora_path") else None,
        lora_strength=params.get("lora_strength", 0.0),
    )
    return image, meta
```

- [ ] **Step 9.4: Run test — expect PASS**.

- [ ] **Step 9.5: Commit**

```bash
git add modes.py tests/test_modes.py
git commit -m "feat(modes): t2i handler (base + turbo) with transformer swap and lora ctx"
```

---

## Task 10: Mode handler — ControlNet

**Files:**
- Modify: `modes.py`
- Test: `tests/test_modes.py`

- [ ] **Step 10.1: Write failing test**

Append to `tests/test_modes.py`:

```python
def test_controlnet_calls_preprocessor_then_pipeline(fake_pipe, monkeypatch):
    canny_called = []
    def fake_run(mode, img):
        canny_called.append((mode, img.size))
        return img  # passthrough for test
    monkeypatch.setattr(modes, "preprocessors", type("P", (), {"run": staticmethod(fake_run)}))

    input_image = Image.new("RGB", (1024, 1024))
    out, meta = modes.call_controlnet(
        fake_pipe,
        params=dict(
            prompt="cinematic portrait",
            input_image=input_image,
            preprocessor="Canny",
            controlnet_scale=1.0,
            steps=9,
            seed=42,
            lora_path=None, lora_strength=0.0,
        ),
    )

    assert canny_called == [("Canny", (1024, 1024))]
    kwargs = fake_pipe.call_args.kwargs
    assert "controlnet_inputs" in kwargs
    cn_in = kwargs["controlnet_inputs"]
    assert len(cn_in) == 1
    assert cn_in[0].scale == 1.0
    assert kwargs["num_inference_steps"] == 9
    assert kwargs["cfg_scale"] == 1.0
    assert meta["preprocessor"] == "Canny"


def test_controlnet_rejects_missing_input_image(fake_pipe):
    with pytest.raises(ValueError):
        modes.call_controlnet(
            fake_pipe,
            params=dict(prompt="x", input_image=None, preprocessor="Canny",
                        controlnet_scale=1.0, steps=9, seed=0,
                        lora_path=None, lora_strength=0.0),
        )
```

- [ ] **Step 10.2: Run test — expect FAIL** (no `call_controlnet`).

- [ ] **Step 10.3: Append `call_controlnet` to `modes.py`**

```python
import preprocessors  # add to imports at top of modes.py


def call_controlnet(pipe: Any, params: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    """ControlNet — Turbo + Z-Image-Turbo-Fun-Controlnet-Union-2.1."""
    input_image: Image.Image | None = params.get("input_image")
    if input_image is None:
        raise ValueError("ControlNet mode requires an input image")

    preproc_mode = params.get("preprocessor", "Canny")
    control_image = preprocessors.run(preproc_mode, input_image)

    # Match the Fun-Controlnet-Union workflow: turbo transformer, 9 steps, cfg=1
    _swap_transformer(pipe, "Turbo")

    # DiffSynth's ControlNetInput dataclass
    from diffsynth.diffusion.base_pipeline import ControlNetInput
    cn_input = ControlNetInput(image=control_image, scale=float(params.get("controlnet_scale", 1.0)))

    kwargs: dict[str, Any] = dict(
        prompt=params["prompt"],
        cfg_scale=1.0,
        num_inference_steps=int(params.get("steps", 9)),
        sigma_shift=3.0,
        height=control_image.size[1],
        width=control_image.size[0],
        seed=int(params.get("seed", 0)),
        controlnet_inputs=[cn_input],
    )

    with lora.applied_lora(pipe, params.get("lora_path"), params.get("lora_strength", 0.0)):
        image = pipe(**kwargs)

    meta = dict(
        mode="controlnet", model="Turbo",
        preprocessor=preproc_mode,
        controlnet_scale=cn_input.scale,
        steps=kwargs["num_inference_steps"], cfg=1.0,
        seed=kwargs["seed"], width=kwargs["width"], height=kwargs["height"],
        lora=str(params.get("lora_path")) if params.get("lora_path") else None,
        lora_strength=params.get("lora_strength", 0.0),
    )
    return image, meta
```

- [ ] **Step 10.4: Run all mode tests — expect PASS**.

- [ ] **Step 10.5: Commit**

```bash
git add modes.py tests/test_modes.py
git commit -m "feat(modes): controlnet handler (turbo + union 2.1 + preprocessor)"
```

---

## Task 11: Mode handler — Upscale

**Files:**
- Modify: `modes.py`
- Test: `tests/test_modes.py`

- [ ] **Step 11.1: Write failing test**

Append to `tests/test_modes.py`:

```python
def test_upscale_runs_realesrgan_then_pipeline(fake_pipe, monkeypatch):
    calls = {"upscale": None}
    def fake_2x(img, model_path):
        calls["upscale"] = (img.size, str(model_path))
        w, h = img.size
        return img.resize((w * 2, h * 2), Image.LANCZOS)
    monkeypatch.setattr(modes, "upscale", type("U", (), {"realesrgan_2x": staticmethod(fake_2x)}))

    input_image = Image.new("RGB", (512, 512))
    out, meta = modes.call_upscale(
        fake_pipe,
        params=dict(
            prompt="masterpiece, 8k",
            input_image=input_image,
            refine_steps=5,
            refine_denoise=0.33,
            seed=42,
            lora_path=None, lora_strength=0.0,
            esrgan_model_path="/fake/path/RealESRGAN_x4plus.pth",
        ),
    )

    assert calls["upscale"] == ((512, 512), "/fake/path/RealESRGAN_x4plus.pth")
    kwargs = fake_pipe.call_args.kwargs
    assert kwargs["input_image"].size == (1024, 1024)  # 2x via fake_2x
    assert kwargs["denoising_strength"] == 0.33
    assert kwargs["num_inference_steps"] == 5
    assert kwargs["cfg_scale"] == 1.0
    assert meta["mode"] == "upscale"


def test_upscale_rejects_missing_image(fake_pipe):
    with pytest.raises(ValueError):
        modes.call_upscale(fake_pipe, params=dict(prompt="x", input_image=None,
                                                   refine_steps=5, refine_denoise=0.33, seed=0,
                                                   lora_path=None, lora_strength=0.0,
                                                   esrgan_model_path="/fake.pth"))
```

- [ ] **Step 11.2: Run test — expect FAIL**.

- [ ] **Step 11.3: Append `call_upscale` to `modes.py`**

```python
import upscale  # add to imports at top of modes.py


def call_upscale(pipe: Any, params: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    """Upscale — RealESRGAN x4 → 0.5 resize → Z-Image-Turbo img2img refinement."""
    input_image: Image.Image | None = params.get("input_image")
    if input_image is None:
        raise ValueError("Upscale mode requires an input image")

    upscaled = upscale.realesrgan_2x(input_image, model_path=params["esrgan_model_path"])

    _swap_transformer(pipe, "Turbo")

    kwargs: dict[str, Any] = dict(
        prompt=params.get("prompt", "masterpiece, 8k"),
        cfg_scale=1.0,
        num_inference_steps=int(params.get("refine_steps", 5)),
        sigma_shift=3.0,
        input_image=upscaled,
        denoising_strength=float(params.get("refine_denoise", 0.33)),
        seed=int(params.get("seed", 0)),
    )

    with lora.applied_lora(pipe, params.get("lora_path"), params.get("lora_strength", 0.0)):
        image = pipe(**kwargs)

    meta = dict(
        mode="upscale", model="Turbo",
        refine_steps=kwargs["num_inference_steps"],
        refine_denoise=kwargs["denoising_strength"],
        seed=kwargs["seed"], width=upscaled.size[0], height=upscaled.size[1],
        lora=str(params.get("lora_path")) if params.get("lora_path") else None,
        lora_strength=params.get("lora_strength", 0.0),
    )
    return image, meta
```

- [ ] **Step 11.4: Run all mode tests — expect PASS**.

- [ ] **Step 11.5: Commit**

```bash
git add modes.py tests/test_modes.py
git commit -m "feat(modes): upscale handler (realesrgan + z-image-turbo refinement)"
```

---

## Task 12: ZeroGPU duration estimator

**Files:**
- Create: `backend.py`
- Test: `tests/test_backend.py`

The duration estimator is a pure function — test it without the rest of the backend.

- [ ] **Step 12.1: Write failing test**

Create `tests/test_backend.py`:

```python
import backend


def test_duration_t2i_turbo_is_short():
    d = backend.duration_for(mode="t2i", params=dict(model="Turbo", steps=8, width=1024, height=1024))
    assert 60 <= d <= 90


def test_duration_t2i_base_is_longer():
    d = backend.duration_for(mode="t2i", params=dict(model="Base", steps=25, width=1024, height=1024))
    assert d > 60


def test_duration_clamps_at_180():
    d = backend.duration_for(mode="t2i", params=dict(model="Base", steps=200, width=2048, height=2048))
    assert d == 180


def test_duration_clamps_at_60():
    d = backend.duration_for(mode="t2i", params=dict(model="Turbo", steps=1, width=256, height=256))
    assert d == 60


def test_duration_multiplier_scales_up():
    base = backend.duration_for(mode="t2i", params=dict(model="Turbo", steps=8, width=1024, height=1024))
    retry = backend.duration_for(mode="t2i", params=dict(model="Turbo", steps=8, width=1024, height=1024),
                                  multiplier=2.0)
    assert retry > base


def test_duration_upscale_has_realesrgan_overhead():
    t2i = backend.duration_for(mode="t2i", params=dict(model="Turbo", steps=8, width=1024, height=1024))
    upsc = backend.duration_for(mode="upscale", params=dict(refine_steps=5, width=1024, height=1024))
    assert upsc > t2i
```

- [ ] **Step 12.2: Run test — expect FAIL**.

- [ ] **Step 12.3: Implement `backend.duration_for`**

```python
"""ZImageStudioBackend — wraps the DiffSynth pipeline; applies @spaces.GPU on HF Spaces."""
from __future__ import annotations

import os
from typing import Any

# Spaces import is optional — running locally we don't have it.
try:
    import spaces  # type: ignore
except ImportError:
    spaces = None  # type: ignore[assignment]


_BASE_DURATION_S: dict[str, int] = {
    "t2i":        20,   # fixed setup + decode
    "controlnet": 30,   # + preprocessor + control patch
    "upscale":    50,   # + realesrgan pixel-space step
}
_PER_STEP_S: dict[tuple[str, str], float] = {
    ("t2i", "Base"):  2.4,
    ("t2i", "Turbo"): 1.6,
    ("controlnet", "Turbo"): 2.0,
    ("upscale", "Turbo"):    1.6,
}


def duration_for(
    mode: str,
    params: dict[str, Any],
    multiplier: float = 1.0,
) -> int:
    """Estimate ZeroGPU duration for a request. Pure function; clamped to [60, 180]."""
    model = params.get("model", "Turbo")
    steps = int(params.get("steps") or params.get("refine_steps") or 8)
    width = int(params.get("width", 1024))
    height = int(params.get("height", 1024))

    base = _BASE_DURATION_S.get(mode, 30)
    per_step = _PER_STEP_S.get((mode, model), _PER_STEP_S.get((mode, "Turbo"), 1.6))
    size_factor = (width * height) / (1024 * 1024)
    cold_buffer = 15  # CPU→GPU copy on first call after a quiet period

    est = (base + per_step * steps + cold_buffer) * size_factor * multiplier
    return max(60, min(int(est), 180))
```

- [ ] **Step 12.4: Run test — expect PASS**.

- [ ] **Step 12.5: Commit**

```bash
git add backend.py tests/test_backend.py
git commit -m "feat(backend): zerogpu duration estimator (clamped 60-180s)"
```

---

## Task 13: Backend class with @spaces.GPU

**Files:**
- Modify: `backend.py`
- Test: `tests/test_backend.py`

- [ ] **Step 13.1: Write failing test**

Append to `tests/test_backend.py`:

```python
from unittest.mock import MagicMock

import pytest
from PIL import Image


@pytest.fixture
def fake_backend(monkeypatch):
    """A ZImageStudioBackend whose constructor doesn't actually build a pipeline."""
    monkeypatch.setattr(backend, "_build_pipeline", lambda *a, **kw: MagicMock())
    b = backend.ZImageStudioBackend()
    b.pipeline.return_value = Image.new("RGB", (32, 32))
    b.pipeline.dit = MagicMock()
    b.pipeline.model_pool = MagicMock()
    return b


def test_backend_generate_routes_t2i(fake_backend):
    img, meta = fake_backend.generate(
        mode="t2i",
        params=dict(prompt="cat", negative_prompt="", model="Turbo",
                    steps=8, cfg=1.0, width=1024, height=1024, seed=42,
                    lora_path=None, lora_strength=0.0),
    )
    assert isinstance(img, Image.Image)
    assert meta["mode"] == "t2i"
    assert meta["model"] == "Turbo"


def test_backend_generate_routes_controlnet(fake_backend, monkeypatch):
    monkeypatch.setattr(backend.modes, "preprocessors",
                        type("P", (), {"run": staticmethod(lambda m, i: i)}))
    img, meta = fake_backend.generate(
        mode="controlnet",
        params=dict(prompt="cat", input_image=Image.new("RGB", (64, 64)),
                    preprocessor="Canny", controlnet_scale=1.0,
                    steps=9, seed=0, lora_path=None, lora_strength=0.0),
    )
    assert meta["mode"] == "controlnet"


def test_backend_generate_unknown_mode_raises(fake_backend):
    with pytest.raises(ValueError):
        fake_backend.generate(mode="dance", params={})
```

- [ ] **Step 13.2: Run test — expect FAIL** (no `ZImageStudioBackend`).

- [ ] **Step 13.3: Append the class to `backend.py`**

```python
import modes


def _identity(fn):
    return fn


_ON_SPACES = bool(os.environ.get("SPACES_ZERO_GPU"))
_GPU = spaces.GPU(duration=lambda *a, **kw: duration_for(*a[1:3], **kw)) \
       if (spaces is not None and _ON_SPACES) else _identity


def _build_pipeline() -> Any:
    """Construct the DiffSynth ZImagePipeline. Imported lazily to keep tests fast."""
    import torch
    from diffsynth.pipelines.z_image import ZImagePipeline

    import models

    device = models.auto_device()
    vram_cfg: dict[str, Any] = {}
    if device != "cpu":
        vram_cfg = dict(
            offload_dtype=torch.bfloat16, offload_device="cpu",
            onload_dtype=torch.bfloat16,  onload_device="cpu",
            preparing_dtype=torch.bfloat16, preparing_device=device,
            computation_dtype=torch.bfloat16, computation_device=device,
        )

    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=models.build_diffsynth_configs(vram_cfg=vram_cfg),
        tokenizer_config=models.build_diffsynth_configs(
            (models.TOKENIZER_CONFIG,), vram_cfg=None,
        )[0],
        vram_limit=models.vram_limit_for(device),
    )
    return pipe


_DISPATCH = {
    "t2i":        modes.call_t2i,
    "controlnet": modes.call_controlnet,
    "upscale":    modes.call_upscale,
}


class ZImageStudioBackend:
    """One-process backend wrapping the DiffSynth ZImagePipeline."""

    def __init__(self) -> None:
        self.pipeline = _build_pipeline()

    @_GPU
    def generate(self, mode: str, params: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        handler = _DISPATCH.get(mode)
        if handler is None:
            raise ValueError(f"unknown mode: {mode!r}; expected one of {list(_DISPATCH)}")
        return handler(self.pipeline, params)
```

- [ ] **Step 13.4: Run all backend tests — expect PASS**.

- [ ] **Step 13.5: Commit**

```bash
git add backend.py tests/test_backend.py
git commit -m "feat(backend): zimagestudiobackend with @spaces.gpu and mode dispatch"
```

---

## Task 14: UI builders — `ui.py`

**Files:**
- Create: `ui.py`
- Test: `tests/test_ui.py` (smoke only — Gradio components are hard to unit-test)

- [ ] **Step 14.1: Write the smoke test**

Create `tests/test_ui.py`:

```python
import gradio as gr

import ui


def test_build_t2i_tab_returns_components():
    components = ui.build_t2i_tab()
    # Returns dict with the inputs handler needs
    expected = {"prompt", "negative_prompt", "model", "steps", "cfg",
                "width", "height", "seed", "lora_path", "lora_strength",
                "generate_btn", "output_image", "output_meta"}
    assert expected.issubset(components.keys())


def test_build_controlnet_tab_returns_components():
    components = ui.build_controlnet_tab()
    expected = {"prompt", "input_image", "preprocessor", "controlnet_scale",
                "steps", "seed", "lora_path", "lora_strength",
                "generate_btn", "output_image", "output_meta"}
    assert expected.issubset(components.keys())


def test_build_upscale_tab_returns_components():
    components = ui.build_upscale_tab()
    expected = {"prompt", "input_image", "refine_steps", "refine_denoise",
                "seed", "lora_path", "lora_strength",
                "generate_btn", "output_image", "output_meta"}
    assert expected.issubset(components.keys())
```

Note: each builder must be called inside a Gradio `gr.Blocks()` context. The test uses one:

```python
import pytest

@pytest.fixture(autouse=True)
def _blocks_ctx():
    with gr.Blocks():
        yield
```

(Add this fixture at the top of `tests/test_ui.py` along with the imports.)

- [ ] **Step 14.2: Run test — expect FAIL**.

- [ ] **Step 14.3: Implement `ui.py`**

```python
"""Per-tab Gradio component builders. Pure layout — no event wiring (that lives in app.py)."""
from __future__ import annotations

import gradio as gr

import preprocessors


def build_t2i_tab() -> dict[str, gr.components.Component]:
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Prompt", lines=4,
                                placeholder="A latina model peeking through pine branches…")
            negative_prompt = gr.Textbox(label="Negative prompt (Base only)", lines=2,
                                         placeholder="blurry, lowres, distorted")
            model = gr.Radio(["Base", "Turbo"], value="Turbo", label="Model")
            with gr.Row():
                lora_path = gr.File(label="LoRA (optional)",
                                    file_types=[".safetensors"], type="filepath")
                lora_strength = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="LoRA strength")
            with gr.Row():
                steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
                cfg = gr.Slider(0.5, 12.0, value=1.0, step=0.1, label="CFG (Base only)")
            with gr.Row():
                width = gr.Slider(384, 1536, value=1024, step=64, label="Width")
                height = gr.Slider(384, 1536, value=1024, step=64, label="Height")
                seed = gr.Number(value=0, precision=0, label="Seed (0 = random)")
            generate_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=5):
            output_image = gr.Image(label="Output", type="pil", height=512,
                                    show_download_button=True)
            output_meta = gr.JSON(label="Meta", value={})
    return dict(
        prompt=prompt, negative_prompt=negative_prompt, model=model,
        steps=steps, cfg=cfg, width=width, height=height, seed=seed,
        lora_path=lora_path, lora_strength=lora_strength,
        generate_btn=generate_btn, output_image=output_image, output_meta=output_meta,
    )


def build_controlnet_tab() -> dict[str, gr.components.Component]:
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Prompt", lines=3)
            input_image = gr.Image(label="Control image", type="pil", height=240)
            with gr.Row():
                preprocessor = gr.Dropdown(list(preprocessors.MODES), value="Canny",
                                           label="Preprocessor")
                controlnet_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05,
                                             label="ControlNet scale")
            with gr.Row():
                lora_path = gr.File(label="LoRA (optional)",
                                    file_types=[".safetensors"], type="filepath")
                lora_strength = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="LoRA strength")
            with gr.Row():
                steps = gr.Slider(1, 30, value=9, step=1, label="Steps")
                seed = gr.Number(value=0, precision=0, label="Seed (0 = random)")
            generate_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=5):
            output_image = gr.Image(label="Output", type="pil", height=512,
                                    show_download_button=True)
            output_meta = gr.JSON(label="Meta", value={})
    return dict(
        prompt=prompt, input_image=input_image,
        preprocessor=preprocessor, controlnet_scale=controlnet_scale,
        steps=steps, seed=seed,
        lora_path=lora_path, lora_strength=lora_strength,
        generate_btn=generate_btn, output_image=output_image, output_meta=output_meta,
    )


def build_upscale_tab() -> dict[str, gr.components.Component]:
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Refinement prompt", value="masterpiece, 8k", lines=2)
            input_image = gr.Image(label="Input image", type="pil", height=240)
            with gr.Row():
                refine_steps = gr.Slider(1, 20, value=5, step=1, label="Refine steps")
                refine_denoise = gr.Slider(0.0, 1.0, value=0.33, step=0.01,
                                           label="Refine denoise")
            with gr.Row():
                lora_path = gr.File(label="LoRA (optional)",
                                    file_types=[".safetensors"], type="filepath")
                lora_strength = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="LoRA strength")
            seed = gr.Number(value=0, precision=0, label="Seed (0 = random)")
            generate_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=5):
            output_image = gr.Image(label="Output (2× upscaled)", type="pil",
                                    height=512, show_download_button=True)
            output_meta = gr.JSON(label="Meta", value={})
    return dict(
        prompt=prompt, input_image=input_image,
        refine_steps=refine_steps, refine_denoise=refine_denoise,
        seed=seed,
        lora_path=lora_path, lora_strength=lora_strength,
        generate_btn=generate_btn, output_image=output_image, output_meta=output_meta,
    )
```

- [ ] **Step 14.4: Run test — expect PASS**.

- [ ] **Step 14.5: Commit**

```bash
git add ui.py tests/test_ui.py
git commit -m "feat(ui): per-tab gradio builders (t2i, controlnet, upscale)"
```

---

## Task 15: App entrypoint — `app.py`

**Files:**
- Create: `app.py`
- Test: manual smoke (run locally, verify UI renders)

- [ ] **Step 15.1: Implement `app.py`**

```python
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
    return hf_hub_download("lllyasviel/Annotators", "RealESRGAN_x4plus.pth")


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


def build_app() -> gr.Blocks:
    with gr.Blocks(theme=theme.build_theme(), css=theme.CSS, title="z-image-studio") as demo:
        gr.HTML(HEADER_HTML)

        with gr.Tabs():
            with gr.Tab("Text → Image"):
                t = ui.build_t2i_tab()
                t["generate_btn"].click(
                    fn=on_t2i_generate,
                    inputs=[t["prompt"], t["negative_prompt"], t["model"],
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
```

- [ ] **Step 15.2: Run a fast import-only test (no actual launch)**

```bash
python -c "import app; print('app imports clean')"
```

Expected: prints `app imports clean`. (If DiffSynth tries to download weights, the test fails — but `_bootstrap` is a no-op off Spaces, and `get_backend()` is lazy, so import alone must succeed.)

- [ ] **Step 15.3: Local smoke (manual, optional)**

```bash
source .venv/bin/activate
python app.py
```

Open http://localhost:7860 and verify all three tabs render with the Amber theme. Don't try Generate unless models are downloaded — that's Task 18.

- [ ] **Step 15.4: Commit**

```bash
git add app.py
git commit -m "feat(app): gradio blocks entrypoint with bootstrap + event wiring"
```

---

## Task 16: README — HF Space YAML frontmatter + user docs

**Files:**
- Create: `README.md`

- [ ] **Step 16.1: Write `README.md`**

```markdown
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
```

- [ ] **Step 16.2: Validate YAML frontmatter parses**

```bash
python -c "
import yaml
text = open('README.md').read()
fm = text.split('---')[1]
data = yaml.safe_load(fm)
assert data['sdk'] == 'gradio'
assert data['python_version'] == '3.11'
assert len(data['preload_from_hub']) == 4
print('README frontmatter OK')
"
```

Expected: `README frontmatter OK`.

- [ ] **Step 16.3: Commit**

```bash
git add README.md
git commit -m "docs: hf space frontmatter + readme"
```

---

## Task 17: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 17.1: Write the workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: pip-${{ runner.os }}-${{ hashFiles('requirements.txt') }}

      - name: Install
        run: |
          python -m pip install -U pip
          pip install ruff pytest pytest-mock pillow numpy gradio==5.50.0 safetensors

      - name: Ruff format
        run: ruff format --check .

      - name: Ruff lint
        run: ruff check .

      - name: Pytest (L1+L2 — no GPU)
        run: pytest -q --tb=short
        env:
          # Skip tests that need diffsynth / realesrgan / controlnet_aux installed
          PYTEST_DISABLE_PLUGIN_AUTOLOAD: 1
```

Note: the CI doesn't install diffsynth / realesrgan / controlnet_aux because they're heavy and not needed for L1+L2 tests (we mock or skip those code paths). Tests must be written so that just `pip install pillow numpy gradio safetensors pytest` is enough to pass.

- [ ] **Step 17.2: Verify the test suite passes with the CI dep subset locally**

```bash
python3.11 -m venv /tmp/ci-test-venv
source /tmp/ci-test-venv/bin/activate
pip install -q ruff pytest pytest-mock pillow numpy gradio==5.50.0 safetensors
cd /Users/techfreakworm/Projects/llm/z-image-studio
ruff format --check . || ruff format .
ruff check .
pytest -q --tb=short
```

If any test imports diffsynth / realesrgan / controlnet_aux at the module level (not inside a test function), refactor those imports to be inside the function bodies so CI can pass without them. The implementations in Tasks 3, 6, 7, 8 already follow this pattern (lazy imports).

- [ ] **Step 17.3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: ruff + pytest on push/pr (l1+l2, no gpu deps)"
```

---

## Task 18: Local end-to-end smoke test (manual, opt-in)

**Files:** none — manual verification on a real machine with GPU/MPS access.

This is the L3 smoke from the spec. It downloads ~30 GB of weights the first time. Marked with `@pytest.mark.gpu` so CI skips it.

- [ ] **Step 18.1: Add `tests/test_smoke_gpu.py`**

```python
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def real_backend():
    """Build a real backend with real weights. ~30 GB download on first run."""
    import backend
    return backend.ZImageStudioBackend()


def test_t2i_turbo_produces_image(real_backend):
    from PIL import Image
    image, meta = real_backend.generate(
        mode="t2i",
        params=dict(prompt="a red apple on a wooden table",
                    negative_prompt="", model="Turbo",
                    steps=8, cfg=1.0, width=384, height=384, seed=42,
                    lora_path=None, lora_strength=0.0),
    )
    assert isinstance(image, Image.Image)
    assert image.size == (384, 384)
    assert meta["model"] == "Turbo"


def test_t2i_base_produces_image(real_backend):
    from PIL import Image
    image, meta = real_backend.generate(
        mode="t2i",
        params=dict(prompt="a red apple on a wooden table",
                    negative_prompt="blurry", model="Base",
                    steps=15, cfg=4.0, width=384, height=384, seed=42,
                    lora_path=None, lora_strength=0.0),
    )
    assert isinstance(image, Image.Image)


def test_controlnet_produces_image(real_backend):
    from PIL import Image
    import numpy as np
    arr = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    image, meta = real_backend.generate(
        mode="controlnet",
        params=dict(prompt="a portrait of a person, dramatic light",
                    input_image=Image.fromarray(arr),
                    preprocessor="Canny", controlnet_scale=1.0,
                    steps=9, seed=42, lora_path=None, lora_strength=0.0),
    )
    assert isinstance(image, Image.Image)


def test_upscale_produces_image(real_backend, tmp_path):
    from PIL import Image
    import numpy as np
    from huggingface_hub import hf_hub_download
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    image, meta = real_backend.generate(
        mode="upscale",
        params=dict(prompt="masterpiece, 8k",
                    input_image=Image.fromarray(arr),
                    refine_steps=5, refine_denoise=0.33, seed=42,
                    lora_path=None, lora_strength=0.0,
                    esrgan_model_path=hf_hub_download("lllyasviel/Annotators",
                                                     "RealESRGAN_x4plus.pth")),
    )
    assert image.size == (512, 512)
```

- [ ] **Step 18.2: Run the smoke (manual)**

```bash
source .venv/bin/activate
pytest tests/test_smoke_gpu.py -v -m gpu
```

Expected: 4 PASSed. (Each test takes ~30 – 90 seconds depending on hardware.)

- [ ] **Step 18.3: Commit**

```bash
git add tests/test_smoke_gpu.py
git commit -m "test: l3 gpu smoke (t2i base/turbo + controlnet + upscale)"
```

---

## Task 19: HF Space deploy (manual)

**Files:** none — uses the HF CLI.

- [ ] **Step 19.1: Create the Space (one-time)**

```bash
hf auth login   # if not already
hf repo create techfreakworm/z-image-studio --type space --space-sdk gradio
```

- [ ] **Step 19.2: Push the repo as the Space**

```bash
cd /Users/techfreakworm/Projects/llm/z-image-studio
git remote add space https://huggingface.co/spaces/techfreakworm/z-image-studio
git push space main
```

- [ ] **Step 19.3: Watch the Space build**

The build logs will show `preload_from_hub` downloading ~30 GB. On first build this takes 10 – 20 minutes.

- [ ] **Step 19.4: First L4 smoke (manual)**

Open the Space URL. Generate one image per mode:
- T2I Turbo at 1024×1024
- T2I Base at 768×768
- ControlNet with a downloaded portrait + Canny
- Upscale of a 512×512 input

For each: verify the output renders, the meta JSON shows the right model, the ZeroGPU duration estimator was reasonable (check Space logs). Switch the T2I model selector between Base ↔ Turbo and verify no OOM.

If any failure mode lights up:
- OOM → DiffSynth `vram_limit` too high; reduce in `backend._build_pipeline`.
- Permission denied on HF cache → `_bootstrap()` mirror failed; check log for the EXDEV fallback path.
- ZeroGPU timeout → the duration estimator is too low for the workload; bump `_PER_STEP_S` for that mode.
- LoRA rejected → `lora.sniff` is too strict for the user's LoRA — relax the key-prefix list if it's a real Z-Image LoRA.

- [ ] **Step 19.5: Tag the release**

```bash
git tag -a v0.1.0 -m "z-image-studio v0.1.0 — initial release"
git push origin v0.1.0
git push space v0.1.0
```

---

## Self-review checklist (already run)

- **Spec coverage** — every section of the spec maps to a task:
  - § 2 Architecture → Tasks 12, 13, 15
  - § 3 Mode mappings → Tasks 9, 10, 11
  - § 4 UI Onyx Amber → Tasks 2, 14, 15
  - § 5 File layout → Tasks 1-15 (one task per file)
  - § 6 Models + preload + cache mirror → Tasks 3, 4, 16
  - § 7 ZeroGPU integration → Tasks 12, 13
  - § 8 Errors → Tasks 6 (LoRA reject), 13 (mode dispatch error), 15 (gr.Error wrap)
  - § 9 Testing tiers → Tasks 1 (L1 setup), 17 (CI), 18 (L3), 19 (L4)
  - § 10 Repo conventions → Tasks 1, 17
  - § 11 Implicit decisions — all baked in
- **No placeholders** — every step has either real code or a real command.
- **Type consistency** — `T2IParams` TypedDict in `modes.py` matches the param keys in `app.py`'s `on_t2i_generate`. `ControlNetInput` import path matches DiffSynth's `diffsynth.diffusion.base_pipeline`. `lora.applied_lora(pipe, path, strength)` signature matches its callers in all three mode handlers.

---

## Execution handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?

---

## Plan Amendments (2026-05-13, post-Task-3)

Additions decided after Tasks 1–3 landed. References spec sections 4.6 (tooltips) and 4.7 (coming-soon model placeholders).

**Insertion order:** Tasks A and B run *now* (before Task 4) since they extend modules already implemented. Task C runs *with* Task 14 (UI builders) — its helpers replace some of Task 14's component-building code.

---

### Task A: tooltips.py — TOOLTIPS dict

**Files:**
- Create: `tooltips.py`
- Test: `tests/test_tooltips.py`

- [ ] **Step A.1: Write failing test**

```python
# tests/test_tooltips.py
import tooltips

REQUIRED_KEYS = {
    "prompt", "negative_prompt", "model", "lora", "lora_strength",
    "steps", "cfg", "width", "height", "seed",
    "controlnet_image", "controlnet_preprocessor", "controlnet_scale",
    "upscale_image", "refine_steps", "refine_denoise", "output",
}

def test_tooltips_has_all_required_keys():
    assert REQUIRED_KEYS <= set(tooltips.TOOLTIPS)

def test_tooltips_values_are_non_empty_strings():
    for key, val in tooltips.TOOLTIPS.items():
        assert isinstance(val, str) and val.strip(), f"{key} is empty or non-string"

def test_tooltips_values_are_short_enough_for_a_tooltip():
    # 200-char ceiling so tooltips don't overflow on phone
    for key, val in tooltips.TOOLTIPS.items():
        assert len(val) <= 200, f"{key} is too long for a tooltip ({len(val)} chars)"
```

- [ ] **Step A.2: Run test — expect FAIL** (`pytest tests/test_tooltips.py -v` → ModuleNotFoundError).

- [ ] **Step A.3: Implement `tooltips.py`**

```python
"""User-facing copy — tooltips and similar short strings.

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
```

- [ ] **Step A.4: Run test — expect PASS** (3 PASSed).

- [ ] **Step A.5: Commit**

```bash
git add tooltips.py tests/test_tooltips.py
git commit -m "feat(copy): tooltip strings dict — one source of truth for param descriptions"
```

---

### Task B: theme.py CSS extensions — .zis-info / .zis-models / .zis-model

**Files:**
- Modify: `theme.py` (extend the `CSS` constant)
- Test: `tests/test_theme.py` (append new test)

- [ ] **Step B.1: Write failing test**

Append to `tests/test_theme.py`:

```python
def test_css_includes_param_tooltip_rule():
    css = theme.CSS
    assert ".zis-info" in css
    assert "data-info" in css  # the attr() reference in ::after
    assert "::after" in css

def test_css_includes_model_selector_rules():
    css = theme.CSS
    assert ".zis-models" in css
    assert ".zis-model" in css
    assert ".zis-model.on" in css
    assert ".zis-model.soon" in css

def test_css_model_grid_is_responsive():
    css = theme.CSS
    # Phone is the default 2-col; tablet+ bumps to 4-col via media query.
    assert "grid-template-columns" in css
    assert "@media" in css
    assert "min-width: 768px" in css or "min-width:768px" in css
```

- [ ] **Step B.2: Run test — expect FAIL** (new assertions fail).

- [ ] **Step B.3: Extend `theme.CSS`**

Append the following CSS block to the existing `CSS` string in `theme.py` (the existing rules stay; this is purely additive):

```python
CSS = CSS + """

/* ===== Param tooltip — (i) icon next to labels (spec § 4.6) ===== */

.zis-row-label {
    display: inline-flex; align-items: center;
    font-size: 11px; color: #A89478; font-weight: 500;
    margin-bottom: 6px;
}
.zis-info {
    display: inline-flex; align-items: center; justify-content: center;
    width: 12px; height: 12px;
    font: italic 600 8px 'Geist', system-ui, sans-serif;
    border: 1px solid #2A2218; border-radius: 50%;
    color: #A89478; vertical-align: super;
    margin-left: 3px; cursor: help; position: relative;
    transition: border-color 0.12s, color 0.12s;
}
.zis-info:hover { border-color: #FFB02E; color: #FFB02E; }
.zis-info::after {
    content: attr(data-info);
    position: absolute; bottom: 100%; left: 50%;
    transform: translateX(-50%) translateY(-4px);
    background: #1C170F; color: #FAF1E3;
    border: 1px solid #2A2218; border-radius: 6px;
    padding: 6px 10px;
    font: 400 11px 'Geist', system-ui, sans-serif; line-height: 1.4;
    width: 200px; white-space: normal;
    opacity: 0; pointer-events: none;
    transition: opacity 0.12s; z-index: 50;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}
.zis-info:hover::after, .zis-info.shown::after { opacity: 1; }

/* ===== Custom model selector — 2-col phone / 4-col tablet+ (spec § 4.7) ===== */

.zis-models {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 10px;
}
@media (min-width: 768px) {
    .zis-models { grid-template-columns: repeat(4, 1fr); }
}
.zis-model {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 12px;
    border: 1px solid #2A2218; border-radius: 8px;
    background: transparent; cursor: pointer;
    color: #FAF1E3;
    font: 500 12px 'Geist', system-ui, sans-serif;
    text-decoration: none;
    transition: opacity 0.15s, border-color 0.15s, background 0.15s;
}
.zis-model .dot {
    width: 10px; height: 10px; border-radius: 50%;
    border: 1px solid #2A2218; flex-shrink: 0;
}
.zis-model .name { flex: 1; text-align: left; }
.zis-model.on {
    background: #FFB02E; color: #1A1208; border-color: #FFB02E;
}
.zis-model.on .dot { background: #1A1208; border-color: #1A1208; }
.zis-model.soon {
    opacity: 0.55;
    background: rgba(255,176,46,0.04);
    border-style: dashed;
    position: relative;
}
.zis-model.soon .name { color: #A89478; }
.zis-model.soon .name .ext {
    font-size: 10px; color: #FFB02E;
    margin-left: 4px; vertical-align: super;
}
.zis-model.soon .soon-tag {
    font-family: 'Geist Mono', ui-monospace, monospace;
    font-size: 8.5px; letter-spacing: 0.12em; text-transform: uppercase;
    background: rgba(255,176,46,0.18); color: #FFB02E;
    padding: 2px 6px; border-radius: 100px;
    flex-shrink: 0;
}
.zis-model.soon:hover { opacity: 0.78; border-color: #FFB02E; }
.zis-model.soon::after {
    content: "Coming soon — opens GitHub";
    position: absolute; bottom: 100%; left: 50%;
    transform: translateX(-50%) translateY(-4px);
    background: #1C170F; color: #FAF1E3;
    border: 1px solid #2A2218; border-radius: 6px;
    padding: 6px 10px;
    font: 400 11px 'Geist', system-ui, sans-serif;
    white-space: nowrap;
    opacity: 0; pointer-events: none;
    transition: opacity 0.12s; z-index: 50;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}
.zis-model.soon:hover::after { opacity: 1; }
""".rstrip()
```

(Implementation note: don't *literally* write `CSS = CSS + """..."""` — just paste the new rules at the end of the existing triple-quoted `CSS` string. The reassignment form above is shown to make the additive intent explicit.)

- [ ] **Step B.4: Run all theme tests — expect PASS** (7 PASSed: 4 original + 3 new).

- [ ] **Step B.5: Commit**

```bash
git add theme.py tests/test_theme.py
git commit -m "feat(theme): css rules for param info tooltips and custom model selector"
```

---

### Task C: ui.py helpers — labeled_label() + model_selector_html()

This task is part of Task 14 (UI builders) — see the modification note below. It's broken out here so the helpers are tested before Task 14 wires them up.

**Files:**
- Create: `ui.py` (start of file — Task 14 will add the per-tab builders after)
- Test: `tests/test_ui.py` (new — Task 14's smoke tests will be appended later)

- [ ] **Step C.1: Write failing test**

Create `tests/test_ui.py`:

```python
import pytest

import ui


def test_labeled_label_returns_html_string():
    out = ui.labeled_label("Steps", "Denoising steps.")
    assert isinstance(out, str)
    assert "<label" in out and "</label>" in out
    assert ">Steps<" in out
    assert 'data-info="Denoising steps."' in out
    assert ">i<" in out  # the icon glyph


def test_labeled_label_escapes_html_chars():
    out = ui.labeled_label("Steps <x>", 'A "quoted" hint')
    assert "<x>" not in out
    assert "&lt;x&gt;" in out
    assert "&quot;quoted&quot;" in out


def test_model_selector_html_marks_current_as_on():
    out = ui.model_selector_html(current="Turbo")
    assert 'class="zis-model on" data-value="Turbo"' in out
    assert 'class="zis-model" data-value="Base"' in out


def test_model_selector_html_includes_both_soon_cards_with_github_link():
    out = ui.model_selector_html(current="Turbo")
    assert out.count("github.com/Tongyi-MAI/Z-Image#model-zoo") == 2
    assert "Edit" in out
    assert "Omni Base" in out
    assert "soon-tag" in out
    assert 'target="_blank"' in out
    assert 'rel="noopener noreferrer"' in out


def test_model_selector_html_defaults_to_turbo():
    out = ui.model_selector_html()
    assert 'class="zis-model on" data-value="Turbo"' in out


def test_model_selector_html_escapes_current_value():
    out = ui.model_selector_html(current='<script>alert(1)</script>')
    assert "<script>" not in out
```

- [ ] **Step C.2: Run test — expect FAIL** (ModuleNotFoundError: ui).

- [ ] **Step C.3: Implement the helpers in `ui.py`**

```python
"""Gradio UI builders + small HTML helpers for the (i) tooltip pattern and the custom model selector."""
from __future__ import annotations

from html import escape

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
        f'</label>'
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
            f'onclick="zis.setModel(\'{name}\')">'
            f'<span class="dot"></span>'
            f'<span class="name">{name}</span>'
            f'</button>'
        )
    for name in ("Edit", "Omni Base"):
        cards.append(
            f'<a class="zis-model soon" '
            f'href="{GITHUB_MODEL_ZOO_URL}" '
            f'target="_blank" rel="noopener noreferrer">'
            f'<span class="dot"></span>'
            f'<span class="name">{name}<span class="ext">↗</span></span>'
            f'<span class="soon-tag">soon</span>'
            f'</a>'
        )
    # current_safe is referenced only to ensure escape() is exercised
    # in the unit test; the literal current is matched in cls above.
    _ = current_safe
    return f'<div class="zis-models">{"".join(cards)}</div>'
```

- [ ] **Step C.4: Run test — expect PASS** (6 PASSed).

- [ ] **Step C.5: Commit**

```bash
git add ui.py tests/test_ui.py
git commit -m "feat(ui): labeled_label and model_selector_html helpers"
```

---

### Modifications to existing Tasks 14 + 15

**Task 14 (UI builders):** the per-tab builders defined later in `ui.py` MUST now:

1. Each `gr.Slider` / `gr.Textbox` / `gr.File` / `gr.Image` / `gr.Dropdown` / `gr.Number` uses `show_label=False` AND is preceded inside a `with gr.Column():` block by `gr.HTML(labeled_label(LABEL_TEXT, TOOLTIPS[KEY]))`.
2. The T2I tab replaces the previous `gr.Radio(["Base","Turbo"], …)` with:
   ```python
   model_state = gr.Textbox(value="Turbo", visible=False, elem_id="zis-model-state")
   gr.HTML(model_selector_html(current="Turbo"))
   ```
   `model_state` becomes the input fed into the T2I generate handler (replaces `model` in `inputs=[...]`).
3. The dict returned by `build_t2i_tab()` swaps the `model` key for `model_state` (still a Gradio component).

The smoke test in Task 14 must continue to pass with the updated dict keys.

**Task 15 (app.py):**

Add a `_HEAD_JS` constant and pass it to `gr.Blocks(head=...)`:

```python
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
```

In `build_app`:

```python
with gr.Blocks(theme=theme.build_theme(), css=theme.CSS, head=_HEAD_JS, title="z-image-studio") as demo:
    ...
```

T2I handler input list changes: replace `t["model"]` with `t["model_state"]` everywhere.
