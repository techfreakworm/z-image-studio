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
