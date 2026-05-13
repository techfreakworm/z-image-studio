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
