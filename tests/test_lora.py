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
    _write_safetensors(
        p,
        {
            "transformer.layer1.lora_A.weight": {"dtype": "BF16", "shape": [64, 3840]},
            "transformer.layer1.lora_B.weight": {"dtype": "BF16", "shape": [3840, 64]},
            "__metadata__": {"rank": "64"},
        },
    )
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
    _write_safetensors(
        p,
        {
            "down_blocks.0.weight": {"dtype": "F32", "shape": [320, 320]},
        },
    )
    with pytest.raises(lora.LoRAValidationError) as exc:
        lora.sniff(p)
    msg = str(exc.value).lower()
    assert "down_blocks" in msg or "unexpected" in msg


class _FakePipe:
    """Minimal stand-in for DiffSynth's ZImagePipeline.dit hook surface."""

    def __init__(self):
        self.applied = []  # list of (path, strength) tuples
        self.reverted = []


def test_applied_lora_calls_apply_then_revert(tmp_path, monkeypatch):
    p = tmp_path / "ok.safetensors"
    _write_safetensors(
        p,
        {
            "transformer.x.lora_A.weight": {"dtype": "BF16", "shape": [32, 3840]},
            "transformer.x.lora_B.weight": {"dtype": "BF16", "shape": [3840, 32]},
        },
    )
    pipe = _FakePipe()

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
    _write_safetensors(
        p,
        {
            "transformer.x.lora_A.weight": {"dtype": "BF16", "shape": [16, 3840]},
            "transformer.x.lora_B.weight": {"dtype": "BF16", "shape": [3840, 16]},
        },
    )
    pipe = _FakePipe()
    monkeypatch.setattr(lora, "_apply_lora_impl", lambda pipe, p, s: pipe.applied.append((p, s)))
    monkeypatch.setattr(lora, "_revert_lora_impl", lambda pipe: pipe.reverted.append(True))

    with pytest.raises(RuntimeError):
        with lora.applied_lora(pipe, p, strength=1.0):
            raise RuntimeError("inference failed mid-step")

    assert pipe.reverted == [True], "must still revert on exception"
