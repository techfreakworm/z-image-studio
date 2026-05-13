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
