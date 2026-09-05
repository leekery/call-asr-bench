from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

from callasr.dataset import load_dataset_manifest
from callasr.io import load_wav

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "examples" / "smoke"


def test_smoke_manifest_loads_with_russian_and_english_metadata() -> None:
    items = load_dataset_manifest(_FIXTURE / "dataset.jsonl")

    assert [item.id for item in items] == ["smoke-ru", "smoke-en"]
    assert [item.language for item in items] == ["ru", "en"]
    assert [item.reference for item in items] == ["синтетический тест", "synthetic test"]
    assert all(item.audio.is_file() for item in items)


def test_committed_smoke_wavs_are_tiny_pcm16_16khz_quarter_second() -> None:
    items = load_dataset_manifest(_FIXTURE / "dataset.jsonl")

    for item in items:
        with wave.open(str(item.audio), "rb") as reader:
            assert reader.getnchannels() == 1
            assert reader.getsampwidth() == 2
            assert reader.getframerate() == 16_000
            assert reader.getnframes() == 4_000
            assert reader.getcomptype() == "NONE"
        loaded = load_wav(item.audio)
        assert loaded.sample_rate == 16_000
        assert loaded.samples.size == 4_000
        assert item.audio.stat().st_size < 9_000


def test_generator_reproduces_committed_wav_bytes(tmp_path: Path) -> None:
    generator = _FIXTURE / "generate.py"
    subprocess.run(
        [sys.executable, str(generator), "--output", str(tmp_path)],
        check=True,
        cwd=_ROOT,
    )

    for name in ("tone-ru.wav", "tone-en.wav"):
        assert (tmp_path / name).read_bytes() == (_FIXTURE / "audio" / name).read_bytes()


def test_fixture_readme_makes_provenance_and_non_quality_purpose_explicit() -> None:
    text = (_FIXTURE / "README.md").read_text(encoding="utf-8").lower()

    assert "synthetic non-speech" in text
    assert "mit" in text
    assert "not a quality benchmark" in text
    assert "--model tiny" in text
    assert "first use" in text and "download" in text


def test_fixture_payload_remains_small() -> None:
    audio_dir = _FIXTURE / "audio"
    total_bytes = sum(path.stat().st_size for path in audio_dir.glob("*.wav"))

    assert total_bytes < 20_000
