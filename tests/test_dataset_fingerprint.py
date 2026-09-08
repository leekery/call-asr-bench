from __future__ import annotations

import json
import re
import wave
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from callasr.adapters.base import Transcription
from callasr.benchmark import run_benchmark
from callasr.dataset import DatasetItem, load_dataset_manifest


def _fingerprint(items: tuple[DatasetItem, ...]) -> str:
    try:
        from callasr.dataset import dataset_fingerprint
    except ImportError as exc:
        pytest.fail(f"dataset fingerprint implementation is missing: {exc}")
    return dataset_fingerprint(items)


def _item(
    audio: Path,
    *,
    item_id: str = "item",
    reference: str = "reference",
    language: str | None = "en",
    line_number: int = 1,
) -> DatasetItem:
    return DatasetItem(
        id=item_id,
        audio=audio,
        reference=reference,
        language=language,
        line_number=line_number,
    )


def _write_pcm16_wav(path: Path, *, sample_rate: int = 16_000) -> None:
    samples = np.array([0, 1000, -1000, 500, -500] * 80, dtype="<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.tobytes())


def _artifact(schema: int, *, fingerprint: str | None = None) -> dict[str, object]:
    dataset: dict[str, object] = {"path": "/tmp/dataset.jsonl", "item_count": 1}
    if schema >= 5:
        dataset["fingerprint"] = fingerprint
    return {
        "schema_version": schema,
        "created_at": "2026-09-08T00:00:00+00:00",
        "dataset": dataset,
        "adapter": {
            "name": "fake",
            "model": "fake-model",
            "device": "cpu",
            "compute_type": "float32",
            "decoding_options": {},
        },
        "channel": {
            "codec": "none",
            "packet_loss_rate": 0.0,
            "frame_duration_ms": 20,
            "seed": 0,
            "additive_noise_snr_db": None,
            "jitter_std_ms": None,
            "playout_buffer_ms": None,
        },
        "summary": {
            "total_audio_seconds": 1.0,
            "adapter_seconds": 0.1,
            "wer": 0.0,
            "cer": 0.0,
            "rtf": 0.1,
            "speed_factor": 10.0,
            "numeric_entity_matches": 0,
            "numeric_entity_reference_count": 0,
            "numeric_entity_accuracy": None,
        },
        "items": [],
    }


def _write_artifact(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_fingerprint_is_versioned_deterministic_and_path_independent(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_audio = first_dir / "a.wav"
    second_audio = second_dir / "renamed.wav"
    first_audio.write_bytes(b"exact source wav bytes")
    second_audio.write_bytes(first_audio.read_bytes())

    first = _fingerprint((_item(first_audio),))
    second = _fingerprint((_item(second_audio),))

    assert first == second
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", first)
    assert _fingerprint((_item(first_audio),)) == first


@pytest.mark.parametrize("change", ["id", "reference", "language", "audio"])
def test_fingerprint_changes_when_semantic_dataset_input_changes(
    tmp_path: Path,
    change: str,
) -> None:
    base_audio = tmp_path / "base.wav"
    changed_audio = tmp_path / "changed.wav"
    base_audio.write_bytes(b"audio-a")
    changed_audio.write_bytes(b"audio-b")
    base = _item(base_audio, item_id="a", reference="hello", language="en")

    kwargs: dict[str, object] = {
        "audio": base_audio,
        "item_id": "a",
        "reference": "hello",
        "language": "en",
    }
    if change == "id":
        kwargs["item_id"] = "b"
    elif change == "reference":
        kwargs["reference"] = "different"
    elif change == "language":
        kwargs["language"] = "ru"
    else:
        kwargs["audio"] = changed_audio

    changed = _item(**kwargs)
    assert _fingerprint((base,)) != _fingerprint((changed,))


def test_fingerprint_is_manifest_order_sensitive(tmp_path: Path) -> None:
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    first_audio.write_bytes(b"first")
    second_audio.write_bytes(b"second")
    first = _item(first_audio, item_id="first", line_number=1)
    second = _item(second_audio, item_id="second", line_number=2)

    assert _fingerprint((first, second)) != _fingerprint((second, first))


def test_runner_records_current_schema_dataset_fingerprint(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    _write_pcm16_wav(audio)
    manifest = tmp_path / "dataset.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample",
                "audio": audio.name,
                "reference": "hello",
                "language": "en",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeAdapter:
        name = "fake"
        model = "fake-model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        def transcribe(self, audio, language=None):
            return Transcription(text="hello")

    result = run_benchmark(manifest, FakeAdapter())

    assert result.schema_version == 6
    assert result.dataset.fingerprint == _fingerprint(load_dataset_manifest(manifest))
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", result.dataset.fingerprint)


def test_compare_accepts_matching_v5_fingerprints(tmp_path: Path) -> None:
    from callasr.report import load_comparison_rows

    fingerprint = "sha256:" + "a" * 64
    first = _write_artifact(tmp_path / "first.json", _artifact(5, fingerprint=fingerprint))
    second = _write_artifact(tmp_path / "second.json", _artifact(5, fingerprint=fingerprint))

    rows = load_comparison_rows([first, second])

    assert [row.dataset_fingerprint for row in rows] == [fingerprint, fingerprint]
    assert [row.schema_version for row in rows] == [5, 5]


def test_compare_rejects_conflicting_known_dataset_fingerprints(tmp_path: Path) -> None:
    from callasr.report import ComparisonError, compare_result_artifacts

    first = _write_artifact(
        tmp_path / "first.json",
        _artifact(5, fingerprint="sha256:" + "a" * 64),
    )
    second = _write_artifact(
        tmp_path / "second.json",
        _artifact(5, fingerprint="sha256:" + "b" * 64),
    )

    with pytest.raises(ComparisonError, match=r"dataset fingerprint.*first\.json.*second\.json"):
        compare_result_artifacts([first, second])


@pytest.mark.parametrize(
    "fingerprint",
    [
        None,
        "",
        "md5:" + "a" * 32,
        "sha256:" + "a" * 63,
        "sha256:" + "A" * 64,
        "sha256:" + "g" * 64,
    ],
)
def test_schema_v5_requires_strict_lowercase_sha256_fingerprint(
    tmp_path: Path,
    fingerprint: str | None,
) -> None:
    from callasr.report import ComparisonError, compare_result_artifacts

    path = _write_artifact(tmp_path / "invalid.json", _artifact(5, fingerprint=fingerprint))

    with pytest.raises(ComparisonError, match=r"invalid\.json.*fingerprint"):
        compare_result_artifacts([path])


def test_mixed_v4_v5_remains_comparable_without_claiming_old_fingerprint(tmp_path: Path) -> None:
    from callasr.report import load_comparison_rows

    old = _write_artifact(tmp_path / "old.json", _artifact(4))
    current = _write_artifact(
        tmp_path / "current.json",
        _artifact(5, fingerprint="sha256:" + "c" * 64),
    )

    rows = load_comparison_rows([old, current])

    assert rows[0].dataset_fingerprint is None
    assert rows[1].dataset_fingerprint == "sha256:" + "c" * 64
