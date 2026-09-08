from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from callasr.adapters.base import Transcription
from callasr.audio import AudioBuffer
from callasr.benchmark import BenchmarkResult, run_benchmark
from callasr.dataset import DatasetItem


class FakeAdapter:
    name = "fake"
    model = "fake-model"
    device = "cpu"
    compute_type = "float32"
    decoding_options: ClassVar[dict[str, int]] = {"beam_size": 1}

    def __init__(self) -> None:
        self.calls: list[AudioBuffer] = []

    def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcription:
        self.calls.append(audio)
        return Transcription("hello")


def _item(tmp_path: Path) -> DatasetItem:
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    return DatasetItem(
        id="id-0",
        audio=audio_path,
        reference="hello",
        language="en",
        line_number=1,
    )


def _patch_single_item(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: AudioBuffer,
) -> None:
    item = _item(tmp_path)
    monkeypatch.setattr("callasr.benchmark.load_dataset_manifest", lambda path: (item,))
    monkeypatch.setattr("callasr.benchmark.load_wav", lambda path: source)
    monkeypatch.setattr("callasr.benchmark.perf_counter", lambda: 1.0)


def _artifact(
    schema: int,
    *,
    gain_db: float | None = None,
    clip_threshold: float | None = None,
) -> dict[str, object]:
    dataset: dict[str, object] = {"path": "/tmp/dataset.jsonl", "item_count": 1}
    if schema >= 5:
        dataset["fingerprint"] = "sha256:" + "a" * 64
    channel: dict[str, object] = {
        "codec": "none",
        "packet_loss_rate": 0.0,
        "frame_duration_ms": 20,
        "seed": 0,
    }
    if schema >= 2:
        channel["additive_noise_snr_db"] = None
    if schema >= 3:
        channel["jitter_std_ms"] = None
        channel["playout_buffer_ms"] = None
    if schema >= 6:
        channel["gain_db"] = gain_db
        channel["clip_threshold"] = clip_threshold
    summary: dict[str, object] = {
        "total_audio_seconds": 1.0,
        "adapter_seconds": 0.1,
        "wer": 0.0,
        "cer": 0.0,
        "rtf": 0.1,
        "speed_factor": 10.0,
    }
    if schema >= 4:
        summary.update(
            numeric_entity_matches=0,
            numeric_entity_reference_count=0,
            numeric_entity_accuracy=None,
        )
    return {
        "schema_version": schema,
        "created_at": "2026-09-08T12:00:00+00:00",
        "dataset": dataset,
        "adapter": {
            "name": "fake",
            "model": "fake-model",
            "device": "cpu",
            "compute_type": "float32",
            "decoding_options": {},
        },
        "channel": channel,
        "summary": summary,
        "items": [],
    }


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_runner_applies_gain_clip_before_noise_and_telephone_channel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = AudioBuffer(np.full(8_000, 0.1, dtype=np.float32), 8_000)
    _patch_single_item(monkeypatch, tmp_path, source)
    events: list[tuple[object, ...]] = []

    def gain_clip(
        audio: AudioBuffer,
        *,
        gain_db: float,
        clip_threshold: float | None,
    ) -> AudioBuffer:
        assert audio is source
        events.append(("gain", gain_db, clip_threshold))
        return AudioBuffer(np.full(audio.samples.shape, 0.2, dtype=np.float32), audio.sample_rate)

    def noise(audio: AudioBuffer, *, snr_db: float, seed: int) -> AudioBuffer:
        assert np.all(audio.samples == np.float32(0.2))
        events.append(("noise", snr_db, seed))
        return AudioBuffer(np.full(audio.samples.shape, 0.3, dtype=np.float32), audio.sample_rate)

    def channel(
        audio: AudioBuffer,
        codec: str,
        *,
        packet_loss_rate: float,
        frame_duration_ms: int,
        seed: int,
    ) -> AudioBuffer:
        assert np.all(audio.samples == np.float32(0.3))
        events.append(("channel", codec, packet_loss_rate, frame_duration_ms, seed))
        return audio

    monkeypatch.setattr("callasr.benchmark.apply_gain_and_clip", gain_clip)
    monkeypatch.setattr("callasr.benchmark.apply_additive_noise", noise)
    monkeypatch.setattr("callasr.benchmark.telephone_channel", channel)

    result = run_benchmark(
        tmp_path / "dataset.jsonl",
        FakeAdapter(),
        codec="pcmu",
        packet_loss_rate=0.05,
        gain_db=6.0,
        clip_threshold=0.4,
        snr_db=15.0,
        seed=42,
    )

    noise_seed = int(np.random.SeedSequence([42, 0, 1]).generate_state(1, dtype=np.uint32)[0])
    packet_seed = int(np.random.SeedSequence([42, 0]).generate_state(1, dtype=np.uint32)[0])
    assert events == [
        ("gain", 6.0, 0.4),
        ("noise", 15.0, noise_seed),
        ("channel", "pcmu", 0.05, 20, packet_seed),
    ]
    assert result.schema_version == 6
    assert result.channel.gain_db == 6.0
    assert result.channel.clip_threshold == 0.4


def test_disabled_gain_and_clipping_preserve_existing_audio_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = AudioBuffer(np.full(800, 0.1, dtype=np.float32), 8_000)
    _patch_single_item(monkeypatch, tmp_path, source)

    def forbidden(*args: object, **kwargs: object) -> AudioBuffer:
        raise AssertionError("gain/clipping transform must not run when disabled")

    monkeypatch.setattr("callasr.benchmark.apply_gain_and_clip", forbidden)
    adapter = FakeAdapter()

    result = run_benchmark(tmp_path / "dataset.jsonl", adapter, codec="none")

    assert adapter.calls == [source]
    assert result.channel.gain_db == 0.0
    assert result.channel.clip_threshold is None


def test_clean_run_can_apply_gain_and_clipping_without_telephone_channel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = AudioBuffer(np.full(800, 0.1, dtype=np.float32), 8_000)
    _patch_single_item(monkeypatch, tmp_path, source)
    seen: list[tuple[float, float | None]] = []

    def transform(audio: AudioBuffer, *, gain_db: float, clip_threshold: float | None) -> AudioBuffer:
        seen.append((gain_db, clip_threshold))
        return audio

    def forbidden_channel(*args: object, **kwargs: object) -> AudioBuffer:
        raise AssertionError("telephone channel must not run for codec=none")

    monkeypatch.setattr("callasr.benchmark.apply_gain_and_clip", transform)
    monkeypatch.setattr("callasr.benchmark.telephone_channel", forbidden_channel)

    result = run_benchmark(
        tmp_path / "dataset.jsonl",
        FakeAdapter(),
        codec="none",
        gain_db=-3.0,
        clip_threshold=0.8,
    )

    assert seen == [(-3.0, 0.8)]
    assert result.channel.gain_db == -3.0
    assert result.channel.clip_threshold == 0.8


def test_cli_passes_gain_and_clipping_to_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from callasr import cli

    seen: dict[str, object] = {}
    monkeypatch.setattr(cli, "FasterWhisperAdapter", lambda *args, **kwargs: object())

    def run(*args: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "run_benchmark", run)
    monkeypatch.setattr(cli, "write_result_artifact", lambda result, path: None)

    status = cli.main(
        [
            "run",
            "dataset.jsonl",
            "--adapter",
            "faster-whisper",
            "--model",
            "large-v3",
            "--gain-db",
            "-3",
            "--clip-threshold",
            "0.8",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert status == 0
    assert seen["gain_db"] == -3.0
    assert seen["clip_threshold"] == 0.8


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_cli_rejects_non_finite_gain_before_adapter_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str,
) -> None:
    from callasr import cli

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("adapter must not be constructed")

    monkeypatch.setattr(cli, "FasterWhisperAdapter", forbidden)
    with pytest.raises(SystemExit, match="2"):
        cli.main(
            [
                "run",
                "dataset.jsonl",
                "--adapter",
                "faster-whisper",
                "--model",
                "large-v3",
                f"--gain-db={value}",
                "--output",
                str(tmp_path / "result.json"),
            ]
        )


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_cli_rejects_invalid_clip_threshold_before_adapter_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str,
) -> None:
    from callasr import cli

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("adapter must not be constructed")

    monkeypatch.setattr(cli, "FasterWhisperAdapter", forbidden)
    with pytest.raises(SystemExit, match="2"):
        cli.main(
            [
                "run",
                "dataset.jsonl",
                "--adapter",
                "faster-whisper",
                "--model",
                "large-v3",
                f"--clip-threshold={value}",
                "--output",
                str(tmp_path / "result.json"),
            ]
        )


def test_compare_exposes_gain_clipping_and_supports_v5_v6(tmp_path: Path) -> None:
    from callasr.report import compare_result_artifacts

    old = _write(tmp_path / "v5.json", _artifact(5))
    current = _write(tmp_path / "v6.json", _artifact(6, gain_db=0.0, clip_threshold=0.8))

    lines = compare_result_artifacts([old, current]).splitlines()

    assert lines[0] == (
        "| Artifact | Schema | Adapter | Model | Codec | Loss | Gain dB | Clip | SNR dB | "
        "Jitter ms | WER | CER | RTF | Speed | Numeric entity | Items |"
    )
    assert "| v5.json | 5 | fake | fake-model | none | 0 | — | — | — | — |" in lines[2]
    assert "| v6.json | 6 | fake | fake-model | none | 0 | 0 | 0.8 | — | — |" in lines[3]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["channel"].pop("gain_db"), "gain_db"),
        (lambda payload: payload["channel"].pop("clip_threshold"), "clip_threshold"),
        (lambda payload: payload["channel"].__setitem__("gain_db", float("nan")), "gain_db"),
        (lambda payload: payload["channel"].__setitem__("clip_threshold", 0.0), "clip_threshold"),
    ],
)
def test_compare_strictly_validates_schema_v6_front_end_fields(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    from callasr.report import ComparisonError, compare_result_artifacts

    payload = _artifact(6, gain_db=0.0, clip_threshold=None)
    mutation(payload)
    path = _write(tmp_path / "invalid.json", payload)

    with pytest.raises(ComparisonError, match=rf"invalid\.json.*{message}"):
        compare_result_artifacts([path])


def test_schema_seven_is_unknown(tmp_path: Path) -> None:
    from callasr.report import ComparisonError, compare_result_artifacts

    path = _write(tmp_path / "future.json", _artifact(7, gain_db=0.0, clip_threshold=None))

    with pytest.raises(ComparisonError, match=r"future\.json.*schema_version"):
        compare_result_artifacts([path])
