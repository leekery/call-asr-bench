from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from callasr.adapters.base import Transcription
from callasr.audio import AudioBuffer
from callasr.benchmark import run_benchmark
from callasr.dataset import DatasetItem


class FakeAdapter:
    name = "fake"
    model = "fake-model"
    device = "cpu"
    compute_type = "float32"
    decoding_options = {"beam_size": 1}

    def __init__(self, outputs: list[str | Exception]) -> None:
        self.outputs = iter(outputs)
        self.calls: list[tuple[AudioBuffer, str | None]] = []

    def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcription:
        self.calls.append((audio, language))
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return Transcription(value)


def _item(
    tmp_path: Path,
    index: int,
    *,
    reference: str = "hello",
    language: str | None = "en",
) -> DatasetItem:
    audio_path = tmp_path / f"audio-{index}.wav"
    audio_path.touch()
    return DatasetItem(
        id=f"id-{index}",
        audio=audio_path,
        reference=reference,
        language=language,
        line_number=index + 1,
    )


def test_clean_run_preserves_order_and_does_not_apply_channel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    items = (_item(tmp_path, 0), _item(tmp_path, 1))
    monkeypatch.setattr("callasr.benchmark.load_dataset_manifest", lambda path: items)
    monkeypatch.setattr(
        "callasr.benchmark.load_wav",
        lambda path: AudioBuffer(np.zeros(16_000, dtype=np.float32), 16_000),
    )

    def forbidden_channel(*args: object, **kwargs: object) -> None:
        raise AssertionError("telephone channel called")

    monkeypatch.setattr("callasr.benchmark.telephone_channel", forbidden_channel)
    times = iter([1.0, 1.25, 2.0, 2.5])
    monkeypatch.setattr("callasr.benchmark.perf_counter", lambda: next(times))

    result = run_benchmark(
        tmp_path / "dataset.jsonl",
        FakeAdapter(["hello", "hello"]),
        codec="none",
    )

    assert [item.id for item in result.items] == ["id-0", "id-1"]
    assert result.channel.codec == "none"
    assert result.summary.total_audio_seconds == 2.0
    assert result.summary.adapter_seconds == 0.75


def test_telephone_run_uses_deterministic_per_item_seeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    items = (_item(tmp_path, 0), _item(tmp_path, 1))
    monkeypatch.setattr("callasr.benchmark.load_dataset_manifest", lambda path: items)
    monkeypatch.setattr(
        "callasr.benchmark.load_wav",
        lambda path: AudioBuffer(np.zeros(8_000, dtype=np.float32), 8_000),
    )
    seen: list[tuple[str, float, int, int]] = []

    def channel(
        audio: AudioBuffer,
        codec: str,
        *,
        packet_loss_rate: float,
        frame_duration_ms: int,
        seed: int,
    ) -> AudioBuffer:
        seen.append((codec, packet_loss_rate, frame_duration_ms, seed))
        return audio

    monkeypatch.setattr("callasr.benchmark.telephone_channel", channel)
    times = iter([0.0, 1.0, 1.0, 2.0])
    monkeypatch.setattr("callasr.benchmark.perf_counter", lambda: next(times))

    run_benchmark(
        tmp_path / "dataset.jsonl",
        FakeAdapter(["hello", "hello"]),
        codec="pcmu",
        packet_loss_rate=0.05,
        frame_duration_ms=20,
        seed=42,
    )

    expected = [
        int(np.random.SeedSequence([42, index]).generate_state(1, dtype=np.uint32)[0])
        for index in range(2)
    ]
    assert [call[3] for call in seen] == expected
    assert all(call[:3] == ("pcmu", 0.05, 20) for call in seen)


def test_timing_rtf_and_speed_factor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    items = (_item(tmp_path, 0),)
    monkeypatch.setattr("callasr.benchmark.load_dataset_manifest", lambda path: items)
    monkeypatch.setattr(
        "callasr.benchmark.load_wav",
        lambda path: AudioBuffer(np.zeros(32_000, dtype=np.float32), 16_000),
    )
    times = iter([10.0, 10.5])
    monkeypatch.setattr("callasr.benchmark.perf_counter", lambda: next(times))

    result = run_benchmark(
        tmp_path / "dataset.jsonl",
        FakeAdapter(["hello"]),
        codec="none",
    )

    assert result.items[0].adapter_seconds == 0.5
    assert result.summary.rtf == 0.25
    assert result.summary.speed_factor == 4.0


def test_zero_adapter_time_has_null_speed_factor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    items = (_item(tmp_path, 0),)
    monkeypatch.setattr("callasr.benchmark.load_dataset_manifest", lambda path: items)
    monkeypatch.setattr(
        "callasr.benchmark.load_wav",
        lambda path: AudioBuffer(np.zeros(16_000, dtype=np.float32), 16_000),
    )
    monkeypatch.setattr("callasr.benchmark.perf_counter", lambda: 5.0)

    result = run_benchmark(
        tmp_path / "dataset.jsonl",
        FakeAdapter(["hello"]),
        codec="none",
    )

    assert result.summary.adapter_seconds == 0.0
    assert result.summary.rtf == 0.0
    assert result.summary.speed_factor is None


def test_metrics_use_micro_aggregation_and_empty_reference_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    items = (
        _item(tmp_path, 0, reference="a b"),
        _item(tmp_path, 1, reference=""),
    )
    monkeypatch.setattr("callasr.benchmark.load_dataset_manifest", lambda path: items)
    monkeypatch.setattr(
        "callasr.benchmark.load_wav",
        lambda path: AudioBuffer(np.zeros(8_000, dtype=np.float32), 8_000),
    )
    times = iter([0.0, 1.0, 1.0, 2.0])
    monkeypatch.setattr("callasr.benchmark.perf_counter", lambda: next(times))

    result = run_benchmark(
        tmp_path / "dataset.jsonl",
        FakeAdapter(["a x", "oops"]),
        codec="none",
    )

    assert result.items[0].wer == 0.5
    assert result.items[1].wer == 1.0
    assert result.summary.wer == pytest.approx(2 / 3)


def test_transcription_failure_stops_before_later_items(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    items = (_item(tmp_path, 0), _item(tmp_path, 1), _item(tmp_path, 2))
    monkeypatch.setattr("callasr.benchmark.load_dataset_manifest", lambda path: items)
    loaded: list[Path] = []

    def load(path: Path) -> AudioBuffer:
        loaded.append(path)
        return AudioBuffer(np.zeros(10, dtype=np.float32), 10)

    monkeypatch.setattr("callasr.benchmark.load_wav", load)
    times = iter([0.0, 1.0])
    monkeypatch.setattr("callasr.benchmark.perf_counter", lambda: next(times))
    adapter = FakeAdapter([RuntimeError("boom"), "hello", "hello"])

    with pytest.raises(RuntimeError, match="boom"):
        run_benchmark(tmp_path / "dataset.jsonl", adapter, codec="none")

    assert len(loaded) == 1
    assert len(adapter.calls) == 1


def test_channel_failure_does_not_call_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    items = (_item(tmp_path, 0), _item(tmp_path, 1))
    monkeypatch.setattr("callasr.benchmark.load_dataset_manifest", lambda path: items)
    monkeypatch.setattr(
        "callasr.benchmark.load_wav",
        lambda path: AudioBuffer(np.zeros(8_000, dtype=np.float32), 8_000),
    )

    def broken_channel(*args: object, **kwargs: object) -> AudioBuffer:
        raise ValueError("bad channel")

    monkeypatch.setattr("callasr.benchmark.telephone_channel", broken_channel)
    adapter = FakeAdapter(["hello", "hello"])

    with pytest.raises(ValueError, match="bad channel"):
        run_benchmark(tmp_path / "dataset.jsonl", adapter, codec="pcma")

    assert adapter.calls == []
