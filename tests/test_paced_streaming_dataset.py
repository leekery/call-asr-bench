from __future__ import annotations

import asyncio
import importlib
import json
import wave
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from callasr.audio import AudioBuffer
from callasr.dataset import dataset_fingerprint, load_dataset_manifest
from callasr.paced_streaming import StreamingError
from callasr.streaming import StreamingUpdate


def _api():
    try:
        return importlib.import_module("callasr.streaming_dataset")
    except ImportError as exc:
        pytest.fail(f"paced streaming dataset implementation is missing: {exc}")


def _write_wav(path: Path, *, marker: int = 0) -> None:
    samples = np.full(320, marker, dtype="<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(samples.tobytes())


def _write_manifest(tmp_path: Path) -> Path:
    rows = [
        {"id": "item-0", "audio": "item-0.wav", "reference": "a", "language": "ru"},
        {"id": "item-1", "audio": "item-1.wav", "reference": "b", "language": "en"},
        {"id": "item-2", "audio": "item-2.wav", "reference": "x y"},
    ]
    for index in range(len(rows)):
        _write_wav(tmp_path / f"item-{index}.wav", marker=index)
    manifest = tmp_path / "dataset.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return manifest


class FakeClock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSleeper:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.advance(seconds)
        await asyncio.sleep(0)


class ScriptedPacedAdapter:
    name = "fake-paced"
    model = "fake-model"
    device = "cpu"
    compute_type = "float32"
    decoding_options: ClassVar[dict[str, object]] = {"mode": "fake"}

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls = 0
        self.languages: list[str | None] = []

    async def stream_duplex(self, frames, language: str | None = None):
        call = self.calls
        self.calls += 1
        self.languages.append(language)
        iterator = aiter(frames)
        if call == 2:
            self.clock.advance(0.03)
        await anext(iterator)

        if call == 0:
            self.clock.advance(0.1)
            yield StreamingUpdate(text="hello", is_final=False)
        elif call == 2:
            self.clock.advance(0.04)
            yield StreamingUpdate(text="a b", is_final=False)

        async for _ in iterator:
            pass
        self.clock.advance((0.2, 0.1, 0.06)[call])
        final_text = ("hello there", "done", "a b")[call]
        yield StreamingUpdate(text=final_text, is_final=True)


def test_paced_dataset_runner_preserves_order_fingerprint_and_exact_aggregates(
    tmp_path: Path,
) -> None:
    dataset_module = _api()
    manifest = _write_manifest(tmp_path)
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    adapter = ScriptedPacedAdapter(clock)

    result = asyncio.run(
        dataset_module.run_paced_streaming_dataset_benchmark(
            manifest,
            adapter,
            frame_duration_ms=20,
            realtime_factor=1.0,
            language_mode="manifest",
            clock=clock,
            sleeper=sleeper,
        )
    )

    assert result.timing_mode == "paced"
    assert result.dataset_path == str(manifest.resolve())
    assert result.dataset_fingerprint == dataset_fingerprint(load_dataset_manifest(manifest))
    assert result.frame_duration_ms == 20
    assert result.realtime_factor == 1.0
    assert result.language_mode == "manifest"
    assert [item.id for item in result.items] == ["item-0", "item-1", "item-2"]
    assert [item.audio for item in result.items] == ["item-0.wav", "item-1.wav", "item-2.wav"]
    assert [item.audio_seconds for item in result.items] == pytest.approx([0.02, 0.02, 0.02])
    assert [item.frame_count for item in result.items] == [1, 1, 1]
    assert [item.final_text for item in result.items] == ["hello there", "done", "a b"]
    assert [item.partial_update_count for item in result.items] == [1, 0, 1]
    assert [item.session_setup_seconds for item in result.items] == pytest.approx([0.0, 0.0, 0.03])
    assert [item.time_to_first_partial_seconds for item in result.items] == [
        pytest.approx(0.1),
        None,
        pytest.approx(0.04),
    ]
    assert [item.audio_submitted_seconds_at_first_partial for item in result.items] == [
        pytest.approx(0.02),
        None,
        pytest.approx(0.02),
    ]
    assert [item.audio_submission_wall_seconds for item in result.items] == pytest.approx(
        [0.12, 0.02, 0.06]
    )
    assert [item.finalization_latency_seconds for item in result.items] == pytest.approx(
        [0.2, 0.1, 0.06]
    )
    assert [item.total_session_wall_seconds for item in result.items] == pytest.approx(
        [0.32, 0.12, 0.15]
    )
    assert [item.partial_stability for item in result.items] == pytest.approx([0.5, None, 1.0])
    assert adapter.languages == ["ru", "en", None]
    assert sleeper.calls == pytest.approx([0.02, 0.02, 0.02])

    summary = result.summary
    assert summary.item_count == 3
    assert summary.completed_items == 3
    assert summary.failed_items == 0
    assert summary.total_audio_seconds == pytest.approx(0.06)
    assert summary.time_to_first_partial_count == 2
    assert summary.time_to_first_partial_p50_seconds == pytest.approx(0.04)
    assert summary.time_to_first_partial_p95_seconds == pytest.approx(0.1)
    assert summary.audio_submitted_at_first_partial_count == 2
    assert summary.audio_submitted_at_first_partial_p50_seconds == pytest.approx(0.02)
    assert summary.audio_submitted_at_first_partial_p95_seconds == pytest.approx(0.02)
    assert summary.session_setup_p50_seconds == pytest.approx(0.0)
    assert summary.session_setup_p95_seconds == pytest.approx(0.03)
    assert summary.session_setup_max_seconds == pytest.approx(0.03)
    assert summary.audio_submission_wall_p50_seconds == pytest.approx(0.06)
    assert summary.audio_submission_wall_p95_seconds == pytest.approx(0.12)
    assert summary.audio_submission_wall_max_seconds == pytest.approx(0.12)
    assert summary.finalization_latency_p50_seconds == pytest.approx(0.1)
    assert summary.finalization_latency_p95_seconds == pytest.approx(0.2)
    assert summary.finalization_latency_max_seconds == pytest.approx(0.2)
    assert summary.total_session_wall_p50_seconds == pytest.approx(0.15)
    assert summary.total_session_wall_p95_seconds == pytest.approx(0.32)
    assert summary.total_session_wall_max_seconds == pytest.approx(0.32)
    assert summary.partial_stability_count == 2
    assert summary.partial_stability_mean == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("language_mode", "expected_languages"),
    [
        ("manifest", ["ru", "en", None]),
        ("autodetect", [None, None, None]),
    ],
)
def test_paced_runner_propagates_frame_duration_factor_and_language_mode(
    tmp_path: Path,
    language_mode: str,
    expected_languages: list[str | None],
) -> None:
    dataset_module = _api()
    manifest = _write_manifest(tmp_path)
    clock = FakeClock()
    sleeper = FakeSleeper(clock)

    class DrainingAdapter:
        name = "draining"
        model = "fake"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        def __init__(self) -> None:
            self.languages: list[str | None] = []
            self.frame_sizes: list[list[int]] = []

        async def stream_duplex(self, frames, language: str | None = None):
            self.languages.append(language)
            frame_sizes = []
            async for frame in frames:
                assert isinstance(frame, AudioBuffer)
                frame_sizes.append(frame.samples.size)
            self.frame_sizes.append(frame_sizes)
            yield StreamingUpdate(text="ok", is_final=True)

    adapter = DrainingAdapter()
    result = asyncio.run(
        dataset_module.run_paced_streaming_dataset_benchmark(
            manifest,
            adapter,
            frame_duration_ms=10,
            realtime_factor=2.0,
            language_mode=language_mode,
            clock=clock,
            sleeper=sleeper,
        )
    )

    assert result.frame_duration_ms == 10
    assert result.realtime_factor == 2.0
    assert adapter.languages == expected_languages
    assert adapter.frame_sizes == [[160, 160], [160, 160], [160, 160]]
    assert sleeper.calls == pytest.approx([0.005] * 6)
    assert [item.frame_count for item in result.items] == [2, 2, 2]


def test_paced_no_partial_aggregates_are_null(tmp_path: Path) -> None:
    dataset_module = _api()
    _write_wav(tmp_path / "one.wav")
    manifest = tmp_path / "dataset.jsonl"
    manifest.write_text('{"id":"one","audio":"one.wav","reference":"done"}\n')
    clock = FakeClock()
    sleeper = FakeSleeper(clock)

    class FinalOnlyAdapter:
        name = "final-only"
        model = "fake"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        async def stream_duplex(self, frames, language: str | None = None):
            async for _ in frames:
                pass
            yield StreamingUpdate(text="done", is_final=True)

    result = asyncio.run(
        dataset_module.run_paced_streaming_dataset_benchmark(
            manifest,
            FinalOnlyAdapter(),
            clock=clock,
            sleeper=sleeper,
        )
    )

    assert result.summary.time_to_first_partial_count == 0
    assert result.summary.time_to_first_partial_p50_seconds is None
    assert result.summary.time_to_first_partial_p95_seconds is None
    assert result.summary.audio_submitted_at_first_partial_count == 0
    assert result.summary.audio_submitted_at_first_partial_p50_seconds is None
    assert result.summary.audio_submitted_at_first_partial_p95_seconds is None
    assert result.summary.partial_stability_count == 0
    assert result.summary.partial_stability_mean is None
    assert result.items[0].time_to_first_partial_seconds is None
    assert result.items[0].audio_submitted_seconds_at_first_partial is None
    assert result.items[0].partial_stability is None


@pytest.mark.parametrize("realtime_factor", [0.0, -1.0, float("inf"), float("nan"), True])
def test_invalid_realtime_factor_fails_before_adapter_use(
    tmp_path: Path,
    realtime_factor: object,
) -> None:
    dataset_module = _api()
    manifest = _write_manifest(tmp_path)
    adapter = ScriptedPacedAdapter(FakeClock())

    with pytest.raises(StreamingError, match="realtime_factor"):
        asyncio.run(
            dataset_module.run_paced_streaming_dataset_benchmark(
                manifest,
                adapter,
                realtime_factor=realtime_factor,
            )
        )
    assert adapter.calls == 0


def test_paced_dataset_failure_propagates_without_partial_result(tmp_path: Path) -> None:
    dataset_module = _api()
    manifest = _write_manifest(tmp_path)
    marker = RuntimeError("duplex stream failed")
    calls = 0

    class FailingAdapter:
        name = "fail"
        model = "fake"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        async def stream_duplex(self, frames, language: str | None = None):
            nonlocal calls
            async for _ in frames:
                pass
            calls += 1
            if calls == 2:
                raise marker
            yield StreamingUpdate(text="ok", is_final=True)

    with pytest.raises(RuntimeError) as captured:
        asyncio.run(
            dataset_module.run_paced_streaming_dataset_benchmark(manifest, FailingAdapter())
        )

    assert captured.value is marker
    assert calls == 2


def test_paced_result_models_are_immutable_and_public(tmp_path: Path) -> None:
    dataset_module = _api()
    manifest = _write_manifest(tmp_path)
    clock = FakeClock()
    result = asyncio.run(
        dataset_module.run_paced_streaming_dataset_benchmark(
            manifest,
            ScriptedPacedAdapter(clock),
            clock=clock,
            sleeper=FakeSleeper(clock),
        )
    )

    with pytest.raises(FrozenInstanceError):
        result.language_mode = "autodetect"

    import callasr

    assert callasr.PacedStreamingDatasetItemResult is dataset_module.PacedStreamingDatasetItemResult
    assert callasr.PacedStreamingDatasetResult is dataset_module.PacedStreamingDatasetResult
    assert callasr.PacedStreamingDatasetSummary is dataset_module.PacedStreamingDatasetSummary
    assert (
        callasr.run_paced_streaming_dataset_benchmark
        is dataset_module.run_paced_streaming_dataset_benchmark
    )
