from __future__ import annotations

from collections.abc import Iterable
from dataclasses import FrozenInstanceError
from typing import ClassVar

import numpy as np
import pytest

from callasr.audio import AudioBuffer


def _api():
    try:
        from callasr.streaming import (
            StreamingASRAdapter,
            StreamingBenchmarkResult,
            StreamingError,
            StreamingUpdate,
            frame_audio,
            partial_stability_score,
            run_streaming_benchmark,
        )
    except ImportError as exc:
        pytest.fail(f"streaming foundation implementation is missing: {exc}")
    return (
        StreamingASRAdapter,
        StreamingBenchmarkResult,
        StreamingError,
        StreamingUpdate,
        frame_audio,
        partial_stability_score,
        run_streaming_benchmark,
    )


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _audio(sample_count: int, sample_rate: int = 1_000) -> AudioBuffer:
    samples = np.arange(sample_count, dtype=np.float32) / max(sample_count, 1)
    return AudioBuffer(samples=samples, sample_rate=sample_rate)


def test_streaming_update_is_immutable_and_protocol_is_runtime_checkable() -> None:
    StreamingASRAdapter, _, _, StreamingUpdate, _, _, _ = _api()

    update = StreamingUpdate(text="hello", is_final=False)
    with pytest.raises(FrozenInstanceError):
        update.text = "changed"

    class Adapter:
        name = "fake-stream"
        model = "fake-model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        def stream(
            self,
            frames: Iterable[AudioBuffer],
            language: str | None = None,
        ) -> Iterable[object]:
            for _ in frames:
                pass
            yield StreamingUpdate(text="done", is_final=True)

    assert isinstance(Adapter(), StreamingASRAdapter)


def test_frame_audio_preserves_samples_and_exact_frame_boundaries() -> None:
    _, _, _, _, frame_audio, _, _ = _api()
    source = _audio(10, sample_rate=1_000)

    frames = frame_audio(source, frame_duration_ms=4)

    assert isinstance(frames, tuple)
    assert [frame.samples.size for frame in frames] == [4, 4, 2]
    assert all(frame.sample_rate == 1_000 for frame in frames)
    np.testing.assert_array_equal(
        np.concatenate([frame.samples for frame in frames]),
        source.samples,
    )
    assert frames[0] is not source


@pytest.mark.parametrize("frame_duration_ms", [0, -1, True, 1.5])
def test_frame_audio_rejects_invalid_duration(frame_duration_ms: object) -> None:
    _, _, StreamingError, _, frame_audio, _, _ = _api()

    with pytest.raises(StreamingError, match="frame_duration_ms"):
        frame_audio(_audio(10), frame_duration_ms=frame_duration_ms)


def test_frame_audio_rejects_fractional_sample_frame_size() -> None:
    _, _, StreamingError, _, frame_audio, _, _ = _api()

    with pytest.raises(StreamingError, match="integer number of samples"):
        frame_audio(_audio(1_000, sample_rate=11_025), frame_duration_ms=20)


def test_streaming_runner_has_exact_fake_clock_timing_boundaries() -> None:
    _, StreamingBenchmarkResult, _, StreamingUpdate, _, _, run_streaming_benchmark = _api()
    clock = FakeClock(100.0)

    class Adapter:
        name = "fake-stream"
        model = "fake-model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        def stream(
            self,
            frames: Iterable[AudioBuffer],
            language: str | None = None,
        ) -> Iterable[object]:
            iterator = iter(frames)
            next(iterator)
            clock.advance(0.05)
            yield StreamingUpdate(text="", is_final=False)

            next(iterator)
            clock.advance(0.10)
            yield StreamingUpdate(text="hello", is_final=False)

            next(iterator)
            clock.advance(0.20)
            yield StreamingUpdate(text="hello world", is_final=False)

            with pytest.raises(StopIteration):
                next(iterator)
            clock.advance(0.30)
            yield StreamingUpdate(text="hello world", is_final=True)
            clock.advance(0.05)

    result = run_streaming_benchmark(
        _audio(5),
        Adapter(),
        language="en",
        frame_duration_ms=2,
        clock=clock,
    )

    assert isinstance(result, StreamingBenchmarkResult)
    assert result.frame_duration_ms == 2
    assert result.frame_count == 3
    assert result.audio_seconds == pytest.approx(0.005)
    assert result.final_text == "hello world"
    assert result.partial_update_count == 3
    assert result.time_to_first_partial_seconds == pytest.approx(0.15)
    assert result.finalization_latency_seconds == pytest.approx(0.50)
    assert result.total_streaming_wall_seconds == pytest.approx(0.70)
    assert result.partial_stability == pytest.approx(0.75)
    assert [item.observed_seconds for item in result.updates] == pytest.approx(
        [0.05, 0.15, 0.35, 0.65]
    )
    assert [item.is_final for item in result.updates] == [False, False, False, True]
    assert not hasattr(result, "schema_version")


def test_no_non_empty_partial_has_null_first_partial_and_stability() -> None:
    _, _, _, StreamingUpdate, _, _, run_streaming_benchmark = _api()
    clock = FakeClock(5.0)

    class Adapter:
        name = "fake-stream"
        model = "fake-model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        def stream(self, frames: Iterable[AudioBuffer], language: str | None = None):
            for _ in frames:
                clock.advance(0.01)
            yield StreamingUpdate(text="final", is_final=True)

    result = run_streaming_benchmark(_audio(4), Adapter(), frame_duration_ms=2, clock=clock)

    assert result.time_to_first_partial_seconds is None
    assert result.partial_stability is None
    assert result.partial_update_count == 0
    assert result.final_text == "final"


def test_partial_stability_uses_normalized_token_revision_similarity() -> None:
    _, _, _, _, _, partial_stability_score, _ = _api()

    assert partial_stability_score(["Hello world"], "hello, WORLD!") == 1.0
    assert partial_stability_score(["one two", "one three"], "one three four") == pytest.approx(
        (0.5 + 2.0 / 3.0) / 2.0
    )


def test_partial_stability_collapses_duplicate_normalized_partials() -> None:
    _, _, _, _, _, partial_stability_score, _ = _api()

    score = partial_stability_score(["a", "A!", "a b"], "a b")

    assert score == pytest.approx(0.75)


def test_partial_stability_is_null_without_non_empty_partial() -> None:
    _, _, _, _, _, partial_stability_score, _ = _api()

    assert partial_stability_score([], "final") is None
    assert partial_stability_score(["", "!!!"], "final") is None


def test_streaming_runner_rejects_adapter_that_does_not_consume_all_frames() -> None:
    _, _, StreamingError, StreamingUpdate, _, _, run_streaming_benchmark = _api()

    class Adapter:
        name = "fake-stream"
        model = "fake-model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        def stream(self, frames: Iterable[AudioBuffer], language: str | None = None):
            next(iter(frames))
            yield StreamingUpdate(text="done", is_final=True)

    with pytest.raises(StreamingError, match="consume every frame"):
        run_streaming_benchmark(_audio(6), Adapter(), frame_duration_ms=2, clock=FakeClock())


def test_streaming_runner_requires_final_update() -> None:
    _, _, StreamingError, StreamingUpdate, _, _, run_streaming_benchmark = _api()

    class Adapter:
        name = "fake-stream"
        model = "fake-model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        def stream(self, frames: Iterable[AudioBuffer], language: str | None = None):
            for _ in frames:
                pass
            yield StreamingUpdate(text="partial", is_final=False)

    with pytest.raises(StreamingError, match="final update"):
        run_streaming_benchmark(_audio(4), Adapter(), frame_duration_ms=2, clock=FakeClock())


def test_streaming_runner_rejects_second_final_update() -> None:
    _, _, StreamingError, StreamingUpdate, _, _, run_streaming_benchmark = _api()

    class Adapter:
        name = "fake-stream"
        model = "fake-model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        def stream(self, frames: Iterable[AudioBuffer], language: str | None = None):
            for _ in frames:
                pass
            yield StreamingUpdate(text="final", is_final=True)
            yield StreamingUpdate(text="final again", is_final=True)

    with pytest.raises(StreamingError, match="after final"):
        run_streaming_benchmark(_audio(4), Adapter(), frame_duration_ms=2, clock=FakeClock())


def test_streaming_runner_rejects_partial_after_final() -> None:
    _, _, StreamingError, StreamingUpdate, _, _, run_streaming_benchmark = _api()

    class Adapter:
        name = "fake-stream"
        model = "fake-model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        def stream(self, frames: Iterable[AudioBuffer], language: str | None = None):
            for _ in frames:
                pass
            yield StreamingUpdate(text="final", is_final=True)
            yield StreamingUpdate(text="late partial", is_final=False)

    with pytest.raises(StreamingError, match="after final"):
        run_streaming_benchmark(_audio(4), Adapter(), frame_duration_ms=2, clock=FakeClock())


def test_streaming_runner_rejects_invalid_update_text() -> None:
    _, _, StreamingError, StreamingUpdate, _, _, run_streaming_benchmark = _api()

    class Adapter:
        name = "fake-stream"
        model = "fake-model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        def stream(self, frames: Iterable[AudioBuffer], language: str | None = None):
            for _ in frames:
                pass
            yield StreamingUpdate(text=123, is_final=True)

    with pytest.raises(StreamingError, match="text must be a string"):
        run_streaming_benchmark(_audio(4), Adapter(), frame_duration_ms=2, clock=FakeClock())


def test_streaming_runner_rejects_non_monotonic_clock() -> None:
    _, _, StreamingError, StreamingUpdate, _, _, run_streaming_benchmark = _api()
    clock = FakeClock(1.0)

    class Adapter:
        name = "fake-stream"
        model = "fake-model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        def stream(self, frames: Iterable[AudioBuffer], language: str | None = None):
            for _ in frames:
                pass
            clock.now = 0.5
            yield StreamingUpdate(text="final", is_final=True)

    with pytest.raises(StreamingError, match="monotonic"):
        run_streaming_benchmark(_audio(4), Adapter(), frame_duration_ms=2, clock=clock)


def test_streaming_api_is_exported_from_top_level_package() -> None:
    try:
        from callasr import (
            StreamingASRAdapter,
            StreamingBenchmarkResult,
            StreamingError,
            StreamingUpdate,
            frame_audio,
            partial_stability_score,
            run_streaming_benchmark,
        )
    except ImportError as exc:
        pytest.fail(f"public streaming API is missing: {exc}")

    api = _api()
    assert StreamingASRAdapter is api[0]
    assert StreamingBenchmarkResult is api[1]
    assert StreamingError is api[2]
    assert StreamingUpdate is api[3]
    assert frame_audio is api[4]
    assert partial_stability_score is api[5]
    assert run_streaming_benchmark is api[6]
