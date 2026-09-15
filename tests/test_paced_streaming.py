from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from dataclasses import FrozenInstanceError
from typing import ClassVar

import numpy as np
import pytest

from callasr.audio import AudioBuffer
from callasr.paced_streaming import (
    PacedStreamingASRAdapter,
    PacedStreamingBenchmarkResult,
    run_paced_streaming_benchmark,
    validate_realtime_factor,
)
from callasr.streaming import StreamingASRAdapter, StreamingError, StreamingUpdate


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

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


def _audio(sample_count: int, sample_rate: int = 1_000) -> AudioBuffer:
    return AudioBuffer(np.zeros(sample_count, dtype=np.float32), sample_rate=sample_rate)


def test_paced_protocol_is_separate_from_file_upload_protocol() -> None:
    class SyncAdapter:
        name = "sync"
        model = "model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        def stream(
            self,
            frames: Iterable[AudioBuffer],
            language: str | None = None,
        ) -> Iterable[StreamingUpdate]:
            for _ in frames:
                pass
            yield StreamingUpdate("done", True)

    class PacedAdapter:
        name = "paced"
        model = "model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        async def stream_duplex(
            self,
            frames: AsyncIterable[AudioBuffer],
            language: str | None = None,
        ) -> AsyncIterator[StreamingUpdate]:
            async for _ in frames:
                pass
            yield StreamingUpdate("done", True)

    assert isinstance(SyncAdapter(), StreamingASRAdapter)
    assert not isinstance(SyncAdapter(), PacedStreamingASRAdapter)
    assert isinstance(PacedAdapter(), PacedStreamingASRAdapter)
    assert not isinstance(PacedAdapter(), StreamingASRAdapter)


@pytest.mark.parametrize(
    "realtime_factor",
    [0.0, -1.0, float("inf"), float("nan"), True, "1.0"],
)
def test_invalid_realtime_factor_is_rejected(realtime_factor: object) -> None:
    with pytest.raises(StreamingError, match="realtime_factor"):
        validate_realtime_factor(realtime_factor)


@pytest.mark.parametrize(
    ("realtime_factor", "expected_sleeps", "expected_wall"),
    [
        (1.0, [0.004, 0.004, 0.002], 0.010),
        (2.0, [0.002, 0.002, 0.001], 0.005),
    ],
)
def test_exact_pacing_includes_short_final_frame(
    realtime_factor: float,
    expected_sleeps: list[float],
    expected_wall: float,
) -> None:
    clock = FakeClock(100.0)
    sleeper = FakeSleeper(clock)

    class Adapter:
        name = "fake-paced"
        model = "model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        async def stream_duplex(self, frames, language=None):
            async for _ in frames:
                pass
            yield StreamingUpdate("done", True)

    result = asyncio.run(
        run_paced_streaming_benchmark(
            _audio(10),
            Adapter(),
            frame_duration_ms=4,
            realtime_factor=realtime_factor,
            clock=clock,
            sleeper=sleeper,
        )
    )

    assert sleeper.calls == pytest.approx(expected_sleeps)
    assert result.timing_mode == "paced"
    assert result.frame_count == 3
    assert result.audio_seconds == pytest.approx(0.010)
    assert result.audio_submission_wall_seconds == pytest.approx(expected_wall)
    assert result.finalization_latency_seconds == pytest.approx(0.0)
    assert result.total_session_wall_seconds == pytest.approx(expected_wall)
    assert result.updates[0].observed_seconds == pytest.approx(expected_wall)


def test_timing_boundaries_and_audio_progress_are_exact() -> None:
    clock = FakeClock(20.0)
    sleeper = FakeSleeper(clock)

    class Adapter:
        name = "fake-paced"
        model = "model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        async def stream_duplex(self, frames, language=None):
            clock.advance(0.010)
            iterator = aiter(frames)
            await anext(iterator)
            clock.advance(0.003)
            yield StreamingUpdate("", False)
            clock.advance(0.002)
            yield StreamingUpdate("hello", False)
            async for _ in iterator:
                pass
            clock.advance(0.007)
            yield StreamingUpdate("hello world", True)

    result = asyncio.run(
        run_paced_streaming_benchmark(
            _audio(10),
            Adapter(),
            frame_duration_ms=4,
            clock=clock,
            sleeper=sleeper,
        )
    )

    assert result.session_setup_seconds == pytest.approx(0.010)
    assert result.time_to_first_partial_seconds == pytest.approx(0.005)
    assert result.audio_submitted_seconds_at_first_partial == pytest.approx(0.004)
    assert result.audio_submission_wall_seconds == pytest.approx(0.015)
    assert result.finalization_latency_seconds == pytest.approx(0.007)
    assert result.total_session_wall_seconds == pytest.approx(0.032)
    assert result.partial_update_count == 2
    assert result.partial_stability == pytest.approx(0.5)
    assert [update.observed_seconds for update in result.updates] == pytest.approx(
        [0.003, 0.005, 0.022]
    )


def test_no_non_empty_partial_has_nullable_metrics() -> None:
    clock = FakeClock()
    sleeper = FakeSleeper(clock)

    class Adapter:
        name = "fake-paced"
        model = "model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        async def stream_duplex(self, frames, language=None):
            async for _ in frames:
                pass
            yield StreamingUpdate("final", True)

    result = asyncio.run(
        run_paced_streaming_benchmark(
            _audio(4), Adapter(), frame_duration_ms=2, clock=clock, sleeper=sleeper
        )
    )

    assert result.time_to_first_partial_seconds is None
    assert result.audio_submitted_seconds_at_first_partial is None
    assert result.partial_stability is None
    assert result.partial_update_count == 0


def test_paced_result_is_immutable() -> None:
    clock = FakeClock()
    sleeper = FakeSleeper(clock)

    class Adapter:
        name = "fake-paced"
        model = "model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        async def stream_duplex(self, frames, language=None):
            async for _ in frames:
                pass
            yield StreamingUpdate("final", True)

    result = asyncio.run(
        run_paced_streaming_benchmark(
            _audio(2), Adapter(), frame_duration_ms=2, clock=clock, sleeper=sleeper
        )
    )
    assert isinstance(result, PacedStreamingBenchmarkResult)
    with pytest.raises(FrozenInstanceError):
        result.final_text = "changed"


def test_update_before_first_frame_is_contract_error() -> None:
    class Adapter:
        name = "fake-paced"
        model = "model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        async def stream_duplex(self, frames, language=None):
            yield StreamingUpdate("too early", False)

    with pytest.raises(StreamingError, match="before first frame"):
        asyncio.run(run_paced_streaming_benchmark(_audio(2), Adapter(), frame_duration_ms=2))


def test_runner_requires_input_exhaustion_after_final_frame() -> None:
    clock = FakeClock()
    sleeper = FakeSleeper(clock)

    class Adapter:
        name = "fake-paced"
        model = "model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        async def stream_duplex(self, frames, language=None):
            iterator = aiter(frames)
            await anext(iterator)
            await anext(iterator)
            yield StreamingUpdate("final", True)

    with pytest.raises(StreamingError, match="input exhaustion"):
        asyncio.run(
            run_paced_streaming_benchmark(
                _audio(4), Adapter(), frame_duration_ms=2, clock=clock, sleeper=sleeper
            )
        )


def test_final_observed_before_simulated_audio_end_is_rejected() -> None:
    clock = FakeClock()
    sleeper = FakeSleeper(clock)

    class Adapter:
        name = "fake-paced"
        model = "model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        async def stream_duplex(self, frames, language=None):
            iterator = aiter(frames)
            await anext(iterator)
            yield StreamingUpdate("final", True)
            async for _ in iterator:
                pass

    with pytest.raises(StreamingError, match="non-negative"):
        asyncio.run(
            run_paced_streaming_benchmark(
                _audio(4), Adapter(), frame_duration_ms=2, clock=clock, sleeper=sleeper
            )
        )


def test_non_monotonic_clock_is_rejected() -> None:
    clock = FakeClock(1.0)

    class Adapter:
        name = "fake-paced"
        model = "model"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        async def stream_duplex(self, frames, language=None):
            async for _ in frames:
                clock.now = 0.5
            yield StreamingUpdate("final", True)

    with pytest.raises(StreamingError, match="monotonic"):
        asyncio.run(
            run_paced_streaming_benchmark(
                _audio(2), Adapter(), frame_duration_ms=2, clock=clock, sleeper=asyncio.sleep
            )
        )
