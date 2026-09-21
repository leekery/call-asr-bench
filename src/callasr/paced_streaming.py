"""Python-only paced full-duplex streaming ASR benchmark contract."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from time import perf_counter
from typing import Literal, Protocol, runtime_checkable

from callasr.adapters.base import AdapterOption
from callasr.audio import AudioBuffer
from callasr.streaming import (
    ObservedStreamingUpdate,
    StreamingError,
    StreamingUpdate,
    _monotonic_sampler,
    frame_audio,
    partial_stability_score,
)

Sleeper = Callable[[float], Awaitable[object]]


@dataclass(frozen=True, slots=True)
class PacedStreamingBenchmarkResult:
    """Python-only result model for one paced full-duplex transcription."""

    frame_duration_ms: int
    frame_count: int
    audio_seconds: float
    realtime_factor: float
    final_text: str
    updates: tuple[ObservedStreamingUpdate, ...]
    partial_update_count: int
    session_setup_seconds: float
    time_to_first_partial_seconds: float | None
    audio_submitted_seconds_at_first_partial: float | None
    audio_submission_wall_seconds: float
    finalization_latency_seconds: float
    total_session_wall_seconds: float
    partial_stability: float | None
    timing_mode: Literal["paced"] = field(init=False, default="paced")


@runtime_checkable
class PacedStreamingASRAdapter(Protocol):
    """Async full-duplex ASR adapter consuming benchmark-paced audio frames."""

    name: str
    model: str
    device: str
    compute_type: str

    @property
    def decoding_options(self) -> dict[str, AdapterOption]: ...

    def stream_duplex(
        self,
        frames: AsyncIterable[AudioBuffer],
        language: str | None = None,
    ) -> AsyncIterator[StreamingUpdate]: ...


def validate_realtime_factor(realtime_factor: object) -> float:
    """Validate and normalize a paced-audio speed multiplier."""

    if (
        not isinstance(realtime_factor, Real)
        or isinstance(realtime_factor, bool)
        or not isfinite(realtime_factor)
        or realtime_factor <= 0.0
    ):
        raise StreamingError("realtime_factor must be a finite positive number")
    return float(realtime_factor)


async def run_paced_streaming_benchmark(
    audio: AudioBuffer,
    adapter: PacedStreamingASRAdapter,
    *,
    language: str | None = None,
    frame_duration_ms: int = 20,
    realtime_factor: float = 1.0,
    clock: Callable[[], float] = perf_counter,
    sleeper: Sleeper = asyncio.sleep,
) -> PacedStreamingBenchmarkResult:
    """Run one audio buffer through an async adapter with benchmark-owned pacing."""

    factor = validate_realtime_factor(realtime_factor)
    frames = frame_audio(audio, frame_duration_ms=frame_duration_ms)
    sample_clock = _monotonic_sampler(clock)

    submitted_count = 0
    submitted_audio_seconds = 0.0
    first_frame_submitted_at: float | None = None
    audio_finished_at: float | None = None

    async def paced_frames() -> AsyncIterator[AudioBuffer]:
        nonlocal submitted_count
        nonlocal submitted_audio_seconds
        nonlocal first_frame_submitted_at
        nonlocal audio_finished_at

        for frame in frames:
            submitted_at = sample_clock()
            if first_frame_submitted_at is None:
                first_frame_submitted_at = submitted_at
            submitted_count += 1
            frame_seconds = frame.samples.size / frame.sample_rate
            submitted_audio_seconds += frame_seconds
            yield frame
            await sleeper(frame_seconds / factor)
        audio_finished_at = sample_clock()

    session_started_at = sample_clock()
    try:
        updates = aiter(adapter.stream_duplex(paced_frames(), language=language))
    except TypeError as exc:
        raise StreamingError(
            "paced streaming adapter must return an async iterator of updates"
        ) from exc

    observed_updates: list[ObservedStreamingUpdate] = []
    partial_texts: list[str] = []
    partial_update_count = 0
    first_partial_observed_at: float | None = None
    audio_at_first_partial: float | None = None
    final_observed_at: float | None = None
    final_text: str | None = None
    final_seen = False

    async for update in updates:
        observed_at = sample_clock()
        if first_frame_submitted_at is None:
            raise StreamingError(
                "paced streaming adapter yielded an update before first frame submission"
            )
        if not isinstance(update, StreamingUpdate):
            raise StreamingError("paced streaming adapter must yield StreamingUpdate values")
        if final_seen:
            raise StreamingError("paced streaming adapter yielded an update after final update")
        if not isinstance(update.text, str):
            raise StreamingError("streaming update text must be a string")
        if not isinstance(update.is_final, bool):
            raise StreamingError("streaming update is_final must be a bool")

        observed_seconds = observed_at - first_frame_submitted_at
        if observed_seconds < 0.0:
            raise StreamingError("streaming timing durations must be non-negative")
        observed_updates.append(
            ObservedStreamingUpdate(
                text=update.text,
                is_final=update.is_final,
                observed_seconds=observed_seconds,
            )
        )

        if update.is_final:
            final_seen = True
            final_observed_at = observed_at
            final_text = update.text
        else:
            partial_update_count += 1
            partial_texts.append(update.text)
            if first_partial_observed_at is None and update.text.strip():
                first_partial_observed_at = observed_at
                audio_at_first_partial = submitted_audio_seconds

    session_ended_at = sample_clock()
    if submitted_count != len(frames) or audio_finished_at is None:
        raise StreamingError(
            "paced streaming adapter must consume every frame through input exhaustion"
        )
    if first_frame_submitted_at is None:
        raise StreamingError("paced streaming benchmark did not submit an audio frame")
    if not final_seen or final_observed_at is None or final_text is None:
        raise StreamingError("paced streaming adapter must yield exactly one final update")

    session_setup = first_frame_submitted_at - session_started_at
    audio_submission_wall = audio_finished_at - first_frame_submitted_at
    finalization_latency = final_observed_at - audio_finished_at
    total_session_wall = session_ended_at - session_started_at
    time_to_first_partial = (
        None
        if first_partial_observed_at is None
        else first_partial_observed_at - first_frame_submitted_at
    )

    derived = [
        session_setup,
        audio_submission_wall,
        finalization_latency,
        total_session_wall,
    ]
    if time_to_first_partial is not None:
        derived.append(time_to_first_partial)
    if any(value < 0.0 for value in derived):
        raise StreamingError("streaming timing durations must be non-negative")

    return PacedStreamingBenchmarkResult(
        frame_duration_ms=frame_duration_ms,
        frame_count=len(frames),
        audio_seconds=audio.samples.size / audio.sample_rate,
        realtime_factor=factor,
        final_text=final_text,
        updates=tuple(observed_updates),
        partial_update_count=partial_update_count,
        session_setup_seconds=session_setup,
        time_to_first_partial_seconds=time_to_first_partial,
        audio_submitted_seconds_at_first_partial=audio_at_first_partial,
        audio_submission_wall_seconds=audio_submission_wall,
        finalization_latency_seconds=finalization_latency,
        total_session_wall_seconds=total_session_wall,
        partial_stability=partial_stability_score(partial_texts, final_text),
    )
