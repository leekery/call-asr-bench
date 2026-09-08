"""Python-only streaming ASR benchmark foundation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite
from numbers import Real
from time import perf_counter
from typing import Protocol, runtime_checkable

from callasr.adapters.base import AdapterOption
from callasr.audio import AudioBuffer
from callasr.metrics.wer import normalize_text


class StreamingError(ValueError):
    """A streaming adapter or benchmark contract violation."""


@dataclass(frozen=True, slots=True)
class StreamingUpdate:
    """One transcript update emitted by a streaming adapter."""

    text: str
    is_final: bool


@dataclass(frozen=True, slots=True)
class ObservedStreamingUpdate:
    """One streaming update annotated with benchmark-relative observation time."""

    text: str
    is_final: bool
    observed_seconds: float


@dataclass(frozen=True, slots=True)
class StreamingBenchmarkResult:
    """Python-only result model for one streaming transcription."""

    frame_duration_ms: int
    frame_count: int
    audio_seconds: float
    final_text: str
    updates: tuple[ObservedStreamingUpdate, ...]
    partial_update_count: int
    time_to_first_partial_seconds: float | None
    finalization_latency_seconds: float
    total_streaming_wall_seconds: float
    partial_stability: float | None


@runtime_checkable
class StreamingASRAdapter(Protocol):
    """Model-agnostic pull interface for streaming ASR adapters."""

    name: str
    model: str
    device: str
    compute_type: str

    @property
    def decoding_options(self) -> dict[str, AdapterOption]: ...

    def stream(
        self,
        frames: Iterable[AudioBuffer],
        language: str | None = None,
    ) -> Iterable[StreamingUpdate]: ...


def frame_audio(audio: AudioBuffer, frame_duration_ms: int = 20) -> tuple[AudioBuffer, ...]:
    """Split audio into deterministic exact-sample frames."""

    if (
        not isinstance(frame_duration_ms, int)
        or isinstance(frame_duration_ms, bool)
        or frame_duration_ms <= 0
    ):
        raise StreamingError("frame_duration_ms must be a positive integer")

    frame_numerator = audio.sample_rate * frame_duration_ms
    if frame_numerator % 1_000 != 0:
        raise StreamingError("frame_duration_ms must resolve to an exact integer number of samples")
    frame_samples = frame_numerator // 1_000
    if frame_samples <= 0:
        raise StreamingError("frame_duration_ms produces an empty audio frame")

    return tuple(
        AudioBuffer(audio.samples[start : start + frame_samples], audio.sample_rate)
        for start in range(0, audio.samples.size, frame_samples)
    )


def _token_edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            substitution = previous[hyp_index - 1] + (ref_token != hyp_token)
            deletion = previous[hyp_index] + 1
            insertion = current[hyp_index - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    return previous[-1]


def _normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(normalize_text(text).split())


def partial_stability_score(partials: Iterable[str], final_text: str) -> float | None:
    """Score normalized token revision stability across partials and the final transcript."""

    states: list[tuple[str, ...]] = []
    for partial in partials:
        if not isinstance(partial, str):
            raise StreamingError("partial transcript text must be a string")
        tokens = _normalized_tokens(partial)
        if not tokens:
            continue
        if not states or states[-1] != tokens:
            states.append(tokens)

    if not states:
        return None
    if not isinstance(final_text, str):
        raise StreamingError("final transcript text must be a string")
    states.append(_normalized_tokens(final_text))

    similarities: list[float] = []
    for previous, current in pairwise(states):
        denominator = max(len(previous), len(current), 1)
        edits = _token_edit_distance(previous, current)
        similarities.append(1.0 - edits / denominator)
    return sum(similarities) / len(similarities)


def _monotonic_sampler(clock: Callable[[], float]) -> Callable[[], float]:
    last_value: float | None = None

    def sample() -> float:
        nonlocal last_value
        raw = clock()
        if not isinstance(raw, Real) or isinstance(raw, bool) or not isfinite(raw):
            raise StreamingError("clock must return a finite number")
        value = float(raw)
        if last_value is not None and value < last_value:
            raise StreamingError("clock must be monotonic")
        last_value = value
        return value

    return sample


def run_streaming_benchmark(
    audio: AudioBuffer,
    adapter: StreamingASRAdapter,
    *,
    language: str | None = None,
    frame_duration_ms: int = 20,
    clock: Callable[[], float] = perf_counter,
) -> StreamingBenchmarkResult:
    """Run one audio buffer through an instrumented streaming adapter."""

    frames = frame_audio(audio, frame_duration_ms=frame_duration_ms)
    sample_clock = _monotonic_sampler(clock)
    started_at = sample_clock()
    submitted_count = 0
    last_frame_submitted_at: float | None = None

    def instrumented_frames() -> Iterable[AudioBuffer]:
        nonlocal submitted_count, last_frame_submitted_at
        for frame in frames:
            last_frame_submitted_at = sample_clock()
            submitted_count += 1
            yield frame

    try:
        updates = iter(adapter.stream(instrumented_frames(), language=language))
    except TypeError as exc:
        raise StreamingError("streaming adapter must return an iterable of updates") from exc

    observed_updates: list[ObservedStreamingUpdate] = []
    partial_texts: list[str] = []
    partial_update_count = 0
    first_partial_observed_at: float | None = None
    final_observed_at: float | None = None
    final_text: str | None = None
    final_seen = False

    for update in updates:
        observed_at = sample_clock()
        if not isinstance(update, StreamingUpdate):
            raise StreamingError("streaming adapter must yield StreamingUpdate values")
        if final_seen:
            raise StreamingError("streaming adapter yielded an update after final update")
        if not isinstance(update.text, str):
            raise StreamingError("streaming update text must be a string")
        if not isinstance(update.is_final, bool):
            raise StreamingError("streaming update is_final must be a bool")

        observed_seconds = observed_at - started_at
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

    ended_at = sample_clock()
    if submitted_count != len(frames):
        raise StreamingError("streaming adapter must consume every frame")
    if not final_seen or final_observed_at is None or final_text is None:
        raise StreamingError("streaming adapter must yield exactly one final update")
    if last_frame_submitted_at is None:
        raise StreamingError("streaming benchmark did not submit an audio frame")

    finalization_latency = final_observed_at - last_frame_submitted_at
    total_wall = ended_at - started_at
    time_to_first_partial = (
        None if first_partial_observed_at is None else first_partial_observed_at - started_at
    )
    derived = [finalization_latency, total_wall]
    if time_to_first_partial is not None:
        derived.append(time_to_first_partial)
    if any(value < 0.0 for value in derived):
        raise StreamingError("streaming timing durations must be non-negative")

    return StreamingBenchmarkResult(
        frame_duration_ms=frame_duration_ms,
        frame_count=len(frames),
        audio_seconds=audio.samples.size / audio.sample_rate,
        final_text=final_text,
        updates=tuple(observed_updates),
        partial_update_count=partial_update_count,
        time_to_first_partial_seconds=time_to_first_partial,
        finalization_latency_seconds=finalization_latency,
        total_streaming_wall_seconds=total_wall,
        partial_stability=partial_stability_score(partial_texts, final_text),
    )
