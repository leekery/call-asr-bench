"""Dataset-level aggregation for the Python streaming benchmark foundation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Literal

from callasr.dataset import dataset_fingerprint, load_dataset_manifest
from callasr.io import load_wav
from callasr.paced_streaming import (
    PacedStreamingASRAdapter,
    PacedStreamingBenchmarkResult,
    Sleeper,
    run_paced_streaming_benchmark,
    validate_realtime_factor,
)
from callasr.streaming import (
    ObservedStreamingUpdate,
    StreamingASRAdapter,
    run_streaming_benchmark,
)

LanguageMode = Literal["manifest", "autodetect"]


class StreamingDatasetError(ValueError):
    """A streaming dataset benchmark configuration error."""


@dataclass(frozen=True, slots=True)
class StreamingDatasetItemResult:
    """Streaming metrics and trace for one manifest item."""

    id: str
    audio: str
    audio_seconds: float
    frame_count: int
    final_text: str
    partial_update_count: int
    time_to_first_partial_seconds: float | None
    finalization_latency_seconds: float
    total_streaming_wall_seconds: float
    partial_stability: float | None
    updates: tuple[ObservedStreamingUpdate, ...]


@dataclass(frozen=True, slots=True)
class StreamingDatasetSummary:
    """Aggregate metrics across a successful streaming dataset run."""

    item_count: int
    completed_items: int
    failed_items: int
    total_audio_seconds: float
    time_to_first_partial_count: int
    time_to_first_partial_p50_seconds: float | None
    time_to_first_partial_p95_seconds: float | None
    finalization_latency_p50_seconds: float
    finalization_latency_p95_seconds: float
    finalization_latency_max_seconds: float
    streaming_wall_p50_seconds: float
    streaming_wall_p95_seconds: float
    streaming_wall_max_seconds: float
    partial_stability_count: int
    partial_stability_mean: float | None


@dataclass(frozen=True, slots=True)
class StreamingDatasetResult:
    """Python-only result for a manifest-wide streaming benchmark."""

    dataset_path: str
    dataset_fingerprint: str
    timing_mode: Literal["file_upload"]
    frame_duration_ms: int
    language_mode: LanguageMode
    summary: StreamingDatasetSummary
    items: tuple[StreamingDatasetItemResult, ...]


@dataclass(frozen=True, slots=True)
class PacedStreamingDatasetItemResult:
    """Paced streaming metrics and update trace for one manifest item."""

    id: str
    audio: str
    audio_seconds: float
    frame_count: int
    final_text: str
    partial_update_count: int
    session_setup_seconds: float
    time_to_first_partial_seconds: float | None
    audio_submitted_seconds_at_first_partial: float | None
    audio_submission_wall_seconds: float
    finalization_latency_seconds: float
    total_session_wall_seconds: float
    partial_stability: float | None
    updates: tuple[ObservedStreamingUpdate, ...]


@dataclass(frozen=True, slots=True)
class PacedStreamingDatasetSummary:
    """Aggregate metrics across a successful paced streaming dataset run."""

    item_count: int
    completed_items: int
    failed_items: int
    total_audio_seconds: float
    time_to_first_partial_count: int
    time_to_first_partial_p50_seconds: float | None
    time_to_first_partial_p95_seconds: float | None
    audio_submitted_at_first_partial_count: int
    audio_submitted_at_first_partial_p50_seconds: float | None
    audio_submitted_at_first_partial_p95_seconds: float | None
    session_setup_p50_seconds: float
    session_setup_p95_seconds: float
    session_setup_max_seconds: float
    audio_submission_wall_p50_seconds: float
    audio_submission_wall_p95_seconds: float
    audio_submission_wall_max_seconds: float
    finalization_latency_p50_seconds: float
    finalization_latency_p95_seconds: float
    finalization_latency_max_seconds: float
    total_session_wall_p50_seconds: float
    total_session_wall_p95_seconds: float
    total_session_wall_max_seconds: float
    partial_stability_count: int
    partial_stability_mean: float | None


@dataclass(frozen=True, slots=True)
class PacedStreamingDatasetResult:
    """Python-only result for a manifest-wide paced streaming benchmark."""

    dataset_path: str
    dataset_fingerprint: str
    frame_duration_ms: int
    realtime_factor: float
    language_mode: LanguageMode
    summary: PacedStreamingDatasetSummary
    items: tuple[PacedStreamingDatasetItemResult, ...]
    timing_mode: Literal["paced"] = field(init=False, default="paced")


def _artifact_audio_path(audio_path: Path, manifest_path: Path) -> str:
    try:
        return str(audio_path.relative_to(manifest_path.parent))
    except ValueError:
        return str(audio_path)


def _nearest_rank(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    rank = ceil(percentile * len(ordered) / 100)
    rank = min(max(rank, 1), len(ordered))
    return ordered[rank - 1]


def _optional_nearest_rank(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    return _nearest_rank(values, percentile)


def run_streaming_dataset_benchmark(
    manifest_path: str | Path,
    adapter: StreamingASRAdapter,
    *,
    frame_duration_ms: int = 20,
    language_mode: LanguageMode = "manifest",
    clock: Callable[[], float] = perf_counter,
) -> StreamingDatasetResult:
    """Run a manifest sequentially through the existing streaming benchmark contract."""

    if language_mode not in {"manifest", "autodetect"}:
        raise StreamingDatasetError("language_mode must be 'manifest' or 'autodetect'")

    resolved_manifest = Path(manifest_path).expanduser().resolve()
    dataset_items = load_dataset_manifest(resolved_manifest)
    if not dataset_items:
        raise StreamingDatasetError("streaming dataset benchmark requires at least one item")

    item_results: list[StreamingDatasetItemResult] = []
    for item in dataset_items:
        audio = load_wav(item.audio)
        language = item.language if language_mode == "manifest" else None
        result = run_streaming_benchmark(
            audio,
            adapter,
            language=language,
            frame_duration_ms=frame_duration_ms,
            clock=clock,
        )
        item_results.append(
            StreamingDatasetItemResult(
                id=item.id,
                audio=_artifact_audio_path(item.audio, resolved_manifest),
                audio_seconds=result.audio_seconds,
                frame_count=result.frame_count,
                final_text=result.final_text,
                partial_update_count=result.partial_update_count,
                time_to_first_partial_seconds=result.time_to_first_partial_seconds,
                finalization_latency_seconds=result.finalization_latency_seconds,
                total_streaming_wall_seconds=result.total_streaming_wall_seconds,
                partial_stability=result.partial_stability,
                updates=result.updates,
            )
        )

    ttft_values = [
        item.time_to_first_partial_seconds
        for item in item_results
        if item.time_to_first_partial_seconds is not None
    ]
    finalization_values = [item.finalization_latency_seconds for item in item_results]
    wall_values = [item.total_streaming_wall_seconds for item in item_results]
    stability_values = [
        item.partial_stability for item in item_results if item.partial_stability is not None
    ]

    summary = StreamingDatasetSummary(
        item_count=len(item_results),
        completed_items=len(item_results),
        failed_items=0,
        total_audio_seconds=sum(item.audio_seconds for item in item_results),
        time_to_first_partial_count=len(ttft_values),
        time_to_first_partial_p50_seconds=_optional_nearest_rank(ttft_values, 50),
        time_to_first_partial_p95_seconds=_optional_nearest_rank(ttft_values, 95),
        finalization_latency_p50_seconds=_nearest_rank(finalization_values, 50),
        finalization_latency_p95_seconds=_nearest_rank(finalization_values, 95),
        finalization_latency_max_seconds=max(finalization_values),
        streaming_wall_p50_seconds=_nearest_rank(wall_values, 50),
        streaming_wall_p95_seconds=_nearest_rank(wall_values, 95),
        streaming_wall_max_seconds=max(wall_values),
        partial_stability_count=len(stability_values),
        partial_stability_mean=(
            None if not stability_values else sum(stability_values) / len(stability_values)
        ),
    )
    return StreamingDatasetResult(
        dataset_path=str(resolved_manifest),
        dataset_fingerprint=dataset_fingerprint(dataset_items),
        timing_mode="file_upload",
        frame_duration_ms=frame_duration_ms,
        language_mode=language_mode,
        summary=summary,
        items=tuple(item_results),
    )


async def run_paced_streaming_dataset_benchmark(
    manifest_path: str | Path,
    adapter: PacedStreamingASRAdapter,
    *,
    frame_duration_ms: int = 20,
    realtime_factor: float = 1.0,
    language_mode: LanguageMode = "manifest",
    clock: Callable[[], float] = perf_counter,
    sleeper: Sleeper = asyncio.sleep,
) -> PacedStreamingDatasetResult:
    """Run a manifest sequentially through the paced duplex benchmark contract."""

    if language_mode not in {"manifest", "autodetect"}:
        raise StreamingDatasetError("language_mode must be 'manifest' or 'autodetect'")
    factor = validate_realtime_factor(realtime_factor)

    resolved_manifest = Path(manifest_path).expanduser().resolve()
    dataset_items = load_dataset_manifest(resolved_manifest)
    if not dataset_items:
        raise StreamingDatasetError("paced streaming dataset benchmark requires at least one item")

    item_results: list[PacedStreamingDatasetItemResult] = []
    for item in dataset_items:
        audio = load_wav(item.audio)
        language = item.language if language_mode == "manifest" else None
        result: PacedStreamingBenchmarkResult = await run_paced_streaming_benchmark(
            audio,
            adapter,
            language=language,
            frame_duration_ms=frame_duration_ms,
            realtime_factor=factor,
            clock=clock,
            sleeper=sleeper,
        )
        item_results.append(
            PacedStreamingDatasetItemResult(
                id=item.id,
                audio=_artifact_audio_path(item.audio, resolved_manifest),
                audio_seconds=result.audio_seconds,
                frame_count=result.frame_count,
                final_text=result.final_text,
                partial_update_count=result.partial_update_count,
                session_setup_seconds=result.session_setup_seconds,
                time_to_first_partial_seconds=result.time_to_first_partial_seconds,
                audio_submitted_seconds_at_first_partial=(
                    result.audio_submitted_seconds_at_first_partial
                ),
                audio_submission_wall_seconds=result.audio_submission_wall_seconds,
                finalization_latency_seconds=result.finalization_latency_seconds,
                total_session_wall_seconds=result.total_session_wall_seconds,
                partial_stability=result.partial_stability,
                updates=result.updates,
            )
        )

    ttft_values = [
        item.time_to_first_partial_seconds
        for item in item_results
        if item.time_to_first_partial_seconds is not None
    ]
    audio_progress_values = [
        item.audio_submitted_seconds_at_first_partial
        for item in item_results
        if item.audio_submitted_seconds_at_first_partial is not None
    ]
    session_setup_values = [item.session_setup_seconds for item in item_results]
    audio_submission_values = [item.audio_submission_wall_seconds for item in item_results]
    finalization_values = [item.finalization_latency_seconds for item in item_results]
    total_session_values = [item.total_session_wall_seconds for item in item_results]
    stability_values = [
        item.partial_stability for item in item_results if item.partial_stability is not None
    ]

    summary = PacedStreamingDatasetSummary(
        item_count=len(item_results),
        completed_items=len(item_results),
        failed_items=0,
        total_audio_seconds=sum(item.audio_seconds for item in item_results),
        time_to_first_partial_count=len(ttft_values),
        time_to_first_partial_p50_seconds=_optional_nearest_rank(ttft_values, 50),
        time_to_first_partial_p95_seconds=_optional_nearest_rank(ttft_values, 95),
        audio_submitted_at_first_partial_count=len(audio_progress_values),
        audio_submitted_at_first_partial_p50_seconds=_optional_nearest_rank(
            audio_progress_values, 50
        ),
        audio_submitted_at_first_partial_p95_seconds=_optional_nearest_rank(
            audio_progress_values, 95
        ),
        session_setup_p50_seconds=_nearest_rank(session_setup_values, 50),
        session_setup_p95_seconds=_nearest_rank(session_setup_values, 95),
        session_setup_max_seconds=max(session_setup_values),
        audio_submission_wall_p50_seconds=_nearest_rank(audio_submission_values, 50),
        audio_submission_wall_p95_seconds=_nearest_rank(audio_submission_values, 95),
        audio_submission_wall_max_seconds=max(audio_submission_values),
        finalization_latency_p50_seconds=_nearest_rank(finalization_values, 50),
        finalization_latency_p95_seconds=_nearest_rank(finalization_values, 95),
        finalization_latency_max_seconds=max(finalization_values),
        total_session_wall_p50_seconds=_nearest_rank(total_session_values, 50),
        total_session_wall_p95_seconds=_nearest_rank(total_session_values, 95),
        total_session_wall_max_seconds=max(total_session_values),
        partial_stability_count=len(stability_values),
        partial_stability_mean=(
            None if not stability_values else sum(stability_values) / len(stability_values)
        ),
    )
    return PacedStreamingDatasetResult(
        dataset_path=str(resolved_manifest),
        dataset_fingerprint=dataset_fingerprint(dataset_items),
        frame_duration_ms=frame_duration_ms,
        realtime_factor=factor,
        language_mode=language_mode,
        summary=summary,
        items=tuple(item_results),
    )
