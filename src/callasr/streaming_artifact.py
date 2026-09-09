"""Versioned JSON-ready artifacts for streaming dataset benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from callasr.adapters.base import AdapterOption
from callasr.streaming_dataset import (
    StreamingDatasetItemResult,
    StreamingDatasetResult,
    StreamingDatasetSummary,
)


@dataclass(frozen=True, slots=True)
class StreamingDatasetInfo:
    path: str
    item_count: int
    fingerprint: str


@dataclass(frozen=True, slots=True)
class StreamingAdapterInfo:
    name: str
    model: str
    device: str
    compute_type: str
    options: dict[str, AdapterOption]


@dataclass(frozen=True, slots=True)
class StreamingConfigInfo:
    frame_duration_ms: int
    language_mode: str


@dataclass(frozen=True, slots=True)
class StreamingArtifact:
    created_at: str
    dataset: StreamingDatasetInfo
    adapter: StreamingAdapterInfo
    streaming: StreamingConfigInfo
    summary: StreamingDatasetSummary
    items: tuple[StreamingDatasetItemResult, ...]
    kind: str = field(default="streaming", init=False)
    schema_version: int = field(default=1, init=False)
    timing_mode: str = field(default="file_upload", init=False)


def _created_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_streaming_artifact(
    result: StreamingDatasetResult,
    *,
    adapter: StreamingAdapterInfo,
    created_at: str | None = None,
) -> StreamingArtifact:
    """Combine a streaming dataset result with non-secret adapter metadata."""

    return StreamingArtifact(
        created_at=_created_at() if created_at is None else created_at,
        dataset=StreamingDatasetInfo(
            path=result.dataset_path,
            item_count=result.summary.item_count,
            fingerprint=result.dataset_fingerprint,
        ),
        adapter=adapter,
        streaming=StreamingConfigInfo(
            frame_duration_ms=result.frame_duration_ms,
            language_mode=result.language_mode,
        ),
        summary=result.summary,
        items=result.items,
    )


def streaming_artifact_to_dict(artifact: StreamingArtifact) -> dict[str, object]:
    """Return the strict streaming schema-v1 JSON-ready mapping."""

    summary = artifact.summary
    return {
        "kind": artifact.kind,
        "schema_version": artifact.schema_version,
        "timing_mode": artifact.timing_mode,
        "created_at": artifact.created_at,
        "dataset": {
            "path": artifact.dataset.path,
            "item_count": artifact.dataset.item_count,
            "fingerprint": artifact.dataset.fingerprint,
        },
        "adapter": {
            "name": artifact.adapter.name,
            "model": artifact.adapter.model,
            "device": artifact.adapter.device,
            "compute_type": artifact.adapter.compute_type,
            "options": dict(artifact.adapter.options),
        },
        "streaming": {
            "frame_duration_ms": artifact.streaming.frame_duration_ms,
            "language_mode": artifact.streaming.language_mode,
        },
        "summary": {
            "item_count": summary.item_count,
            "completed_items": summary.completed_items,
            "failed_items": summary.failed_items,
            "total_audio_seconds": summary.total_audio_seconds,
            "time_to_first_partial_count": summary.time_to_first_partial_count,
            "time_to_first_partial_p50_seconds": summary.time_to_first_partial_p50_seconds,
            "time_to_first_partial_p95_seconds": summary.time_to_first_partial_p95_seconds,
            "finalization_latency_p50_seconds": summary.finalization_latency_p50_seconds,
            "finalization_latency_p95_seconds": summary.finalization_latency_p95_seconds,
            "finalization_latency_max_seconds": summary.finalization_latency_max_seconds,
            "streaming_wall_p50_seconds": summary.streaming_wall_p50_seconds,
            "streaming_wall_p95_seconds": summary.streaming_wall_p95_seconds,
            "streaming_wall_max_seconds": summary.streaming_wall_max_seconds,
            "partial_stability_count": summary.partial_stability_count,
            "partial_stability_mean": summary.partial_stability_mean,
        },
        "items": [
            {
                "id": item.id,
                "audio": item.audio,
                "audio_seconds": item.audio_seconds,
                "frame_count": item.frame_count,
                "final_text": item.final_text,
                "partial_update_count": item.partial_update_count,
                "time_to_first_partial_seconds": item.time_to_first_partial_seconds,
                "finalization_latency_seconds": item.finalization_latency_seconds,
                "total_streaming_wall_seconds": item.total_streaming_wall_seconds,
                "partial_stability": item.partial_stability,
                "updates": [
                    {
                        "text": update.text,
                        "is_final": update.is_final,
                        "observed_seconds": update.observed_seconds,
                    }
                    for update in item.updates
                ],
            }
            for item in artifact.items
        ],
    }
