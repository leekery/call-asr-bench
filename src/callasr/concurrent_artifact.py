"""Versioned JSON-ready artifacts for concurrent benchmark runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from callasr.adapters.base import AdapterOption
from callasr.concurrent import ConcurrentBenchmarkResult


@dataclass(frozen=True, slots=True)
class ConcurrentDatasetInfo:
    path: str
    item_count: int
    fingerprint: str


@dataclass(frozen=True, slots=True)
class ConcurrentAdapterInfo:
    name: str
    model: str
    device: str
    compute_type: str
    options: dict[str, AdapterOption]


@dataclass(frozen=True, slots=True)
class ConcurrentLoadInfo:
    concurrency: int


@dataclass(frozen=True, slots=True)
class ConcurrentSummary:
    item_count: int
    completed_items: int
    failed_items: int
    total_audio_seconds: float
    total_wall_seconds: float
    throughput_speed_factor: float | None
    latency_p50_seconds: float
    latency_p95_seconds: float
    latency_max_seconds: float


@dataclass(frozen=True, slots=True)
class ConcurrentArtifact:
    created_at: str
    dataset: ConcurrentDatasetInfo
    adapter: ConcurrentAdapterInfo
    load: ConcurrentLoadInfo
    summary: ConcurrentSummary
    items: tuple[object, ...]
    kind: str = field(default="concurrent", init=False)
    schema_version: int = field(default=1, init=False)


def _created_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_concurrent_artifact(
    result: ConcurrentBenchmarkResult,
    *,
    manifest_path: str | Path,
    fingerprint: str,
    adapter: ConcurrentAdapterInfo,
    created_at: str | None = None,
) -> ConcurrentArtifact:
    """Combine a concurrent result with reproducibility metadata."""

    resolved_manifest = Path(manifest_path).expanduser().resolve()
    return ConcurrentArtifact(
        created_at=_created_at() if created_at is None else created_at,
        dataset=ConcurrentDatasetInfo(
            path=str(resolved_manifest),
            item_count=result.item_count,
            fingerprint=fingerprint,
        ),
        adapter=adapter,
        load=ConcurrentLoadInfo(concurrency=result.concurrency),
        summary=ConcurrentSummary(
            item_count=result.item_count,
            completed_items=result.completed_items,
            failed_items=result.failed_items,
            total_audio_seconds=result.total_audio_seconds,
            total_wall_seconds=result.total_wall_seconds,
            throughput_speed_factor=result.throughput_speed_factor,
            latency_p50_seconds=result.latency_p50_seconds,
            latency_p95_seconds=result.latency_p95_seconds,
            latency_max_seconds=result.latency_max_seconds,
        ),
        items=result.items,
    )


def concurrent_artifact_to_dict(artifact: ConcurrentArtifact) -> dict[str, object]:
    """Return a JSON-ready mapping preserving the versioned artifact structure."""

    return asdict(artifact)
