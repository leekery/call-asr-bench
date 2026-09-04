"""Benchmark result schema shared by the runner and CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

JsonScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    path: str
    item_count: int


@dataclass(frozen=True, slots=True)
class AdapterInfo:
    name: str
    model: str
    device: str
    compute_type: str
    decoding_options: dict[str, JsonScalar]


@dataclass(frozen=True, slots=True)
class ChannelInfo:
    codec: Literal["none", "pcmu", "pcma"]
    packet_loss_rate: float
    frame_duration_ms: int
    seed: int


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    total_audio_seconds: float
    adapter_seconds: float
    wer: float
    cer: float
    rtf: float
    speed_factor: float | None


@dataclass(frozen=True, slots=True)
class ItemResult:
    id: str
    audio: str
    reference: str
    hypothesis: str
    language: str | None
    audio_seconds: float
    adapter_seconds: float
    wer: float
    cer: float


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    schema_version: int = field(default=1, init=False)
    created_at: str
    dataset: DatasetInfo
    adapter: AdapterInfo
    channel: ChannelInfo
    summary: BenchmarkSummary
    items: tuple[ItemResult, ...]


def result_to_dict(result: BenchmarkResult) -> dict[str, object]:
    """Convert a benchmark result into a JSON-ready mapping."""

    payload = asdict(result)
    payload["items"] = list(payload["items"])
    return payload
