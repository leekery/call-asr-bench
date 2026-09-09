"""Python-only concurrent benchmark foundation for synchronous ASR adapters."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from math import ceil, isfinite
from numbers import Real
from pathlib import Path
from threading import Lock, local
from time import perf_counter

from callasr.adapters.base import ASRAdapter
from callasr.dataset import DatasetItem, load_dataset_manifest
from callasr.io import load_wav


class ConcurrentBenchmarkError(ValueError):
    """A concurrent benchmark configuration or timing contract error."""


@dataclass(frozen=True, slots=True)
class ConcurrentItemResult:
    """Observed result for one concurrently processed manifest item."""

    id: str
    audio: str
    audio_seconds: float
    latency_seconds: float
    hypothesis: str


@dataclass(frozen=True, slots=True)
class ConcurrentBenchmarkResult:
    """Python-only aggregate result for one concurrent benchmark run."""

    concurrency: int
    item_count: int
    total_audio_seconds: float
    total_wall_seconds: float
    throughput_speed_factor: float | None
    latency_p50_seconds: float
    latency_p95_seconds: float
    latency_max_seconds: float
    completed_items: int
    failed_items: int
    items: tuple[ConcurrentItemResult, ...]


def _artifact_audio_path(audio_path: Path, manifest_path: Path) -> str:
    try:
        return str(audio_path.relative_to(manifest_path.parent))
    except ValueError:
        return str(audio_path)


def _monotonic_sampler(clock: Callable[[], float]) -> Callable[[], float]:
    lock = Lock()
    last_value: float | None = None

    def sample() -> float:
        nonlocal last_value
        with lock:
            raw = clock()
            if not isinstance(raw, Real) or isinstance(raw, bool) or not isfinite(raw):
                raise ConcurrentBenchmarkError("clock must return a finite number")
            value = float(raw)
            if last_value is not None and value < last_value:
                raise ConcurrentBenchmarkError("clock must be monotonic")
            last_value = value
            return value

    return sample


def _nearest_rank(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    rank = ceil(percentile * len(ordered) / 100)
    rank = min(max(rank, 1), len(ordered))
    return ordered[rank - 1]


def run_concurrent_benchmark(
    manifest_path: str | Path,
    adapter_factory: Callable[[], ASRAdapter],
    *,
    concurrency: int,
    clock: Callable[[], float] = perf_counter,
) -> ConcurrentBenchmarkResult:
    """Run clean manifest WAV items through worker-local synchronous ASR adapters."""

    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency <= 0:
        raise ConcurrentBenchmarkError("concurrency must be a positive integer")

    resolved_manifest = Path(manifest_path).expanduser().resolve()
    dataset_items = load_dataset_manifest(resolved_manifest)
    if not dataset_items:
        raise ConcurrentBenchmarkError("concurrent benchmark requires at least one item")

    sample_clock = _monotonic_sampler(clock)
    worker_state = local()

    def worker_adapter() -> ASRAdapter:
        adapter = getattr(worker_state, "adapter", None)
        if adapter is None:
            adapter = adapter_factory()
            worker_state.adapter = adapter
        return adapter

    def process_item(item: DatasetItem) -> ConcurrentItemResult:
        audio = load_wav(item.audio)
        audio_seconds = audio.samples.size / audio.sample_rate
        adapter = worker_adapter()
        started_at = sample_clock()
        transcription = adapter.transcribe(audio, language=item.language)
        finished_at = sample_clock()
        return ConcurrentItemResult(
            id=item.id,
            audio=_artifact_audio_path(item.audio, resolved_manifest),
            audio_seconds=audio_seconds,
            latency_seconds=finished_at - started_at,
            hypothesis=transcription.text,
        )

    ordered_results: list[ConcurrentItemResult | None] = [None] * len(dataset_items)
    next_index = 0
    failure_future: Future[ConcurrentItemResult] | None = None
    run_started_at = sample_clock()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        pending: dict[Future[ConcurrentItemResult], int] = {}

        def fill_pending() -> None:
            nonlocal next_index
            while next_index < len(dataset_items) and len(pending) < concurrency:
                index = next_index
                next_index += 1
                pending[executor.submit(process_item, dataset_items[index])] = index

        fill_pending()
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            completed = sorted(done, key=pending.__getitem__)
            failed = [future for future in completed if future.exception() is not None]
            if failed:
                failure_future = failed[0]
                for future in pending:
                    if future is not failure_future:
                        future.cancel()
                break

            for future in completed:
                index = pending.pop(future)
                ordered_results[index] = future.result()
            fill_pending()

    if failure_future is not None:
        failure_future.result()

    run_finished_at = sample_clock()
    total_wall_seconds = run_finished_at - run_started_at
    results = tuple(item for item in ordered_results if item is not None)
    if len(results) != len(dataset_items):
        raise ConcurrentBenchmarkError("concurrent benchmark completed without all item results")

    latencies = [item.latency_seconds for item in results]
    total_audio_seconds = sum(item.audio_seconds for item in results)
    throughput = (
        None if total_wall_seconds == 0.0 else total_audio_seconds / total_wall_seconds
    )
    return ConcurrentBenchmarkResult(
        concurrency=concurrency,
        item_count=len(dataset_items),
        total_audio_seconds=total_audio_seconds,
        total_wall_seconds=total_wall_seconds,
        throughput_speed_factor=throughput,
        latency_p50_seconds=_nearest_rank(latencies, 50),
        latency_p95_seconds=_nearest_rank(latencies, 95),
        latency_max_seconds=max(latencies),
        completed_items=len(results),
        failed_items=0,
        items=results,
    )
