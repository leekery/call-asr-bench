from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Barrier, Event, Lock, get_ident

import numpy as np
import pytest

from callasr.adapters.base import Transcription
from callasr.audio import AudioBuffer


def _api():
    try:
        from callasr.concurrent import (
            ConcurrentBenchmarkError,
            ConcurrentBenchmarkResult,
            ConcurrentItemResult,
            run_concurrent_benchmark,
        )
    except ImportError as exc:
        pytest.fail(f"concurrent benchmark foundation is missing: {exc}")
    return (
        ConcurrentBenchmarkError,
        ConcurrentBenchmarkResult,
        ConcurrentItemResult,
        run_concurrent_benchmark,
    )


def _module():
    try:
        return importlib.import_module("callasr.concurrent")
    except ImportError as exc:
        pytest.fail(f"concurrent benchmark module is missing: {exc}")


def _write_manifest(tmp_path: Path, count: int) -> Path:
    lines: list[str] = []
    for index in range(count):
        audio = tmp_path / f"item-{index}.wav"
        audio.write_bytes(b"stub")
        language = "ru" if index % 2 == 0 else "en"
        lines.append(
            f'{{"id":"item-{index}","audio":"item-{index}.wav",'
            f'"reference":"ref {index}","language":"{language}"}}'
        )
    manifest = tmp_path / "dataset.jsonl"
    manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return manifest


def _patch_audio(monkeypatch: pytest.MonkeyPatch, sample_counts: dict[int, int]) -> None:
    module = _module()

    def fake_load_wav(path: Path) -> AudioBuffer:
        index = int(Path(path).stem.split("-")[-1])
        count = sample_counts[index]
        samples = np.full(count, index / 10.0, dtype=np.float32)
        return AudioBuffer(samples=samples, sample_rate=100)

    monkeypatch.setattr(module, "load_wav", fake_load_wav)


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)
        self._lock = Lock()

    def __call__(self) -> float:
        with self._lock:
            return next(self._values)


def test_concurrent_result_models_are_immutable_and_exported() -> None:
    (
        ConcurrentBenchmarkError,
        ConcurrentBenchmarkResult,
        ConcurrentItemResult,
        run_concurrent_benchmark,
    ) = _api()

    item = ConcurrentItemResult(
        id="x",
        audio="x.wav",
        audio_seconds=1.0,
        latency_seconds=0.2,
        hypothesis="hello",
    )
    with pytest.raises(FrozenInstanceError):
        item.hypothesis = "changed"

    assert issubclass(ConcurrentBenchmarkError, ValueError)
    assert callable(run_concurrent_benchmark)
    assert ConcurrentBenchmarkResult.__dataclass_params__.frozen is True

    import callasr

    assert callasr.ConcurrentBenchmarkError is ConcurrentBenchmarkError
    assert callasr.ConcurrentBenchmarkResult is ConcurrentBenchmarkResult
    assert callasr.ConcurrentItemResult is ConcurrentItemResult
    assert callasr.run_concurrent_benchmark is run_concurrent_benchmark


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5])
def test_invalid_concurrency_is_rejected_before_factory_use(
    tmp_path: Path,
    concurrency: object,
) -> None:
    ConcurrentBenchmarkError, _, _, run_concurrent_benchmark = _api()
    manifest = _write_manifest(tmp_path, 1)
    factory_called = False

    def factory():
        nonlocal factory_called
        factory_called = True
        raise AssertionError("factory must not run")

    with pytest.raises(ConcurrentBenchmarkError, match="concurrency"):
        run_concurrent_benchmark(manifest, factory, concurrency=concurrency)

    assert factory_called is False


def test_empty_dataset_is_rejected_before_factory_use(tmp_path: Path) -> None:
    ConcurrentBenchmarkError, _, _, run_concurrent_benchmark = _api()
    manifest = _write_manifest(tmp_path, 0)
    factory_called = False

    def factory():
        nonlocal factory_called
        factory_called = True
        raise AssertionError("factory must not run")

    with pytest.raises(ConcurrentBenchmarkError, match="at least one item"):
        run_concurrent_benchmark(manifest, factory, concurrency=2)

    assert factory_called is False


def test_concurrency_one_has_exact_timing_throughput_and_nearest_rank_percentiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, ConcurrentBenchmarkResult, _, run_concurrent_benchmark = _api()
    manifest = _write_manifest(tmp_path, 2)
    _patch_audio(monkeypatch, {0: 100, 1: 200})
    clock = SequenceClock([10.0, 10.1, 10.4, 10.5, 11.0, 11.2])
    seen_languages: list[str | None] = []

    class Adapter:
        def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcription:
            seen_languages.append(language)
            index = round(float(audio.samples[0]) * 10)
            return Transcription(text=f"hyp-{index}")

    result = run_concurrent_benchmark(
        manifest,
        Adapter,
        concurrency=1,
        clock=clock,
    )

    assert isinstance(result, ConcurrentBenchmarkResult)
    assert result.concurrency == 1
    assert result.item_count == 2
    assert result.completed_items == 2
    assert result.failed_items == 0
    assert result.total_audio_seconds == pytest.approx(3.0)
    assert result.total_wall_seconds == pytest.approx(1.2)
    assert result.throughput_speed_factor == pytest.approx(2.5)
    assert result.latency_p50_seconds == pytest.approx(0.3)
    assert result.latency_p95_seconds == pytest.approx(0.5)
    assert result.latency_max_seconds == pytest.approx(0.5)
    assert [item.id for item in result.items] == ["item-0", "item-1"]
    assert [item.audio for item in result.items] == ["item-0.wav", "item-1.wav"]
    assert [item.audio_seconds for item in result.items] == pytest.approx([1.0, 2.0])
    assert [item.latency_seconds for item in result.items] == pytest.approx([0.3, 0.5])
    assert [item.hypothesis for item in result.items] == ["hyp-0", "hyp-1"]
    assert seen_languages == ["ru", "en"]
    assert not hasattr(result, "schema_version")


def test_zero_total_wall_time_has_null_throughput(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, run_concurrent_benchmark = _api()
    manifest = _write_manifest(tmp_path, 1)
    _patch_audio(monkeypatch, {0: 100})
    clock = SequenceClock([5.0, 5.0, 5.0, 5.0])

    class Adapter:
        def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcription:
            return Transcription(text="ok")

    result = run_concurrent_benchmark(manifest, Adapter, concurrency=1, clock=clock)

    assert result.total_wall_seconds == 0.0
    assert result.throughput_speed_factor is None
    assert result.latency_p50_seconds == 0.0
    assert result.latency_p95_seconds == 0.0
    assert result.latency_max_seconds == 0.0


def test_non_monotonic_clock_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ConcurrentBenchmarkError, _, _, run_concurrent_benchmark = _api()
    manifest = _write_manifest(tmp_path, 1)
    _patch_audio(monkeypatch, {0: 100})
    clock = SequenceClock([10.0, 9.0])

    class Adapter:
        def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcription:
            return Transcription(text="never")

    with pytest.raises(ConcurrentBenchmarkError, match="monotonic"):
        run_concurrent_benchmark(manifest, Adapter, concurrency=1, clock=clock)


def test_result_order_is_manifest_order_when_workers_finish_out_of_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, run_concurrent_benchmark = _api()
    manifest = _write_manifest(tmp_path, 2)
    _patch_audio(monkeypatch, {0: 100, 1: 100})
    first_started = Event()
    allow_first = Event()

    class Adapter:
        def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcription:
            index = round(float(audio.samples[0]) * 10)
            if index == 0:
                first_started.set()
                assert allow_first.wait(timeout=5)
            else:
                assert first_started.wait(timeout=5)
                allow_first.set()
            return Transcription(text=f"hyp-{index}")

    result = run_concurrent_benchmark(manifest, Adapter, concurrency=2)

    assert [item.id for item in result.items] == ["item-0", "item-1"]
    assert [item.hypothesis for item in result.items] == ["hyp-0", "hyp-1"]


def test_each_worker_thread_reuses_one_lazily_created_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, run_concurrent_benchmark = _api()
    concurrency = 3
    manifest = _write_manifest(tmp_path, 6)
    _patch_audio(monkeypatch, {index: 100 for index in range(6)})
    barrier = Barrier(concurrency)
    state_lock = Lock()
    next_adapter_id = 0
    factory_count = 0
    seen: list[tuple[int, int]] = []

    class Adapter:
        def __init__(self, adapter_id: int) -> None:
            self.adapter_id = adapter_id
            self.first_call = True

        def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcription:
            with state_lock:
                seen.append((get_ident(), self.adapter_id))
            if self.first_call:
                self.first_call = False
                barrier.wait(timeout=5)
            return Transcription(text="ok")

    def factory() -> Adapter:
        nonlocal next_adapter_id, factory_count
        with state_lock:
            adapter_id = next_adapter_id
            next_adapter_id += 1
            factory_count += 1
        return Adapter(adapter_id)

    result = run_concurrent_benchmark(manifest, factory, concurrency=concurrency)

    by_thread: dict[int, set[int]] = {}
    for thread_id, adapter_id in seen:
        by_thread.setdefault(thread_id, set()).add(adapter_id)

    assert result.completed_items == 6
    assert len(by_thread) == concurrency
    assert all(len(adapter_ids) == 1 for adapter_ids in by_thread.values())
    assert factory_count == concurrency


def test_failure_with_concurrency_one_does_not_eagerly_submit_later_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, run_concurrent_benchmark = _api()
    manifest = _write_manifest(tmp_path, 4)
    _patch_audio(monkeypatch, {index: 100 for index in range(4)})
    seen_indices: list[int] = []

    class MarkerError(RuntimeError):
        pass

    marker = MarkerError("adapter boom")

    class Adapter:
        def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcription:
            index = round(float(audio.samples[0]) * 10)
            seen_indices.append(index)
            raise marker

    with pytest.raises(MarkerError) as captured:
        run_concurrent_benchmark(manifest, Adapter, concurrency=1)

    assert captured.value is marker
    assert seen_indices == [0]


def test_factory_error_propagates_without_partial_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, run_concurrent_benchmark = _api()
    manifest = _write_manifest(tmp_path, 2)
    _patch_audio(monkeypatch, {0: 100, 1: 100})

    class FactoryError(RuntimeError):
        pass

    marker = FactoryError("factory boom")
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        raise marker

    with pytest.raises(FactoryError) as captured:
        run_concurrent_benchmark(manifest, factory, concurrency=1)

    assert captured.value is marker
    assert calls == 1
