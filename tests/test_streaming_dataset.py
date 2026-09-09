from __future__ import annotations

import importlib
import json
import wave
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from callasr.report import ComparisonError, compare_result_artifacts
from callasr.streaming import StreamingUpdate


def _api():
    try:
        dataset_module = importlib.import_module("callasr.streaming_dataset")
        artifact_module = importlib.import_module("callasr.streaming_artifact")
    except ImportError as exc:
        pytest.fail(f"streaming dataset/artifact implementation is missing: {exc}")
    return dataset_module, artifact_module


def _write_wav(path: Path, *, marker: int = 0) -> None:
    samples = np.full(320, marker, dtype="<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(samples.tobytes())


def _write_manifest(tmp_path: Path) -> Path:
    rows = [
        {"id": "item-0", "audio": "item-0.wav", "reference": "a", "language": "ru"},
        {"id": "item-1", "audio": "item-1.wav", "reference": "b", "language": "en"},
        {"id": "item-2", "audio": "item-2.wav", "reference": "x y"},
    ]
    for index in range(3):
        _write_wav(tmp_path / f"item-{index}.wav", marker=index)
    manifest = tmp_path / "dataset.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return manifest


class FakeClock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ScriptedAdapter:
    name = "fake-stream"
    model = "fake-model"
    device = "cpu"
    compute_type = "float32"
    decoding_options: ClassVar[dict[str, object]] = {"mode": "fake"}

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls = 0
        self.languages: list[str | None] = []

    def stream(self, frames, language: str | None = None):
        materialized = tuple(frames)
        assert len(materialized) == 1
        self.languages.append(language)
        call = self.calls
        self.calls += 1
        if call == 0:
            self.clock.advance(0.1)
            yield StreamingUpdate(text="a", is_final=False)
            self.clock.advance(0.1)
            yield StreamingUpdate(text="a", is_final=True)
            self.clock.advance(0.05)
            return
        if call == 1:
            self.clock.advance(0.4)
            yield StreamingUpdate(text="b", is_final=True)
            self.clock.advance(0.05)
            return
        self.clock.advance(0.3)
        yield StreamingUpdate(text="x", is_final=False)
        self.clock.advance(0.3)
        yield StreamingUpdate(text="x y", is_final=True)
        self.clock.advance(0.05)


def test_streaming_dataset_runner_aggregates_exact_metrics_in_manifest_order(
    tmp_path: Path,
) -> None:
    dataset_module, _ = _api()
    manifest = _write_manifest(tmp_path)
    clock = FakeClock()
    adapter = ScriptedAdapter(clock)

    result = dataset_module.run_streaming_dataset_benchmark(
        manifest,
        adapter,
        frame_duration_ms=20,
        language_mode="manifest",
        clock=clock,
    )

    assert result.timing_mode == "file_upload"
    assert result.dataset_path == str(manifest.resolve())
    assert result.dataset_fingerprint.startswith("sha256:")
    assert result.frame_duration_ms == 20
    assert result.language_mode == "manifest"
    assert [item.id for item in result.items] == ["item-0", "item-1", "item-2"]
    assert [item.audio for item in result.items] == ["item-0.wav", "item-1.wav", "item-2.wav"]
    assert [item.audio_seconds for item in result.items] == pytest.approx([0.02, 0.02, 0.02])
    assert [item.frame_count for item in result.items] == [1, 1, 1]
    assert [item.final_text for item in result.items] == ["a", "b", "x y"]
    assert [item.partial_update_count for item in result.items] == [1, 0, 1]
    ttft = [item.time_to_first_partial_seconds for item in result.items]
    assert ttft[0] == pytest.approx(0.1)
    assert ttft[1] is None
    assert ttft[2] == pytest.approx(0.3)
    assert [item.finalization_latency_seconds for item in result.items] == pytest.approx(
        [0.2, 0.4, 0.6]
    )
    assert [item.total_streaming_wall_seconds for item in result.items] == pytest.approx(
        [0.25, 0.45, 0.65]
    )
    assert result.items[0].partial_stability == pytest.approx(1.0)
    assert result.items[1].partial_stability is None
    assert result.items[2].partial_stability == pytest.approx(0.5)
    assert adapter.languages == ["ru", "en", None]

    summary = result.summary
    assert summary.item_count == 3
    assert summary.completed_items == 3
    assert summary.failed_items == 0
    assert summary.total_audio_seconds == pytest.approx(0.06)
    assert summary.time_to_first_partial_count == 2
    assert summary.time_to_first_partial_p50_seconds == pytest.approx(0.1)
    assert summary.time_to_first_partial_p95_seconds == pytest.approx(0.3)
    assert summary.finalization_latency_p50_seconds == pytest.approx(0.4)
    assert summary.finalization_latency_p95_seconds == pytest.approx(0.6)
    assert summary.finalization_latency_max_seconds == pytest.approx(0.6)
    assert summary.streaming_wall_p50_seconds == pytest.approx(0.45)
    assert summary.streaming_wall_p95_seconds == pytest.approx(0.65)
    assert summary.streaming_wall_max_seconds == pytest.approx(0.65)
    assert summary.partial_stability_count == 2
    assert summary.partial_stability_mean == pytest.approx(0.75)


def test_autodetect_language_mode_passes_none_for_every_item(tmp_path: Path) -> None:
    dataset_module, _ = _api()
    manifest = _write_manifest(tmp_path)
    clock = FakeClock()
    adapter = ScriptedAdapter(clock)

    result = dataset_module.run_streaming_dataset_benchmark(
        manifest,
        adapter,
        frame_duration_ms=20,
        language_mode="autodetect",
        clock=clock,
    )

    assert result.language_mode == "autodetect"
    assert adapter.languages == [None, None, None]


@pytest.mark.parametrize("language_mode", ["", "drop", None, True])
def test_invalid_language_mode_is_rejected_before_streaming(
    tmp_path: Path,
    language_mode: object,
) -> None:
    dataset_module, _ = _api()
    manifest = _write_manifest(tmp_path)
    adapter = ScriptedAdapter(FakeClock())

    with pytest.raises(dataset_module.StreamingDatasetError, match="language_mode"):
        dataset_module.run_streaming_dataset_benchmark(
            manifest,
            adapter,
            language_mode=language_mode,
        )

    assert adapter.calls == 0


def test_no_partial_aggregates_use_null_percentiles_and_stability(tmp_path: Path) -> None:
    dataset_module, _ = _api()
    _write_wav(tmp_path / "one.wav")
    manifest = tmp_path / "dataset.jsonl"
    manifest.write_text(
        '{"id":"one","audio":"one.wav","reference":"done"}\n',
        encoding="utf-8",
    )
    clock = FakeClock()

    class FinalOnlyAdapter:
        name = "final-only"
        model = "fake"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        def stream(self, frames, language: str | None = None):
            tuple(frames)
            clock.advance(0.2)
            yield StreamingUpdate(text="done", is_final=True)
            clock.advance(0.1)

    result = dataset_module.run_streaming_dataset_benchmark(
        manifest,
        FinalOnlyAdapter(),
        clock=clock,
    )

    assert result.summary.time_to_first_partial_count == 0
    assert result.summary.time_to_first_partial_p50_seconds is None
    assert result.summary.time_to_first_partial_p95_seconds is None
    assert result.summary.partial_stability_count == 0
    assert result.summary.partial_stability_mean is None


def test_streaming_dataset_failure_propagates_without_partial_result(tmp_path: Path) -> None:
    dataset_module, _ = _api()
    manifest = _write_manifest(tmp_path)
    marker = RuntimeError("stream failed")
    calls = 0

    class FailingAdapter:
        name = "fail"
        model = "fake"
        device = "cpu"
        compute_type = "float32"
        decoding_options: ClassVar[dict[str, object]] = {}

        def stream(self, frames, language: str | None = None):
            nonlocal calls
            tuple(frames)
            calls += 1
            if calls == 2:
                raise marker
            yield StreamingUpdate(text="ok", is_final=True)

    with pytest.raises(RuntimeError) as captured:
        dataset_module.run_streaming_dataset_benchmark(manifest, FailingAdapter())

    assert captured.value is marker
    assert calls == 2


def test_streaming_result_models_are_immutable_and_public(tmp_path: Path) -> None:
    dataset_module, _ = _api()
    manifest = _write_manifest(tmp_path)
    result = dataset_module.run_streaming_dataset_benchmark(
        manifest,
        ScriptedAdapter(FakeClock()),
        clock=FakeClock(),
    )

    with pytest.raises(FrozenInstanceError):
        result.language_mode = "autodetect"

    import callasr

    assert callasr.StreamingDatasetError is dataset_module.StreamingDatasetError
    assert callasr.StreamingDatasetItemResult is dataset_module.StreamingDatasetItemResult
    assert callasr.StreamingDatasetResult is dataset_module.StreamingDatasetResult
    assert callasr.StreamingDatasetSummary is dataset_module.StreamingDatasetSummary
    assert callasr.run_streaming_dataset_benchmark is dataset_module.run_streaming_dataset_benchmark


def test_streaming_artifact_serializes_strict_shape_and_update_trace(tmp_path: Path) -> None:
    dataset_module, artifact_module = _api()
    manifest = _write_manifest(tmp_path)
    clock = FakeClock()
    result = dataset_module.run_streaming_dataset_benchmark(
        manifest,
        ScriptedAdapter(clock),
        clock=clock,
    )
    adapter_info = artifact_module.StreamingAdapterInfo(
        name="fake-stream",
        model="fake-model",
        device="cpu",
        compute_type="float32",
        options={"mode": "fake"},
    )

    artifact = artifact_module.build_streaming_artifact(
        result,
        adapter=adapter_info,
        created_at="2026-09-09T12:00:00Z",
    )
    payload = artifact_module.streaming_artifact_to_dict(artifact)

    assert payload["kind"] == "streaming"
    assert payload["schema_version"] == 1
    assert payload["timing_mode"] == "file_upload"
    assert payload["created_at"] == "2026-09-09T12:00:00Z"
    assert payload["dataset"] == {
        "path": str(manifest.resolve()),
        "item_count": 3,
        "fingerprint": result.dataset_fingerprint,
    }
    assert payload["adapter"] == {
        "name": "fake-stream",
        "model": "fake-model",
        "device": "cpu",
        "compute_type": "float32",
        "options": {"mode": "fake"},
    }
    assert payload["streaming"] == {
        "frame_duration_ms": 20,
        "language_mode": "manifest",
    }
    assert payload["summary"]["failed_items"] == 0
    assert payload["summary"]["time_to_first_partial_count"] == 2
    assert payload["items"][0]["updates"] == [
        {"text": "a", "is_final": False, "observed_seconds": pytest.approx(0.1)},
        {"text": "a", "is_final": True, "observed_seconds": pytest.approx(0.2)},
    ]
    assert payload["items"][1]["time_to_first_partial_seconds"] is None
    assert payload["items"][1]["partial_stability"] is None


def test_streaming_artifact_models_are_immutable() -> None:
    _, artifact_module = _api()
    info = artifact_module.StreamingAdapterInfo(
        name="x",
        model="y",
        device="cpu",
        compute_type="float32",
        options={},
    )

    with pytest.raises(FrozenInstanceError):
        info.model = "changed"


def test_batch_compare_explicitly_rejects_streaming_artifact(tmp_path: Path) -> None:
    path = tmp_path / "streaming.json"
    path.write_text(
        json.dumps({"kind": "streaming", "schema_version": 1}),
        encoding="utf-8",
    )

    with pytest.raises(ComparisonError, match="streaming"):
        compare_result_artifacts([path])
