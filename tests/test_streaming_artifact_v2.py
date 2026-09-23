from __future__ import annotations

from pathlib import Path

from callasr.streaming import ObservedStreamingUpdate
from callasr.streaming_artifact import (
    StreamingAdapterInfo,
    build_streaming_artifact,
    build_streaming_artifact_v2,
    streaming_artifact_to_dict,
    streaming_artifact_v2_to_dict,
)
from callasr.streaming_dataset import (
    PacedStreamingDatasetItemResult,
    PacedStreamingDatasetResult,
    PacedStreamingDatasetSummary,
    StreamingDatasetItemResult,
    StreamingDatasetResult,
    StreamingDatasetSummary,
)


def _adapter_info() -> StreamingAdapterInfo:
    return StreamingAdapterInfo(
        name="fake-stream",
        model="fake-model",
        device="remote",
        compute_type="server",
        options={"timeout_seconds": 12.0},
    )


def _file_upload_result(tmp_path: Path) -> StreamingDatasetResult:
    item = StreamingDatasetItemResult(
        id="item-1",
        audio="one.wav",
        audio_seconds=0.02,
        frame_count=1,
        final_text="hello",
        partial_update_count=1,
        time_to_first_partial_seconds=0.1,
        finalization_latency_seconds=0.2,
        total_streaming_wall_seconds=0.25,
        partial_stability=1.0,
        updates=(
            ObservedStreamingUpdate("hel", False, 0.1),
            ObservedStreamingUpdate("hello", True, 0.2),
        ),
    )
    summary = StreamingDatasetSummary(
        item_count=1,
        completed_items=1,
        failed_items=0,
        total_audio_seconds=0.02,
        time_to_first_partial_count=1,
        time_to_first_partial_p50_seconds=0.1,
        time_to_first_partial_p95_seconds=0.1,
        finalization_latency_p50_seconds=0.2,
        finalization_latency_p95_seconds=0.2,
        finalization_latency_max_seconds=0.2,
        streaming_wall_p50_seconds=0.25,
        streaming_wall_p95_seconds=0.25,
        streaming_wall_max_seconds=0.25,
        partial_stability_count=1,
        partial_stability_mean=1.0,
    )
    return StreamingDatasetResult(
        dataset_path=str((tmp_path / "dataset.jsonl").resolve()),
        dataset_fingerprint="sha256:" + "a" * 64,
        timing_mode="file_upload",
        frame_duration_ms=20,
        language_mode="manifest",
        summary=summary,
        items=(item,),
    )


def _paced_result(tmp_path: Path) -> PacedStreamingDatasetResult:
    items = (
        PacedStreamingDatasetItemResult(
            id="paced-1",
            audio="one.wav",
            audio_seconds=0.02,
            frame_count=1,
            final_text="hello",
            partial_update_count=1,
            session_setup_seconds=0.03,
            time_to_first_partial_seconds=0.1,
            audio_submitted_seconds_at_first_partial=0.02,
            audio_submission_wall_seconds=0.02,
            finalization_latency_seconds=0.2,
            total_session_wall_seconds=0.25,
            partial_stability=1.0,
            updates=(
                ObservedStreamingUpdate("hel", False, 0.1),
                ObservedStreamingUpdate("hello", True, 0.3),
            ),
        ),
        PacedStreamingDatasetItemResult(
            id="paced-2",
            audio="two.wav",
            audio_seconds=0.04,
            frame_count=2,
            final_text="done",
            partial_update_count=0,
            session_setup_seconds=0.01,
            time_to_first_partial_seconds=None,
            audio_submitted_seconds_at_first_partial=None,
            audio_submission_wall_seconds=0.04,
            finalization_latency_seconds=0.12,
            total_session_wall_seconds=0.17,
            partial_stability=None,
            updates=(ObservedStreamingUpdate("done", True, 0.16),),
        ),
    )
    summary = PacedStreamingDatasetSummary(
        item_count=2,
        completed_items=2,
        failed_items=0,
        total_audio_seconds=0.06,
        time_to_first_partial_count=1,
        time_to_first_partial_p50_seconds=0.1,
        time_to_first_partial_p95_seconds=0.1,
        audio_submitted_at_first_partial_count=1,
        audio_submitted_at_first_partial_p50_seconds=0.02,
        audio_submitted_at_first_partial_p95_seconds=0.02,
        session_setup_p50_seconds=0.01,
        session_setup_p95_seconds=0.03,
        session_setup_max_seconds=0.03,
        audio_submission_wall_p50_seconds=0.02,
        audio_submission_wall_p95_seconds=0.04,
        audio_submission_wall_max_seconds=0.04,
        finalization_latency_p50_seconds=0.12,
        finalization_latency_p95_seconds=0.2,
        finalization_latency_max_seconds=0.2,
        total_session_wall_p50_seconds=0.17,
        total_session_wall_p95_seconds=0.25,
        total_session_wall_max_seconds=0.25,
        partial_stability_count=1,
        partial_stability_mean=1.0,
    )
    return PacedStreamingDatasetResult(
        dataset_path=str((tmp_path / "dataset.jsonl").resolve()),
        dataset_fingerprint="sha256:" + "b" * 64,
        frame_duration_ms=20,
        realtime_factor=1.5,
        language_mode="autodetect",
        summary=summary,
        items=items,
    )


def test_schema_v2_file_upload_preserves_schema_v1_metric_shape(tmp_path: Path) -> None:
    result = _file_upload_result(tmp_path)
    adapter = _adapter_info()
    v1 = streaming_artifact_to_dict(
        build_streaming_artifact(result, adapter=adapter, created_at="2026-09-23T00:00:00Z")
    )
    v2 = streaming_artifact_v2_to_dict(
        build_streaming_artifact_v2(result, adapter=adapter, created_at="2026-09-23T00:00:00Z")
    )

    assert v1["schema_version"] == 1
    assert v1["timing_mode"] == "file_upload"
    assert v2["schema_version"] == 2
    assert v2["timing_mode"] == "file_upload"
    assert v2["streaming"] == {"frame_duration_ms": 20, "language_mode": "manifest"}
    assert "realtime_factor" not in v2["streaming"]
    assert {**v2, "schema_version": 1} == v1


def test_schema_v2_paced_has_strict_timing_config_metrics_and_trace(tmp_path: Path) -> None:
    result = _paced_result(tmp_path)
    payload = streaming_artifact_v2_to_dict(
        build_streaming_artifact_v2(
            result,
            adapter=_adapter_info(),
            created_at="2026-09-23T00:00:00Z",
        )
    )

    assert set(payload) == {
        "kind",
        "schema_version",
        "timing_mode",
        "created_at",
        "dataset",
        "adapter",
        "streaming",
        "summary",
        "items",
    }
    assert payload["kind"] == "streaming"
    assert payload["schema_version"] == 2
    assert payload["timing_mode"] == "paced"
    assert payload["dataset"] == {
        "path": result.dataset_path,
        "item_count": 2,
        "fingerprint": result.dataset_fingerprint,
    }
    assert payload["streaming"] == {
        "frame_duration_ms": 20,
        "realtime_factor": 1.5,
        "language_mode": "autodetect",
    }
    assert payload["summary"] == {
        "item_count": 2,
        "completed_items": 2,
        "failed_items": 0,
        "total_audio_seconds": 0.06,
        "time_to_first_partial_count": 1,
        "time_to_first_partial_p50_seconds": 0.1,
        "time_to_first_partial_p95_seconds": 0.1,
        "audio_submitted_at_first_partial_count": 1,
        "audio_submitted_at_first_partial_p50_seconds": 0.02,
        "audio_submitted_at_first_partial_p95_seconds": 0.02,
        "session_setup_p50_seconds": 0.01,
        "session_setup_p95_seconds": 0.03,
        "session_setup_max_seconds": 0.03,
        "audio_submission_wall_p50_seconds": 0.02,
        "audio_submission_wall_p95_seconds": 0.04,
        "audio_submission_wall_max_seconds": 0.04,
        "finalization_latency_p50_seconds": 0.12,
        "finalization_latency_p95_seconds": 0.2,
        "finalization_latency_max_seconds": 0.2,
        "total_session_wall_p50_seconds": 0.17,
        "total_session_wall_p95_seconds": 0.25,
        "total_session_wall_max_seconds": 0.25,
        "partial_stability_count": 1,
        "partial_stability_mean": 1.0,
    }
    assert payload["items"][0] == {
        "id": "paced-1",
        "audio": "one.wav",
        "audio_seconds": 0.02,
        "frame_count": 1,
        "final_text": "hello",
        "partial_update_count": 1,
        "session_setup_seconds": 0.03,
        "time_to_first_partial_seconds": 0.1,
        "audio_submitted_seconds_at_first_partial": 0.02,
        "audio_submission_wall_seconds": 0.02,
        "finalization_latency_seconds": 0.2,
        "total_session_wall_seconds": 0.25,
        "partial_stability": 1.0,
        "updates": [
            {"text": "hel", "is_final": False, "observed_seconds": 0.1},
            {"text": "hello", "is_final": True, "observed_seconds": 0.3},
        ],
    }
    assert payload["items"][1]["time_to_first_partial_seconds"] is None
    assert payload["items"][1]["audio_submitted_seconds_at_first_partial"] is None
    assert payload["items"][1]["partial_stability"] is None
    assert set(payload["items"][1]) == set(payload["items"][0])
