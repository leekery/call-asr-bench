from __future__ import annotations

import importlib
import json
import wave
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from callasr.streaming import ObservedStreamingUpdate
from callasr.streaming_artifact import StreamingAdapterInfo, build_streaming_artifact
from callasr.streaming_dataset import (
    StreamingDatasetItemResult,
    StreamingDatasetResult,
    StreamingDatasetSummary,
)


def _write_wav(path: Path) -> None:
    samples = np.zeros(320, dtype="<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(samples.tobytes())


def _write_manifest(tmp_path: Path, *, language: str | None) -> Path:
    audio = tmp_path / "audio.wav"
    _write_wav(audio)
    row: dict[str, object] = {"id": "call-1", "audio": "audio.wav", "reference": "hello"}
    if language is not None:
        row["language"] = language
    manifest = tmp_path / "dataset.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


def _result(manifest: Path, *, frame_duration_ms: int, language_mode: str) -> StreamingDatasetResult:
    item = StreamingDatasetItemResult(
        id="call-1",
        audio="audio.wav",
        audio_seconds=0.02,
        frame_count=1,
        final_text="hello",
        partial_update_count=1,
        time_to_first_partial_seconds=0.1,
        finalization_latency_seconds=0.2,
        total_streaming_wall_seconds=0.25,
        partial_stability=1.0,
        updates=(
            ObservedStreamingUpdate(text="hel", is_final=False, observed_seconds=0.1),
            ObservedStreamingUpdate(text="hello", is_final=True, observed_seconds=0.2),
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
        dataset_path=str(manifest.resolve()),
        dataset_fingerprint="sha256:" + "a" * 64,
        timing_mode="file_upload",
        frame_duration_ms=frame_duration_ms,
        language_mode=language_mode,
        summary=summary,
        items=(item,),
    )


class FakeVLLMAdapter:
    name = "vllm-realtime"
    device = "remote"
    compute_type = "server"
    decoding_options: ClassVar[dict[str, object]]

    def __init__(self, model: str, base_url: str, timeout_seconds: float) -> None:
        self.model = model
        normalized = base_url.rstrip("/")
        websocket_scheme = "wss" if normalized.startswith("https://") else "ws"
        websocket_base = normalized.split("://", 1)[1]
        self.decoding_options = {
            "base_url": normalized,
            "websocket_endpoint": f"{websocket_scheme}://{websocket_base}/realtime",
            "input_format": "pcm16",
            "sample_rate_hz": 16_000,
            "timeout_seconds": timeout_seconds,
        }


def test_tagged_manifest_requires_explicit_autodetect_before_adapter_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = importlib.import_module("callasr.cli")
    manifest = _write_manifest(tmp_path, language="ru")
    constructed = False

    def constructor(*args, **kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("adapter must not be constructed")

    monkeypatch.setattr(cli, "VLLMRealtimeAdapter", constructor, raising=False)

    status = cli.main(
        [
            "streaming",
            str(manifest),
            "--adapter",
            "vllm-realtime",
            "--model",
            "model",
            "--base-url",
            "http://localhost:8000/v1",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert status == 2
    assert constructed is False
    assert "--language-mode autodetect" in capsys.readouterr().err


def test_streaming_cli_autodetect_builds_adapter_runs_dataset_and_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = importlib.import_module("callasr.cli")
    manifest = _write_manifest(tmp_path, language="ru")
    output = tmp_path / "streaming.json"
    constructor_calls: list[dict[str, object]] = []
    adapter_holder: list[FakeVLLMAdapter] = []

    def constructor(
        model: str,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float,
    ) -> FakeVLLMAdapter:
        constructor_calls.append(
            {
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
                "timeout_seconds": timeout_seconds,
            }
        )
        adapter = FakeVLLMAdapter(model, base_url, timeout_seconds)
        adapter_holder.append(adapter)
        return adapter

    def fake_runner(manifest_path, adapter, *, frame_duration_ms, language_mode, clock=None):
        assert Path(manifest_path) == manifest.resolve()
        assert adapter is adapter_holder[0]
        assert frame_duration_ms == 40
        assert language_mode == "autodetect"
        return _result(manifest, frame_duration_ms=40, language_mode="autodetect")

    monkeypatch.setattr(cli, "VLLMRealtimeAdapter", constructor, raising=False)
    monkeypatch.setattr(cli, "run_streaming_dataset_benchmark", fake_runner, raising=False)
    monkeypatch.setenv("CALLASR_VLLM_REALTIME_API_KEY", "secret-env-742b")

    status = cli.main(
        [
            "streaming",
            str(manifest),
            "--adapter",
            "vllm-realtime",
            "--model",
            "mistralai/Voxtral-Mini-4B-Realtime-2602",
            "--base-url",
            "http://localhost:8000/v1/",
            "--timeout-seconds",
            "12",
            "--frame-duration-ms",
            "40",
            "--language-mode",
            "autodetect",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert constructor_calls == [
        {
            "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
            "base_url": "http://localhost:8000/v1/",
            "api_key": "secret-env-742b",
            "timeout_seconds": 12.0,
        }
    ]
    raw = output.read_text(encoding="utf-8")
    assert "secret-env-742b" not in raw
    assert raw.endswith("\n")
    payload = json.loads(raw)
    assert payload["kind"] == "streaming"
    assert payload["schema_version"] == 1
    assert payload["timing_mode"] == "file_upload"
    assert payload["streaming"] == {"frame_duration_ms": 40, "language_mode": "autodetect"}
    assert payload["adapter"] == {
        "name": "vllm-realtime",
        "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
        "device": "remote",
        "compute_type": "server",
        "options": {
            "base_url": "http://localhost:8000/v1",
            "websocket_endpoint": "ws://localhost:8000/v1/realtime",
            "input_format": "pcm16",
            "sample_rate_hz": 16_000,
            "timeout_seconds": 12.0,
        },
    }


@pytest.mark.parametrize(
    ("explicit_key", "env_key", "expected"),
    [
        ("secret-explicit-b913", "secret-env-c452", "secret-explicit-b913"),
        (None, "secret-env-c452", "secret-env-c452"),
        (None, None, None),
    ],
)
def test_vllm_streaming_api_key_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit_key: str | None,
    env_key: str | None,
    expected: str | None,
) -> None:
    cli = importlib.import_module("callasr.cli")
    manifest = _write_manifest(tmp_path, language=None)
    output = tmp_path / "result.json"
    captured: list[str | None] = []

    if env_key is None:
        monkeypatch.delenv("CALLASR_VLLM_REALTIME_API_KEY", raising=False)
    else:
        monkeypatch.setenv("CALLASR_VLLM_REALTIME_API_KEY", env_key)

    def constructor(model: str, *, base_url: str, api_key: str | None, timeout_seconds: float):
        captured.append(api_key)
        return FakeVLLMAdapter(model, base_url, timeout_seconds)

    def fake_runner(manifest_path, adapter, *, frame_duration_ms, language_mode, clock=None):
        return _result(manifest, frame_duration_ms=frame_duration_ms, language_mode=language_mode)

    monkeypatch.setattr(cli, "VLLMRealtimeAdapter", constructor, raising=False)
    monkeypatch.setattr(cli, "run_streaming_dataset_benchmark", fake_runner, raising=False)

    argv = [
        "streaming",
        str(manifest),
        "--adapter",
        "vllm-realtime",
        "--model",
        "model",
        "--base-url",
        "https://asr.example/v1",
        "--output",
        str(output),
    ]
    if explicit_key is not None:
        argv.extend(["--api-key", explicit_key])

    assert cli.main(argv) == 0
    assert captured == [expected]
    raw = output.read_text(encoding="utf-8")
    for secret in (explicit_key, env_key):
        if secret is not None:
            assert secret not in raw


def test_untagged_manifest_allows_default_manifest_language_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = importlib.import_module("callasr.cli")
    manifest = _write_manifest(tmp_path, language=None)
    seen_modes: list[str] = []

    monkeypatch.setattr(
        cli,
        "VLLMRealtimeAdapter",
        lambda model, **kwargs: FakeVLLMAdapter(model, kwargs["base_url"], kwargs["timeout_seconds"]),
        raising=False,
    )

    def fake_runner(manifest_path, adapter, *, frame_duration_ms, language_mode, clock=None):
        seen_modes.append(language_mode)
        return _result(manifest, frame_duration_ms=frame_duration_ms, language_mode=language_mode)

    monkeypatch.setattr(cli, "run_streaming_dataset_benchmark", fake_runner, raising=False)

    assert (
        cli.main(
            [
                "streaming",
                str(manifest),
                "--adapter",
                "vllm-realtime",
                "--model",
                "model",
                "--base-url",
                "http://localhost:8000/v1",
                "--output",
                str(tmp_path / "result.json"),
            ]
        )
        == 0
    )
    assert seen_modes == ["manifest"]


def test_streaming_writer_preserves_existing_file_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = importlib.import_module("callasr.cli")
    output = tmp_path / "result.json"
    output.write_text("old\n", encoding="utf-8")
    manifest = _write_manifest(tmp_path, language=None)
    result = _result(manifest, frame_duration_ms=20, language_mode="manifest")
    artifact = build_streaming_artifact(
        result,
        adapter=StreamingAdapterInfo(
            name="vllm-realtime",
            model="model",
            device="remote",
            compute_type="server",
            options={},
        ),
        created_at="2026-09-09T12:00:00Z",
    )

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(cli.os, "replace", fail_replace)

    with pytest.raises(cli.ArtifactError, match="replace failed"):
        cli.write_streaming_artifact(artifact, output)

    assert output.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_streaming_runner_failure_does_not_replace_existing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = importlib.import_module("callasr.cli")
    manifest = _write_manifest(tmp_path, language=None)
    output = tmp_path / "result.json"
    output.write_text("old\n", encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "VLLMRealtimeAdapter",
        lambda model, **kwargs: FakeVLLMAdapter(model, kwargs["base_url"], kwargs["timeout_seconds"]),
        raising=False,
    )

    def fail_runner(*args, **kwargs):
        raise cli.StreamingDatasetError("stream failed")

    monkeypatch.setattr(cli, "run_streaming_dataset_benchmark", fail_runner, raising=False)

    status = cli.main(
        [
            "streaming",
            str(manifest),
            "--adapter",
            "vllm-realtime",
            "--model",
            "model",
            "--base-url",
            "http://localhost:8000/v1",
            "--output",
            str(output),
        ]
    )

    assert status == 2
    assert "stream failed" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "old\n"
