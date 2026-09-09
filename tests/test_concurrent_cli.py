from __future__ import annotations

import importlib
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from callasr.concurrent import ConcurrentBenchmarkResult, ConcurrentItemResult
from callasr.report import ComparisonError, compare_result_artifacts


def _artifact_api():
    try:
        module = importlib.import_module("callasr.concurrent_artifact")
    except ImportError as exc:
        pytest.fail(f"concurrent artifact implementation is missing: {exc}")
    return module


def _write_wav(path: Path, sample_count: int = 160) -> None:
    samples = np.zeros(sample_count, dtype="<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(samples.tobytes())


def _write_manifest(tmp_path: Path) -> Path:
    audio = tmp_path / "audio.wav"
    _write_wav(audio)
    manifest = tmp_path / "dataset.jsonl"
    manifest.write_text(
        '{"id":"call-1","audio":"audio.wav","reference":"hello","language":"en"}\n',
        encoding="utf-8",
    )
    return manifest


def _result(concurrency: int = 2) -> ConcurrentBenchmarkResult:
    return ConcurrentBenchmarkResult(
        concurrency=concurrency,
        item_count=1,
        total_audio_seconds=1.5,
        total_wall_seconds=0.5,
        throughput_speed_factor=3.0,
        latency_p50_seconds=0.2,
        latency_p95_seconds=0.2,
        latency_max_seconds=0.2,
        completed_items=1,
        failed_items=0,
        items=(
            ConcurrentItemResult(
                id="call-1",
                audio="audio.wav",
                audio_seconds=1.5,
                latency_seconds=0.2,
                hypothesis="hello",
            ),
        ),
    )


def test_concurrent_artifact_has_separate_versioned_shape() -> None:
    module = _artifact_api()
    adapter = module.ConcurrentAdapterInfo(
        name="openai-compatible",
        model="served-model",
        device="remote",
        compute_type="server",
        options={
            "base_url": "http://localhost:8000/v1",
            "timeout_seconds": 60.0,
            "response_format": "json",
            "upload_format": "wav_pcm16",
        },
    )

    artifact = module.build_concurrent_artifact(
        _result(concurrency=16),
        manifest_path="/bench/dataset.jsonl",
        fingerprint="sha256:" + "a" * 64,
        adapter=adapter,
        created_at="2026-09-09T10:00:00Z",
    )
    payload = module.concurrent_artifact_to_dict(artifact)

    assert payload == {
        "kind": "concurrent",
        "schema_version": 1,
        "created_at": "2026-09-09T10:00:00Z",
        "dataset": {
            "path": "/bench/dataset.jsonl",
            "item_count": 1,
            "fingerprint": "sha256:" + "a" * 64,
        },
        "adapter": {
            "name": "openai-compatible",
            "model": "served-model",
            "device": "remote",
            "compute_type": "server",
            "options": {
                "base_url": "http://localhost:8000/v1",
                "timeout_seconds": 60.0,
                "response_format": "json",
                "upload_format": "wav_pcm16",
            },
        },
        "load": {"concurrency": 16},
        "summary": {
            "item_count": 1,
            "completed_items": 1,
            "failed_items": 0,
            "total_audio_seconds": 1.5,
            "total_wall_seconds": 0.5,
            "throughput_speed_factor": 3.0,
            "latency_p50_seconds": 0.2,
            "latency_p95_seconds": 0.2,
            "latency_max_seconds": 0.2,
        },
        "items": [
            {
                "id": "call-1",
                "audio": "audio.wav",
                "audio_seconds": 1.5,
                "latency_seconds": 0.2,
                "hypothesis": "hello",
            }
        ],
    }
    assert not hasattr(artifact, "channel")


def test_concurrent_cli_passes_lazy_factory_and_writes_faster_whisper_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = importlib.import_module("callasr.cli")
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "load.json"
    constructor_calls: list[tuple[str, str, str]] = []

    class FakeAdapter:
        pass

    def fake_constructor(model: str, *, device: str, compute_type: str) -> FakeAdapter:
        constructor_calls.append((model, device, compute_type))
        return FakeAdapter()

    def fake_runner(manifest_path, adapter_factory, *, concurrency, clock=None):
        assert Path(manifest_path) == manifest
        assert concurrency == 3
        assert constructor_calls == []
        first = adapter_factory()
        second = adapter_factory()
        assert isinstance(first, FakeAdapter)
        assert isinstance(second, FakeAdapter)
        assert first is not second
        return _result(concurrency=concurrency)

    monkeypatch.setattr(cli, "FasterWhisperAdapter", fake_constructor)
    monkeypatch.setattr(cli, "run_concurrent_benchmark", fake_runner, raising=False)

    status = cli.main(
        [
            "concurrent",
            str(manifest),
            "--adapter",
            "faster-whisper",
            "--model",
            "large-v3",
            "--concurrency",
            "3",
            "--device",
            "cuda",
            "--compute-type",
            "float16",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert constructor_calls == [
        ("large-v3", "cuda", "float16"),
        ("large-v3", "cuda", "float16"),
    ]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["kind"] == "concurrent"
    assert payload["schema_version"] == 1
    assert payload["dataset"]["path"] == str(manifest.resolve())
    assert payload["dataset"]["fingerprint"].startswith("sha256:")
    assert payload["adapter"] == {
        "name": "faster-whisper",
        "model": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
        "options": {"beam_size": 5, "temperature": 0.0},
    }
    assert payload["load"] == {"concurrency": 3}
    assert payload["items"][0]["id"] == "call-1"
    assert output.read_bytes().endswith(b"\n")


@pytest.mark.parametrize(
    ("explicit_key", "callasr_key", "openai_key", "expected"),
    [
        ("explicit", "callasr", "openai", "explicit"),
        (None, "callasr", "openai", "callasr"),
        (None, None, "openai", "openai"),
        (None, None, None, None),
    ],
)
def test_concurrent_openai_key_precedence_and_secret_free_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit_key: str | None,
    callasr_key: str | None,
    openai_key: str | None,
    expected: str | None,
) -> None:
    cli = importlib.import_module("callasr.cli")
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "load.json"
    captured_keys: list[str | None] = []

    if callasr_key is None:
        monkeypatch.delenv("CALLASR_API_KEY", raising=False)
    else:
        monkeypatch.setenv("CALLASR_API_KEY", callasr_key)
    if openai_key is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", openai_key)

    class FakeAdapter:
        pass

    def fake_constructor(
        model: str,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float,
    ) -> FakeAdapter:
        assert model == "served"
        assert base_url == "https://asr.example/v1"
        assert timeout_seconds == 12.0
        captured_keys.append(api_key)
        return FakeAdapter()

    def fake_runner(manifest_path, adapter_factory, *, concurrency, clock=None):
        assert captured_keys == []
        adapter_factory()
        return _result(concurrency=concurrency)

    monkeypatch.setattr(cli, "OpenAICompatibleAdapter", fake_constructor)
    monkeypatch.setattr(cli, "run_concurrent_benchmark", fake_runner, raising=False)

    argv = [
        "concurrent",
        str(manifest),
        "--adapter",
        "openai-compatible",
        "--model",
        "served",
        "--concurrency",
        "2",
        "--base-url",
        "https://asr.example/v1/",
        "--timeout-seconds",
        "12",
        "--output",
        str(output),
    ]
    if explicit_key is not None:
        argv.extend(["--api-key", explicit_key])

    assert cli.main(argv) == 0
    assert captured_keys == [expected]
    raw = output.read_text(encoding="utf-8")
    for secret in (explicit_key, callasr_key, openai_key):
        if secret is not None:
            assert secret not in raw
    payload = json.loads(raw)
    assert payload["adapter"] == {
        "name": "openai-compatible",
        "model": "served",
        "device": "remote",
        "compute_type": "server",
        "options": {
            "base_url": "https://asr.example/v1",
            "timeout_seconds": 12.0,
            "response_format": "json",
            "upload_format": "wav_pcm16",
        },
    }


def test_concurrent_validation_happens_before_adapter_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = importlib.import_module("callasr.cli")
    manifest = _write_manifest(tmp_path)
    constructed = False

    def constructor(*args, **kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("adapter must not be constructed")

    monkeypatch.setattr(cli, "OpenAICompatibleAdapter", constructor)

    status = cli.main(
        [
            "concurrent",
            str(manifest),
            "--adapter",
            "openai-compatible",
            "--model",
            "served",
            "--concurrency",
            "2",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert status == 2
    assert constructed is False
    assert "base-url is required" in capsys.readouterr().err


def test_concurrent_artifact_writer_preserves_existing_file_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = importlib.import_module("callasr.cli")
    module = _artifact_api()
    output = tmp_path / "result.json"
    output.write_text("old\n", encoding="utf-8")
    artifact = module.build_concurrent_artifact(
        _result(),
        manifest_path=tmp_path / "dataset.jsonl",
        fingerprint="sha256:" + "b" * 64,
        adapter=module.ConcurrentAdapterInfo(
            name="faster-whisper",
            model="large-v3",
            device="cpu",
            compute_type="int8",
            options={"beam_size": 5, "temperature": 0.0},
        ),
        created_at="2026-09-09T10:00:00Z",
    )

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(cli.os, "replace", fail_replace)

    with pytest.raises(cli.ArtifactError, match="replace failed"):
        cli.write_concurrent_artifact(artifact, output)

    assert output.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_batch_compare_explicitly_rejects_concurrent_artifact(tmp_path: Path) -> None:
    path = tmp_path / "load.json"
    path.write_text(
        json.dumps({"kind": "concurrent", "schema_version": 1}),
        encoding="utf-8",
    )

    with pytest.raises(ComparisonError, match="concurrent"):
        compare_result_artifacts([path])
