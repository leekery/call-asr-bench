from __future__ import annotations

import json
from pathlib import Path

import pytest

from callasr.adapters.base import AdapterError
from callasr.benchmark import (
    AdapterInfo,
    BenchmarkResult,
    BenchmarkSummary,
    ChannelInfo,
    DatasetInfo,
    ItemResult,
)


def _result() -> BenchmarkResult:
    return BenchmarkResult(
        created_at="2026-09-04T17:00:00+00:00",
        dataset=DatasetInfo(path="/tmp/dataset.jsonl", item_count=1),
        adapter=AdapterInfo(
            name="faster-whisper",
            model="large-v3",
            device="cuda",
            compute_type="float16",
            decoding_options={"beam_size": 5, "temperature": 0.0},
        ),
        channel=ChannelInfo(
            codec="pcmu",
            packet_loss_rate=0.05,
            frame_duration_ms=20,
            seed=42,
        ),
        summary=BenchmarkSummary(
            total_audio_seconds=1.0,
            adapter_seconds=0.25,
            wer=0.0,
            cer=0.0,
            rtf=0.25,
            speed_factor=4.0,
        ),
        items=(
            ItemResult(
                id="call-1",
                audio="audio/call-1.wav",
                reference="добрый день",
                hypothesis="добрый день",
                language="ru",
                audio_seconds=1.0,
                adapter_seconds=0.25,
                wer=0.0,
                cer=0.0,
            ),
        ),
    )


def _argv(output: Path) -> list[str]:
    return [
        "run",
        "dataset.jsonl",
        "--adapter",
        "faster-whisper",
        "--model",
        "large-v3",
        "--codec",
        "pcmu",
        "--packet-loss-rate",
        "0.05",
        "--frame-duration-ms",
        "20",
        "--seed",
        "42",
        "--output",
        str(output),
        "--device",
        "cuda",
        "--compute-type",
        "float16",
    ]


def test_run_constructs_adapter_runs_benchmark_and_writes_utf8_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from callasr import cli

    seen: dict[str, object] = {}

    class FakeAdapter:
        pass

    def make_adapter(model: str, *, device: str, compute_type: str) -> FakeAdapter:
        seen["adapter"] = (model, device, compute_type)
        return FakeAdapter()

    def run(manifest: str, adapter: FakeAdapter, **kwargs: object) -> BenchmarkResult:
        seen["run"] = (manifest, adapter, kwargs)
        return _result()

    monkeypatch.setattr(cli, "FasterWhisperAdapter", make_adapter)
    monkeypatch.setattr(cli, "run_benchmark", run)
    output = tmp_path / "nested" / "result.json"

    assert cli.main(_argv(output)) == 0

    assert seen["adapter"] == ("large-v3", "cuda", "float16")
    manifest, _, kwargs = seen["run"]
    assert manifest == "dataset.jsonl"
    assert kwargs == {
        "codec": "pcmu",
        "packet_loss_rate": 0.05,
        "frame_duration_ms": 20,
        "seed": 42,
    }
    raw = output.read_text(encoding="utf-8")
    assert "добрый день" in raw
    assert "\\u0434" not in raw
    assert raw.startswith("{\n  ")
    payload = json.loads(raw)
    assert payload["schema_version"] == 1
    assert payload["items"][0]["language"] == "ru"


def test_clean_codec_rejects_packet_loss_before_adapter_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from callasr import cli

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("adapter must not be constructed")

    monkeypatch.setattr(cli, "FasterWhisperAdapter", forbidden)
    output = tmp_path / "result.json"
    argv = _argv(output)
    argv[argv.index("pcmu")] = "none"

    assert cli.main(argv) == 2
    assert "packet-loss-rate must be zero when codec is none" in capsys.readouterr().err
    assert not output.exists()


def test_expected_domain_error_returns_two_without_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from callasr import cli

    monkeypatch.setattr(cli, "FasterWhisperAdapter", lambda *args, **kwargs: object())

    def fail(*args: object, **kwargs: object) -> BenchmarkResult:
        raise AdapterError("model unavailable")

    monkeypatch.setattr(cli, "run_benchmark", fail)
    output = tmp_path / "result.json"

    assert cli.main(_argv(output)) == 2
    assert capsys.readouterr().err.strip() == "error: model unavailable"
    assert not output.exists()


def test_failed_run_preserves_existing_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from callasr import cli

    monkeypatch.setattr(cli, "FasterWhisperAdapter", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        cli,
        "run_benchmark",
        lambda *args, **kwargs: (_ for _ in ()).throw(AdapterError("fail")),
    )
    output = tmp_path / "result.json"
    output.write_text('{"old": true}\n', encoding="utf-8")

    assert cli.main(_argv(output)) == 2
    assert output.read_text(encoding="utf-8") == '{"old": true}\n'


def test_unexpected_error_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from callasr import cli

    monkeypatch.setattr(cli, "FasterWhisperAdapter", lambda *args, **kwargs: object())

    def defect(*args: object, **kwargs: object) -> BenchmarkResult:
        raise RuntimeError("developer defect")

    monkeypatch.setattr(cli, "run_benchmark", defect)

    with pytest.raises(RuntimeError, match="developer defect"):
        cli.main(_argv(tmp_path / "result.json"))


def test_artifact_writer_uses_same_directory_atomic_replace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from callasr import cli

    output = tmp_path / "deep" / "result.json"
    real_replace = cli.os.replace
    seen: list[tuple[Path, Path]] = []

    def replace(src: str | Path, dst: str | Path) -> None:
        source = Path(src)
        destination = Path(dst)
        seen.append((source, destination))
        assert source.exists()
        assert source.parent == output.parent
        assert not destination.exists()
        real_replace(src, dst)

    monkeypatch.setattr(cli.os, "replace", replace)
    cli.write_result_artifact(_result(), output)

    assert seen and seen[0][1] == output
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1
