from __future__ import annotations

import json
from pathlib import Path

import pytest


def _api():
    try:
        from callasr.report import (
            ComparisonError,
            compare_result_artifacts,
            load_comparison_rows,
            render_comparison_markdown,
        )
    except ImportError as exc:
        pytest.fail(f"comparison report implementation is missing: {exc}")
    return (
        ComparisonError,
        compare_result_artifacts,
        load_comparison_rows,
        render_comparison_markdown,
    )


def _artifact(
    schema: int,
    *,
    adapter: str = "faster-whisper",
    model: str = "large-v3",
    codec: str = "none",
    packet_loss_rate: float = 0.0,
    snr_db: float | None = None,
    jitter_std_ms: float | None = None,
    playout_buffer_ms: float | None = None,
    wer: float = 0.1,
    cer: float = 0.05,
    rtf: float = 0.2,
    speed_factor: float | None = 5.0,
    numeric_entity_accuracy: float | None = None,
    item_count: int = 2,
) -> dict[str, object]:
    channel: dict[str, object] = {
        "codec": codec,
        "packet_loss_rate": packet_loss_rate,
        "frame_duration_ms": 20,
        "seed": 42,
    }
    if schema >= 2:
        channel["additive_noise_snr_db"] = snr_db
    if schema >= 3:
        channel["jitter_std_ms"] = jitter_std_ms
        channel["playout_buffer_ms"] = playout_buffer_ms

    summary: dict[str, object] = {
        "total_audio_seconds": 2.0,
        "adapter_seconds": 0.4,
        "wer": wer,
        "cer": cer,
        "rtf": rtf,
        "speed_factor": speed_factor,
    }
    if schema >= 4:
        summary.update(
            numeric_entity_matches=1,
            numeric_entity_reference_count=2,
            numeric_entity_accuracy=numeric_entity_accuracy,
        )

    return {
        "schema_version": schema,
        "created_at": "2026-09-05T20:00:00+00:00",
        "dataset": {"path": "/tmp/dataset.jsonl", "item_count": item_count},
        "adapter": {
            "name": adapter,
            "model": model,
            "device": "cpu",
            "compute_type": "float32",
            "decoding_options": {},
        },
        "channel": channel,
        "summary": summary,
        "items": [],
    }


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_mixed_known_schemas_render_common_metrics_and_optional_fields(tmp_path: Path) -> None:
    _, compare_result_artifacts, _, _ = _api()
    old = _write(
        tmp_path / "v1.json",
        _artifact(1, model="old-model", codec="pcmu", packet_loss_rate=0.05),
    )
    current = _write(
        tmp_path / "v4.json",
        _artifact(
            4,
            adapter="openai-compatible",
            model="served-model",
            codec="pcma",
            packet_loss_rate=0.1,
            snr_db=15.0,
            jitter_std_ms=8.0,
            playout_buffer_ms=20.0,
            wer=0.08,
            cer=0.03,
            rtf=0.25,
            speed_factor=4.0,
            numeric_entity_accuracy=0.75,
        ),
    )

    report = compare_result_artifacts([old, current])

    assert report == "\n".join(
        [
            "| Artifact | Schema | Adapter | Model | Codec | Loss | SNR dB | Jitter ms | WER | CER | RTF | Speed | Numeric entity | Items |",
            "| --- | ---: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| v1.json | 1 | faster-whisper | old-model | pcmu | 0.05 | — | — | 0.1 | 0.05 | 0.2 | 5 | — | 2 |",
            "| v4.json | 4 | openai-compatible | served-model | pcma | 0.1 | 15 | 8/20 | 0.08 | 0.03 | 0.25 | 4 | 0.75 | 2 |",
        ]
    )


def test_input_order_is_preserved(tmp_path: Path) -> None:
    _, _, load_comparison_rows, _ = _api()
    second = _write(tmp_path / "second.json", _artifact(4, model="second"))
    first = _write(tmp_path / "first.json", _artifact(4, model="first"))

    rows = load_comparison_rows([second, first])

    assert [row.artifact for row in rows] == ["second.json", "first.json"]
    assert [row.model for row in rows] == ["second", "first"]


def test_optional_numeric_zeroes_are_not_rendered_as_missing(tmp_path: Path) -> None:
    _, compare_result_artifacts, _, _ = _api()
    path = _write(
        tmp_path / "zeroes.json",
        _artifact(
            4,
            codec="pcmu",
            packet_loss_rate=0.0,
            snr_db=0.0,
            jitter_std_ms=0.0,
            playout_buffer_ms=0.0,
            wer=0.0,
            cer=0.0,
            rtf=0.0,
            speed_factor=None,
            numeric_entity_accuracy=0.0,
        ),
    )

    report = compare_result_artifacts([path])
    row = report.splitlines()[2]

    assert "| 0 | 0 | 0/0 | 0 | 0 | 0 | — | 0 |" in row


def test_markdown_special_content_is_escaped_deterministically(tmp_path: Path) -> None:
    _, compare_result_artifacts, _, _ = _api()
    path = _write(
        tmp_path / "pipe|artifact.json",
        _artifact(4, adapter="remote|adapter", model="line one\nline|two"),
    )

    report = compare_result_artifacts([path])
    row = report.splitlines()[2]

    assert "pipe\\|artifact.json" in row
    assert "remote\\|adapter" in row
    assert "line one<br>line\\|two" in row
    assert len(report.splitlines()) == 3


@pytest.mark.parametrize("schema", [0, 5, -1])
def test_unknown_schema_is_rejected_with_artifact_path(tmp_path: Path, schema: int) -> None:
    ComparisonError, compare_result_artifacts, _, _ = _api()
    path = _write(tmp_path / "unknown.json", _artifact(schema))

    with pytest.raises(ComparisonError, match=r"unknown\.json.*schema_version"):
        compare_result_artifacts([path])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("adapter"), "adapter"),
        (lambda payload: payload["summary"].__setitem__("wer", "bad"), "wer"),
        (lambda payload: payload["dataset"].__setitem__("item_count", True), "item_count"),
        (lambda payload: payload["channel"].__setitem__("codec", "mp3"), "codec"),
        (lambda payload: payload["summary"].__setitem__("rtf", float("nan")), "rtf"),
    ],
)
def test_malformed_common_fields_fail_with_path_context(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    ComparisonError, compare_result_artifacts, _, _ = _api()
    payload = _artifact(4)
    mutation(payload)
    path = _write(tmp_path / "malformed.json", payload)

    with pytest.raises(ComparisonError, match=rf"malformed\.json.*{message}"):
        compare_result_artifacts([path])


def test_incomplete_jitter_metadata_is_rejected(tmp_path: Path) -> None:
    ComparisonError, compare_result_artifacts, _, _ = _api()
    payload = _artifact(3, jitter_std_ms=8.0, playout_buffer_ms=20.0)
    payload["channel"].pop("playout_buffer_ms")
    path = _write(tmp_path / "jitter.json", payload)

    with pytest.raises(ComparisonError, match=r"jitter\.json.*jitter"):
        compare_result_artifacts([path])


def test_invalid_json_and_missing_file_are_path_qualified(tmp_path: Path) -> None:
    ComparisonError, compare_result_artifacts, _, _ = _api()
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(ComparisonError, match=r"broken\.json.*JSON"):
        compare_result_artifacts([broken])

    missing = tmp_path / "missing.json"
    with pytest.raises(ComparisonError, match=r"missing\.json.*read"):
        compare_result_artifacts([missing])


def test_render_api_is_pure_and_reuses_loaded_rows(tmp_path: Path) -> None:
    _, _, load_comparison_rows, render_comparison_markdown = _api()
    path = _write(tmp_path / "one.json", _artifact(2, snr_db=10.0))

    rows = load_comparison_rows([path])
    first = render_comparison_markdown(rows)
    second = render_comparison_markdown(rows)

    assert first == second
    assert "| one.json | 2 |" in first


def test_cli_compare_prints_report_without_inference_or_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from callasr import cli

    path = _write(tmp_path / "result.json", _artifact(4))

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("comparison must not invoke benchmark/model paths")

    monkeypatch.setattr(cli, "_build_adapter", forbidden)
    monkeypatch.setattr(cli, "run_benchmark", forbidden)

    status = cli.main(["compare", str(path)])

    assert status == 0
    stdout = capsys.readouterr().out
    assert stdout.startswith("| Artifact | Schema |")
    assert "result.json" in stdout


def test_cli_compare_maps_comparison_error_to_exit_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from callasr import cli

    path = _write(tmp_path / "unknown.json", _artifact(99))

    status = cli.main(["compare", str(path)])

    assert status == 2
    assert "unknown.json" in capsys.readouterr().err
