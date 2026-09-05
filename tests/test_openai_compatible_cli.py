from __future__ import annotations

from pathlib import Path

import pytest


def _remote_argv(output: Path, *extra: str) -> list[str]:
    return [
        "run",
        "dataset.jsonl",
        "--adapter",
        "openai-compatible",
        "--model",
        "served-asr",
        "--base-url",
        "http://localhost:8000/v1",
        *extra,
        "--output",
        str(output),
    ]


def _stub_run(monkeypatch: pytest.MonkeyPatch, cli) -> None:
    monkeypatch.setattr(cli, "run_benchmark", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "write_result_artifact", lambda result, path: None)


def test_cli_builds_remote_adapter_with_explicit_key_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from callasr import cli

    seen: dict[str, object] = {}

    def remote_adapter(
        model: str,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float,
    ) -> object:
        seen.update(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        return object()

    monkeypatch.setattr(cli, "OpenAICompatibleAdapter", remote_adapter)
    monkeypatch.setenv("CALLASR_API_KEY", "env-callasr")
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai")
    _stub_run(monkeypatch, cli)

    status = cli.main(
        _remote_argv(
            tmp_path / "result.json",
            "--api-key",
            "explicit-key",
            "--timeout-seconds",
            "12.5",
        )
    )

    assert status == 0
    assert seen == {
        "model": "served-asr",
        "base_url": "http://localhost:8000/v1",
        "api_key": "explicit-key",
        "timeout_seconds": 12.5,
    }


@pytest.mark.parametrize(
    ("callasr_key", "openai_key", "expected"),
    [
        ("callasr-key", "openai-key", "callasr-key"),
        (None, "openai-key", "openai-key"),
        (None, None, None),
    ],
)
def test_cli_api_key_environment_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    callasr_key: str | None,
    openai_key: str | None,
    expected: str | None,
) -> None:
    from callasr import cli

    monkeypatch.delenv("CALLASR_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if callasr_key is not None:
        monkeypatch.setenv("CALLASR_API_KEY", callasr_key)
    if openai_key is not None:
        monkeypatch.setenv("OPENAI_API_KEY", openai_key)
    seen: dict[str, object] = {}

    def remote_adapter(
        model: str,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float,
    ) -> object:
        seen["api_key"] = api_key
        return object()

    monkeypatch.setattr(cli, "OpenAICompatibleAdapter", remote_adapter)
    _stub_run(monkeypatch, cli)

    assert cli.main(_remote_argv(tmp_path / "result.json")) == 0
    assert seen["api_key"] == expected


def test_remote_adapter_requires_base_url_before_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from callasr import cli

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("remote adapter must not be constructed")

    monkeypatch.setattr(cli, "OpenAICompatibleAdapter", forbidden)
    _stub_run(monkeypatch, cli)

    status = cli.main(
        [
            "run",
            "dataset.jsonl",
            "--adapter",
            "openai-compatible",
            "--model",
            "served-asr",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert status == 2
    assert "base-url" in capsys.readouterr().err


@pytest.mark.parametrize(
    "remote_args",
    [
        ["--base-url", "http://localhost:8000/v1"],
        ["--api-key", "must-not-be-used"],
    ],
)
def test_faster_whisper_rejects_remote_only_options_before_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    remote_args: list[str],
) -> None:
    from callasr import cli

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("faster-whisper adapter must not be constructed")

    monkeypatch.setattr(cli, "FasterWhisperAdapter", forbidden)
    _stub_run(monkeypatch, cli)
    argv = [
        "run",
        "dataset.jsonl",
        "--adapter",
        "faster-whisper",
        "--model",
        "large-v3",
        *remote_args,
        "--output",
        str(tmp_path / "result.json"),
    ]

    assert cli.main(argv) == 2
    assert "only valid with adapter openai-compatible" in capsys.readouterr().err


def test_existing_faster_whisper_path_ignores_remote_key_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from callasr import cli

    monkeypatch.setenv("CALLASR_API_KEY", "should-not-matter")
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-matter-either")
    seen: dict[str, object] = {}

    def local_adapter(model: str, *, device: str, compute_type: str) -> object:
        seen.update(model=model, device=device, compute_type=compute_type)
        return object()

    monkeypatch.setattr(cli, "FasterWhisperAdapter", local_adapter)
    _stub_run(monkeypatch, cli)

    status = cli.main(
        [
            "run",
            "dataset.jsonl",
            "--adapter",
            "faster-whisper",
            "--model",
            "large-v3",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert status == 0
    assert seen == {"model": "large-v3", "device": "auto", "compute_type": "default"}


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_timeout(value: str, tmp_path: Path) -> None:
    from callasr import cli

    with pytest.raises(SystemExit, match="2"):
        cli.main(
            _remote_argv(
                tmp_path / "result.json",
                "--timeout-seconds",
                value,
            )
        )
