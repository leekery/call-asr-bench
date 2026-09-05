"""Command-line interface for call-asr-bench."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from math import isfinite
from pathlib import Path

from callasr.adapters.base import AdapterError, ASRAdapter
from callasr.adapters.faster_whisper import FasterWhisperAdapter
from callasr.adapters.openai_compatible import OpenAICompatibleAdapter
from callasr.benchmark import BenchmarkResult, result_to_dict, run_benchmark
from callasr.dataset import DatasetError
from callasr.io import AudioError


class ConfigurationError(ValueError):
    """A user-facing benchmark configuration error."""


class ArtifactError(OSError):
    """A user-facing result-artifact write error."""


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line parser."""

    parser = argparse.ArgumentParser(prog="callasr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a sequential ASR benchmark")
    run.add_argument("manifest")
    run.add_argument(
        "--adapter",
        choices=("faster-whisper", "openai-compatible"),
        required=True,
    )
    run.add_argument("--model", required=True)
    run.add_argument("--codec", choices=("none", "pcmu", "pcma"), default="none")
    run.add_argument("--packet-loss-rate", type=_probability, default=0.0)
    run.add_argument("--frame-duration-ms", type=_positive_int, default=20)
    run.add_argument("--snr-db", type=_finite_float)
    run.add_argument("--jitter-std-ms", type=_non_negative_float)
    run.add_argument("--playout-buffer-ms", type=_non_negative_float)
    run.add_argument("--seed", type=_non_negative_int, default=0)
    run.add_argument("--output", required=True)
    run.add_argument("--device", default="auto")
    run.add_argument("--compute-type", default="default")
    run.add_argument("--base-url")
    run.add_argument("--api-key")
    run.add_argument("--timeout-seconds", type=_positive_float, default=60.0)
    return parser


def _validate_configuration(args: argparse.Namespace) -> None:
    if args.codec == "none" and args.packet_loss_rate != 0.0:
        raise ConfigurationError("packet-loss-rate must be zero when codec is none")
    if (args.jitter_std_ms is None) != (args.playout_buffer_ms is None):
        raise ConfigurationError("jitter-std-ms and playout-buffer-ms must be provided together")
    if args.jitter_std_ms is not None and args.codec == "none":
        raise ConfigurationError("jitter requires codec pcmu or pcma")

    if args.adapter == "openai-compatible":
        if args.base_url is None:
            raise ConfigurationError("base-url is required with adapter openai-compatible")
    elif (
        args.base_url is not None
        or args.api_key is not None
        or args.timeout_seconds != 60.0
    ):
        raise ConfigurationError(
            "base-url, api-key, and non-default timeout-seconds are only valid with adapter "
            "openai-compatible"
        )


def _build_adapter(args: argparse.Namespace) -> ASRAdapter:
    if args.adapter == "faster-whisper":
        return FasterWhisperAdapter(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
        )
    if args.adapter == "openai-compatible":
        api_key = args.api_key
        if api_key is None:
            api_key = os.environ.get("CALLASR_API_KEY")
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        return OpenAICompatibleAdapter(
            args.model,
            base_url=args.base_url,
            api_key=api_key,
            timeout_seconds=args.timeout_seconds,
        )
    raise ConfigurationError(f"unsupported adapter: {args.adapter}")


def write_result_artifact(result: BenchmarkResult, path: str | Path) -> None:
    """Write a complete benchmark result with same-directory atomic replacement."""

    output_path = Path(path).expanduser()
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(result_to_dict(result), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ArtifactError(f"cannot write result artifact {output_path}: {exc}") from exc
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _run(args: argparse.Namespace) -> int:
    _validate_configuration(args)
    adapter = _build_adapter(args)
    result = run_benchmark(
        args.manifest,
        adapter,
        codec=args.codec,
        packet_loss_rate=args.packet_loss_rate,
        frame_duration_ms=args.frame_duration_ms,
        snr_db=args.snr_db,
        jitter_std_ms=args.jitter_std_ms,
        playout_buffer_ms=args.playout_buffer_ms,
        seed=args.seed,
    )
    write_result_artifact(result, args.output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit status."""

    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
    except (DatasetError, AudioError, AdapterError, ConfigurationError, ArtifactError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")
