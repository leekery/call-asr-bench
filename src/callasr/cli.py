"""Command-line interface for call-asr-bench."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from math import isfinite
from pathlib import Path

from callasr.adapters.base import AdapterError, ASRAdapter
from callasr.adapters.faster_whisper import FasterWhisperAdapter
from callasr.adapters.openai_compatible import OpenAICompatibleAdapter
from callasr.adapters.vllm_realtime import VLLMRealtimeAdapter
from callasr.benchmark import BenchmarkResult, result_to_dict, run_benchmark
from callasr.concurrent import ConcurrentBenchmarkError, run_concurrent_benchmark
from callasr.concurrent_artifact import (
    ConcurrentAdapterInfo,
    ConcurrentArtifact,
    build_concurrent_artifact,
    concurrent_artifact_to_dict,
)
from callasr.dataset import DatasetError, dataset_fingerprint, load_dataset_manifest
from callasr.io import AudioError
from callasr.report import ComparisonError, compare_result_artifacts
from callasr.streaming import StreamingError
from callasr.streaming_artifact import (
    StreamingAdapterInfo,
    StreamingArtifact,
    build_streaming_artifact,
    streaming_artifact_to_dict,
)
from callasr.streaming_dataset import StreamingDatasetError, run_streaming_dataset_benchmark


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


def _add_adapter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--adapter",
        choices=("faster-whisper", "openai-compatible"),
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout-seconds", type=_positive_float, default=60.0)


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line parser."""

    parser = argparse.ArgumentParser(prog="callasr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a sequential ASR benchmark")
    run.add_argument("manifest")
    _add_adapter_arguments(run)
    run.add_argument("--codec", choices=("none", "pcmu", "pcma"), default="none")
    run.add_argument("--packet-loss-rate", type=_probability, default=0.0)
    run.add_argument("--frame-duration-ms", type=_positive_int, default=20)
    run.add_argument("--gain-db", type=_finite_float, default=0.0)
    run.add_argument("--clip-threshold", type=_positive_float)
    run.add_argument("--snr-db", type=_finite_float)
    run.add_argument("--jitter-std-ms", type=_non_negative_float)
    run.add_argument("--playout-buffer-ms", type=_non_negative_float)
    run.add_argument("--seed", type=_non_negative_int, default=0)
    run.add_argument("--output", required=True)

    concurrent = subparsers.add_parser("concurrent", help="run a concurrent ASR load benchmark")
    concurrent.add_argument("manifest")
    _add_adapter_arguments(concurrent)
    concurrent.add_argument("--concurrency", type=_positive_int, required=True)
    concurrent.add_argument("--output", required=True)

    streaming = subparsers.add_parser("streaming", help="run a streaming ASR benchmark")
    streaming.add_argument("manifest")
    streaming.add_argument("--adapter", choices=("vllm-realtime",), required=True)
    streaming.add_argument("--model", required=True)
    streaming.add_argument("--base-url", required=True)
    streaming.add_argument("--api-key")
    streaming.add_argument("--timeout-seconds", type=_positive_float, default=60.0)
    streaming.add_argument("--frame-duration-ms", type=_positive_int, default=20)
    streaming.add_argument(
        "--language-mode",
        choices=("manifest", "autodetect"),
        default="manifest",
    )
    streaming.add_argument("--output", required=True)

    compare = subparsers.add_parser("compare", help="compare saved benchmark artifacts")
    compare.add_argument("results", nargs="+")
    return parser


def _validate_adapter_configuration(args: argparse.Namespace) -> None:
    if args.adapter == "openai-compatible":
        if args.base_url is None:
            raise ConfigurationError("base-url is required with adapter openai-compatible")
    elif args.base_url is not None or args.api_key is not None or args.timeout_seconds != 60.0:
        raise ConfigurationError(
            "base-url, api-key, and non-default timeout-seconds are only valid with adapter "
            "openai-compatible"
        )


def _validate_configuration(args: argparse.Namespace) -> None:
    if args.codec == "none" and args.packet_loss_rate != 0.0:
        raise ConfigurationError("packet-loss-rate must be zero when codec is none")
    if (args.jitter_std_ms is None) != (args.playout_buffer_ms is None):
        raise ConfigurationError("jitter-std-ms and playout-buffer-ms must be provided together")
    if args.jitter_std_ms is not None and args.codec == "none":
        raise ConfigurationError("jitter requires codec pcmu or pcma")
    _validate_adapter_configuration(args)


def _resolved_api_key(args: argparse.Namespace) -> str | None:
    api_key = args.api_key
    if api_key is None:
        api_key = os.environ.get("CALLASR_API_KEY")
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    return api_key


def _resolved_vllm_api_key(args: argparse.Namespace) -> str | None:
    if args.api_key is not None:
        return args.api_key
    return os.environ.get("CALLASR_VLLM_REALTIME_API_KEY")


def _build_adapter(args: argparse.Namespace) -> ASRAdapter:
    if args.adapter == "faster-whisper":
        return FasterWhisperAdapter(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
        )
    if args.adapter == "openai-compatible":
        return OpenAICompatibleAdapter(
            args.model,
            base_url=args.base_url,
            api_key=_resolved_api_key(args),
            timeout_seconds=args.timeout_seconds,
        )
    raise ConfigurationError(f"unsupported adapter: {args.adapter}")


def _concurrent_adapter_factory(args: argparse.Namespace) -> Callable[[], ASRAdapter]:
    if args.adapter == "faster-whisper":
        return lambda: FasterWhisperAdapter(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
        )
    if args.adapter == "openai-compatible":
        api_key = _resolved_api_key(args)
        base_url = args.base_url.rstrip("/")
        return lambda: OpenAICompatibleAdapter(
            args.model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=args.timeout_seconds,
        )
    raise ConfigurationError(f"unsupported adapter: {args.adapter}")


def _concurrent_adapter_info(args: argparse.Namespace) -> ConcurrentAdapterInfo:
    if args.adapter == "faster-whisper":
        return ConcurrentAdapterInfo(
            name="faster-whisper",
            model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            options={"beam_size": 5, "temperature": 0.0},
        )
    if args.adapter == "openai-compatible":
        return ConcurrentAdapterInfo(
            name="openai-compatible",
            model=args.model,
            device="remote",
            compute_type="server",
            options={
                "base_url": args.base_url.rstrip("/"),
                "timeout_seconds": args.timeout_seconds,
                "response_format": "json",
                "upload_format": "wav_pcm16",
            },
        )
    raise ConfigurationError(f"unsupported adapter: {args.adapter}")


def _write_json_artifact(payload: dict[str, object], path: str | Path) -> None:
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
            json.dump(payload, handle, ensure_ascii=False, indent=2)
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


def write_result_artifact(result: BenchmarkResult, path: str | Path) -> None:
    """Write a complete benchmark result with same-directory atomic replacement."""

    _write_json_artifact(result_to_dict(result), path)


def write_concurrent_artifact(result: ConcurrentArtifact, path: str | Path) -> None:
    """Write a complete concurrent result with same-directory atomic replacement."""

    _write_json_artifact(concurrent_artifact_to_dict(result), path)


def write_streaming_artifact(result: StreamingArtifact, path: str | Path) -> None:
    """Write a complete streaming result with same-directory atomic replacement."""

    _write_json_artifact(streaming_artifact_to_dict(result), path)


def _run(args: argparse.Namespace) -> int:
    _validate_configuration(args)
    adapter = _build_adapter(args)
    result = run_benchmark(
        args.manifest,
        adapter,
        codec=args.codec,
        packet_loss_rate=args.packet_loss_rate,
        frame_duration_ms=args.frame_duration_ms,
        gain_db=args.gain_db,
        clip_threshold=args.clip_threshold,
        snr_db=args.snr_db,
        jitter_std_ms=args.jitter_std_ms,
        playout_buffer_ms=args.playout_buffer_ms,
        seed=args.seed,
    )
    write_result_artifact(result, args.output)
    return 0


def _concurrent(args: argparse.Namespace) -> int:
    _validate_adapter_configuration(args)
    manifest_path = Path(args.manifest).expanduser().resolve()
    items = load_dataset_manifest(manifest_path)
    fingerprint = dataset_fingerprint(items)
    adapter_info = _concurrent_adapter_info(args)
    adapter_factory = _concurrent_adapter_factory(args)
    result = run_concurrent_benchmark(
        manifest_path,
        adapter_factory,
        concurrency=args.concurrency,
    )
    artifact = build_concurrent_artifact(
        result,
        manifest_path=manifest_path,
        fingerprint=fingerprint,
        adapter=adapter_info,
    )
    write_concurrent_artifact(artifact, args.output)
    return 0


def _streaming(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    items = load_dataset_manifest(manifest_path)
    if args.language_mode == "manifest" and any(item.language is not None for item in items):
        raise ConfigurationError(
            "vLLM Realtime does not support explicit manifest language tags; "
            "use --language-mode autodetect to explicitly ignore them"
        )

    adapter = VLLMRealtimeAdapter(
        args.model,
        base_url=args.base_url,
        api_key=_resolved_vllm_api_key(args),
        timeout_seconds=args.timeout_seconds,
    )
    adapter_info = StreamingAdapterInfo(
        name=adapter.name,
        model=adapter.model,
        device=adapter.device,
        compute_type=adapter.compute_type,
        options=dict(adapter.decoding_options),
    )
    result = run_streaming_dataset_benchmark(
        manifest_path,
        adapter,
        frame_duration_ms=args.frame_duration_ms,
        language_mode=args.language_mode,
    )
    artifact = build_streaming_artifact(result, adapter=adapter_info)
    write_streaming_artifact(artifact, args.output)
    return 0


def _compare(args: argparse.Namespace) -> int:
    print(compare_result_artifacts(args.results))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit status."""

    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "concurrent":
            return _concurrent(args)
        if args.command == "streaming":
            return _streaming(args)
        if args.command == "compare":
            return _compare(args)
    except (
        DatasetError,
        AudioError,
        AdapterError,
        StreamingError,
        StreamingDatasetError,
        ConcurrentBenchmarkError,
        ConfigurationError,
        ArtifactError,
        ComparisonError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")
