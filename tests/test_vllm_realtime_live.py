from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from callasr.adapters.vllm_realtime import VLLMRealtimeAdapter
from callasr.adapters.vllm_realtime_paced import VLLMPacedRealtimeAdapter
from callasr.io import load_wav
from callasr.paced_streaming import run_paced_streaming_benchmark
from callasr.streaming import run_streaming_benchmark


def test_vllm_realtime_live_smoke() -> None:
    base_url = os.environ.get("CALLASR_VLLM_REALTIME_URL")
    model = os.environ.get("CALLASR_VLLM_REALTIME_MODEL")
    wav_path = os.environ.get("CALLASR_VLLM_REALTIME_WAV")
    if base_url is None or model is None or wav_path is None:
        pytest.skip(
            "set CALLASR_VLLM_REALTIME_URL, CALLASR_VLLM_REALTIME_MODEL, and "
            "CALLASR_VLLM_REALTIME_WAV to run the live vLLM Realtime smoke test"
        )

    audio = load_wav(Path(wav_path))
    if audio.sample_rate != 16_000:
        pytest.fail("CALLASR_VLLM_REALTIME_WAV must be mono PCM WAV at 16000 Hz")

    adapter = VLLMRealtimeAdapter(
        model,
        base_url=base_url,
        api_key=os.environ.get("CALLASR_VLLM_REALTIME_API_KEY"),
    )
    result = run_streaming_benchmark(audio, adapter, frame_duration_ms=20)

    assert result.frame_count > 0
    assert result.total_streaming_wall_seconds >= 0.0
    assert result.finalization_latency_seconds >= 0.0
    assert isinstance(result.final_text, str)


def test_vllm_realtime_paced_live_smoke() -> None:
    if os.environ.get("CALLASR_VLLM_REALTIME_PACED") != "1":
        pytest.skip("set CALLASR_VLLM_REALTIME_PACED=1 to run the paced live smoke test")

    base_url = os.environ.get("CALLASR_VLLM_REALTIME_URL")
    model = os.environ.get("CALLASR_VLLM_REALTIME_MODEL")
    wav_path = os.environ.get("CALLASR_VLLM_REALTIME_WAV")
    if base_url is None or model is None or wav_path is None:
        pytest.skip(
            "set CALLASR_VLLM_REALTIME_URL, CALLASR_VLLM_REALTIME_MODEL, and "
            "CALLASR_VLLM_REALTIME_WAV to run the paced live vLLM Realtime smoke test"
        )

    audio = load_wav(Path(wav_path))
    if audio.sample_rate != 16_000:
        pytest.fail("CALLASR_VLLM_REALTIME_WAV must be mono PCM WAV at 16000 Hz")

    adapter = VLLMPacedRealtimeAdapter(
        model,
        base_url=base_url,
        api_key=os.environ.get("CALLASR_VLLM_REALTIME_API_KEY"),
    )
    result = asyncio.run(run_paced_streaming_benchmark(audio, adapter, frame_duration_ms=20))

    assert result.timing_mode == "paced"
    assert result.frame_count > 0
    assert result.audio_submission_wall_seconds >= 0.0
    assert result.finalization_latency_seconds >= 0.0
    assert isinstance(result.final_text, str)


def test_vllm_realtime_paced_cli_live_smoke(tmp_path: Path) -> None:
    if os.environ.get("CALLASR_VLLM_REALTIME_PACED") != "1":
        pytest.skip("set CALLASR_VLLM_REALTIME_PACED=1 to run the paced CLI smoke test")

    base_url = os.environ.get("CALLASR_VLLM_REALTIME_URL")
    model = os.environ.get("CALLASR_VLLM_REALTIME_MODEL")
    wav_path = os.environ.get("CALLASR_VLLM_REALTIME_WAV")
    if base_url is None or model is None or wav_path is None:
        pytest.skip(
            "set CALLASR_VLLM_REALTIME_URL, CALLASR_VLLM_REALTIME_MODEL, and "
            "CALLASR_VLLM_REALTIME_WAV to run the paced CLI smoke test"
        )

    audio_path = Path(wav_path).expanduser().resolve()
    audio = load_wav(audio_path)
    if audio.sample_rate != 16_000:
        pytest.fail("CALLASR_VLLM_REALTIME_WAV must be mono PCM WAV at 16000 Hz")

    manifest = tmp_path / "live.jsonl"
    manifest.write_text(
        json.dumps({"id": "live-smoke", "audio": str(audio_path), "reference": ""}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "live-result.json"

    from callasr.cli import main

    status = main(
        [
            "streaming",
            str(manifest),
            "--adapter",
            "vllm-realtime",
            "--model",
            model,
            "--base-url",
            base_url,
            "--timing-mode",
            "paced",
            "--frame-duration-ms",
            "20",
            "--realtime-factor",
            "1.0",
            "--language-mode",
            "autodetect",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["timing_mode"] == "paced"
    assert payload["streaming"]["realtime_factor"] == 1.0
    assert payload["summary"]["completed_items"] == 1
    assert payload["items"][0]["frame_count"] > 0
