from __future__ import annotations

import os
from pathlib import Path

import pytest

from callasr.adapters.vllm_realtime import VLLMRealtimeAdapter
from callasr.io import load_wav
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
