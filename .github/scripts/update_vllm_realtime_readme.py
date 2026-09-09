from pathlib import Path

path = Path("README.md")
text = path.read_text(encoding="utf-8")

old = '''For an OpenAI-compatible HTTP endpoint, install the lightweight HTTP extra:

```bash
uv sync --extra openai-compatible
```

The examples below use `uv run callasr` so activating the virtual environment is'''
new = '''For an OpenAI-compatible HTTP endpoint, install the lightweight HTTP extra:

```bash
uv sync --extra openai-compatible
```

For the vLLM Realtime WebSocket streaming adapter, install its lightweight
transport extra:

```bash
uv sync --extra vllm-realtime
```

The examples below use `uv run callasr` so activating the virtual environment is'''
if text.count(old) != 1:
    raise SystemExit("install anchor not found")
text = text.replace(old, new, 1)

old = '''This foundation does **not** yet define a streaming CLI, serialized streaming
artifact schema, or provider-specific streaming adapter. Those should be added
only after a real adapter validates the Python contract in practice.

## Concurrent benchmark foundation'''
new = '''The streaming foundation itself remains separate from the batch CLI/artifact
contract. A real vLLM Realtime adapter now validates this Python protocol, while a
streaming CLI and serialized streaming artifact remain later work.

### vLLM Realtime adapter

`VLLMRealtimeAdapter` connects to vLLM's WebSocket Realtime transcription API.
Give it the same HTTP(S) API root convention used elsewhere in this project; the
adapter normalizes it to the corresponding WS(S) `/realtime` endpoint.

```python
import os

from callasr import VLLMRealtimeAdapter, run_streaming_benchmark
from callasr.io import load_wav

source = load_wav("audio/example-16k.wav")
adapter = VLLMRealtimeAdapter(
    "mistralai/Voxtral-Mini-4B-Realtime-2602",
    base_url="http://localhost:8000/v1",
    api_key=os.environ.get("CALLASR_VLLM_REALTIME_API_KEY"),
)
result = run_streaming_benchmark(source, adapter, frame_duration_ms=20)
print(result.final_text)
print(result.finalization_latency_seconds)
```

The current vLLM Realtime transport requires mono 16 kHz audio. The adapter
rejects any other sample rate instead of resampling individual streaming frames.
Frames are clipped to `[-1, 1]` only at the transport boundary, encoded as
little-endian PCM16, base64-encoded, and sent as `input_audio_buffer.append`
events. Incremental `transcription.delta` text is accumulated into full current
partial states before it enters call-asr-bench's partial-stability metric;
`transcription.done` is authoritative for the final transcript.

The current vLLM Realtime session contract does not expose an explicit language
field. Passing a non-`None` language therefore fails before opening the WebSocket
instead of silently dropping the request.

This first real adapter intentionally follows vLLM's file-style sequence: all
audio frames are submitted before transcription events are consumed. Its
**finalization latency** remains useful, but its time-to-first-partial includes the
full audio upload and must not be interpreted as paced, full-duplex microphone
TTFT. A future full-duplex adapter contract should make that timing boundary
explicit rather than changing this result silently.

For an opt-in live smoke test against a running vLLM server, provide a real mono
16 kHz PCM WAV and run with the transport extra installed:

```bash
CALLASR_VLLM_REALTIME_URL=http://localhost:8000/v1 \\
CALLASR_VLLM_REALTIME_MODEL=mistralai/Voxtral-Mini-4B-Realtime-2602 \\
CALLASR_VLLM_REALTIME_WAV=/absolute/path/to/example-16k.wav \\
uv run --extra vllm-realtime pytest -q tests/test_vllm_realtime_live.py
```

Set `CALLASR_VLLM_REALTIME_API_KEY` as well when the server requires Bearer
authentication. The live test is skipped by default, so normal CI performs no
network access and downloads no model.

## Concurrent benchmark foundation'''
if text.count(old) != 1:
    raise SystemExit("streaming provider anchor not found")
text = text.replace(old, new, 1)

old = '''- a real provider-backed streaming adapter and streaming CLI/artifact contract;
- impairment pipelines inside concurrent load runs;'''
new = '''- a streaming CLI/artifact contract and paced full-duplex microphone timing;
- impairment pipelines inside concurrent load runs;'''
if text.count(old) != 1:
    raise SystemExit("limitations anchor not found")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
