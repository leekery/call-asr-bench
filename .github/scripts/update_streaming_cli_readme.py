from pathlib import Path

path = Path("README.md")
text = path.read_text(encoding="utf-8")

old = '''The streaming foundation remains separate from the batch CLI/artifact contract.
A real vLLM Realtime adapter validates the single-audio protocol, and current
`main` now also provides a generic dataset-level streaming runner plus a separate
versioned streaming artifact. A user-facing streaming CLI remains later work.
'''
new = '''The streaming foundation remains separate from the batch CLI/artifact contract.
A real vLLM Realtime adapter validates the single-audio protocol, and current
`main` provides a generic dataset-level streaming runner, a separate versioned
streaming artifact, and a `callasr streaming` CLI for the existing file-upload
vLLM timing mode.
'''
if text.count(old) != 1:
    raise SystemExit("streaming status paragraph not found")
text = text.replace(old, new, 1)

anchor = "### vLLM Realtime adapter\n"
if text.count(anchor) != 1:
    raise SystemExit("vLLM section anchor not found")
section = '''### Run a vLLM streaming dataset benchmark

Install the WebSocket transport extra and run the normal JSONL manifest through
`callasr streaming`:

```bash
CALLASR_VLLM_REALTIME_API_KEY=your-secret-key \\
uv run --extra vllm-realtime callasr streaming dataset/dataset.jsonl \\
  --adapter vllm-realtime \\
  --model mistralai/Voxtral-Mini-4B-Realtime-2602 \\
  --base-url http://localhost:8000/v1 \\
  --frame-duration-ms 20 \\
  --language-mode autodetect \\
  --output runs/voxtral-streaming.json
```

The command writes exactly the existing streaming artifact contract:

```text
kind: streaming
schema_version: 1
timing_mode: file_upload
```

No API key is serialized. `--api-key` takes precedence over
`CALLASR_VLLM_REALTIME_API_KEY`; omit both when the server doesn't require Bearer
authentication. `--timeout-seconds` defaults to 60 seconds.

`--language-mode` defaults to `manifest`. Because the current vLLM Realtime
session protocol doesn't expose an explicit language setting, a manifest that
contains any `language` field is rejected **before adapter construction or a
WebSocket connection** unless the user explicitly selects
`--language-mode autodetect`. This makes dropping RU/EN language hints an explicit
choice rather than a silent behavior change. Manifests without language tags may
use the default `manifest` mode because every item still passes `None`.

The CLI reuses `run_streaming_dataset_benchmark` for all aggregation and
`build_streaming_artifact` for serialization; it does not define a second timing
or metric implementation. Output writing uses the same same-directory atomic
replacement contract as batch and concurrent artifacts, so a failed run or failed
replace leaves an existing completed output untouched.

The resulting TTFT is still **file-upload TTFT**, not paced microphone TTFT. The
current vLLM adapter submits all supplied frames before consuming transcript
events. `timing_mode=file_upload` remains mandatory in schema 1 precisely so a
future full-duplex mode cannot silently redefine those numbers.

'''
text = text.replace(anchor, section + anchor, 1)

old = "- a streaming CLI and paced full-duplex microphone timing;\n"
new = "- paced full-duplex microphone timing;\n"
if text.count(old) != 1:
    raise SystemExit("streaming limitation line not found")
text = text.replace(old, new, 1)

old = '''1. Expose the streaming dataset/artifact contract through a vLLM Realtime CLI
   without changing its `file_upload` timing semantics.
2. Define and validate a distinct paced/full-duplex streaming timing contract.
3. More local ASR adapters, including GigaAM Multilingual when its packaging path
   is stable.
4. Add impairment composition to concurrent load runs only after the clean-load
   artifact contract has practical usage.
5. Add reporting/leaderboard layers only when datasets and artifact identities
   are comparable by construction.
'''
new = '''1. Define and validate a distinct paced/full-duplex streaming timing contract.
2. More local ASR adapters, including GigaAM Multilingual when its packaging path
   is stable.
3. Add impairment composition to concurrent load runs only after the clean-load
   artifact contract has practical usage.
4. Add additional critical-entity slices such as names and addresses.
5. Add reporting/leaderboard layers only when datasets and artifact identities
   are comparable by construction.
'''
if text.count(old) != 1:
    raise SystemExit("roadmap block not found")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
