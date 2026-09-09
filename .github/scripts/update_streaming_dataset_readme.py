from pathlib import Path

path = Path("README.md")
text = path.read_text(encoding="utf-8")

old = '''The streaming foundation itself remains separate from the batch CLI/artifact
contract. A real vLLM Realtime adapter now validates this Python protocol, while a
streaming CLI and serialized streaming artifact remain later work.
'''
new = '''The streaming foundation remains separate from the batch CLI/artifact contract.
A real vLLM Realtime adapter validates the single-audio protocol, and current
`main` now also provides a generic dataset-level streaming runner plus a separate
versioned streaming artifact. A user-facing streaming CLI remains later work.
'''
if text.count(old) != 1:
    raise SystemExit("streaming foundation status paragraph not found")
text = text.replace(old, new, 1)

anchor = "### vLLM Realtime adapter\n"
if text.count(anchor) != 1:
    raise SystemExit("vLLM section anchor not found")
section = '''### Streaming datasets and artifacts

`run_streaming_dataset_benchmark` applies the existing single-audio streaming
contract sequentially to the normal JSONL/WAV manifest. It preserves manifest
order, records the existing dataset fingerprint, and fails on the first dataset,
audio, or streaming error rather than returning a partial-success result.

Language handling is explicit:

- `language_mode="manifest"` passes each manifest item's optional language to the
  streaming adapter;
- `language_mode="autodetect"` passes `None` for every item and records that
  choice in the result.

This distinction matters for streaming protocols such as the current vLLM
Realtime API, which doesn't expose an explicit language field. Callers must opt
into autodetection instead of silently dropping dataset language metadata.

Dataset-level streaming results use a third, separate artifact family:

```text
kind: streaming
schema_version: 1
timing_mode: file_upload
```

`timing_mode` is part of the public contract. Schema 1 means the existing
file-upload timing semantics: the adapter may submit the complete audio stream
before transcript events are observed. A future paced/full-duplex benchmark must
use a distinct timing mode rather than silently changing what TTFT means.

Each streaming item stores final text, frame count, partial count, nullable TTFT,
finalization latency, total observed streaming wall time, nullable partial
stability, and the complete observed update trace with relative timestamps.
Aggregate metrics include nearest-rank p50/p95 latency values, maxima, contributor
counts for nullable TTFT/stability metrics, and the mean of only non-null
stability values.

`callasr compare` remains a batch-accuracy tool and explicitly rejects both
`kind=concurrent` and `kind=streaming` artifacts instead of mixing incompatible
metric families.

'''
text = text.replace(anchor, section + anchor, 1)

old = "- a streaming CLI/artifact contract and paced full-duplex microphone timing;\n"
new = "- a streaming CLI and paced full-duplex microphone timing;\n"
if text.count(old) != 1:
    raise SystemExit("streaming limitation line not found")
text = text.replace(old, new, 1)

old = '''1. Validate the streaming contract against a real self-hosted/production ASR
   implementation before defining streaming serialization.
2. More local ASR adapters, including GigaAM Multilingual when its packaging path
   is stable.
3. Add impairment composition to concurrent load runs only after the clean-load
   artifact contract has practical usage.
4. Additional critical-entity slices such as names and addresses after the
   numeric contract is stable.
5. Add reporting/leaderboard layers only when datasets and artifact identities
   are comparable by construction.
'''
new = '''1. Expose the streaming dataset/artifact contract through a vLLM Realtime CLI
   without changing its `file_upload` timing semantics.
2. Define and validate a distinct paced/full-duplex streaming timing contract.
3. More local ASR adapters, including GigaAM Multilingual when its packaging path
   is stable.
4. Add impairment composition to concurrent load runs only after the clean-load
   artifact contract has practical usage.
5. Add reporting/leaderboard layers only when datasets and artifact identities
   are comparable by construction.
'''
if text.count(old) != 1:
    raise SystemExit("roadmap block not found")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
