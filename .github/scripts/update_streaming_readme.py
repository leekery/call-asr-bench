from pathlib import Path

path = Path("README.md")
text = path.read_text(encoding="utf-8")
anchor = "## Current development limitations\n"
if text.count(anchor) != 1:
    raise SystemExit("unexpected README streaming anchor")
section = """## Streaming benchmark foundation

Current `main` includes a Python-only streaming foundation in `callasr.streaming`.
It is intentionally separate from the batch `ASRAdapter`, `callasr run`, and the
schema-v6 batch artifact contract.

`StreamingASRAdapter` pulls deterministic `AudioBuffer` frames and yields
`StreamingUpdate` values. `run_streaming_benchmark` can then measure, with an
injectable monotonic clock:

- time to the first non-empty non-final partial transcript;
- finalization latency from submission of the last audio frame to the final update;
- total observed streaming wall time, including adapter cleanup after the final update;
- normalized token-level partial stability across transcript revisions.

`frame_audio` rejects frame durations that do not map to an exact integer sample
count instead of hiding cumulative rounding drift. Streaming contract violations,
such as consuming only part of the audio or yielding updates after the final
update, raise `StreamingError`.

Partial stability normalizes transcript text, collapses consecutive duplicate
partials, appends the final transcript, and averages adjacent token-level
similarities:

```text
similarity(a, b) = 1 - levenshtein(a, b) / max(len(a), len(b), 1)
```

If no non-empty partial is observed, time-to-first-partial and stability are
`None` where applicable rather than inventing a perfect score.

This foundation does **not** yet define a streaming CLI, serialized streaming
artifact schema, or provider-specific streaming adapter. Those should be added
only after a real adapter validates the Python contract in practice.

"""
path.write_text(text.replace(anchor, section + anchor, 1), encoding="utf-8")
