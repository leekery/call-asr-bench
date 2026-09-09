from pathlib import Path

path = Path("README.md")
text = path.read_text(encoding="utf-8")

replacements = [
    (
        "- the `callasr run` CLI;\n",
        "- the `callasr run` CLI;\n- the `callasr concurrent` load-test CLI with worker-local adapter instances;\n",
    ),
    (
        "Published `v0.2.0` artifacts remain schema version 1 and published `v0.3.0`\nartifacts remain schema version 4. Current `main` writes schema version 6; published\ntags and their artifacts are not rewritten.",
        "Published `v0.2.0` artifacts remain schema version 1, published `v0.3.0`\nartifacts remain schema version 4, and published `v0.4.0` batch artifacts remain\nschema version 6. Current `main` still writes batch schema version 6; published\ntags and their artifacts are not rewritten.",
    ),
    (
        "Unknown schema\nversions, malformed fields, unreadable files, and invalid JSON fail with an\nartifact-path-qualified error instead of silently filling defaults.",
        "Unknown schema\nversions, malformed fields, unreadable files, and invalid JSON fail with an\nartifact-path-qualified error instead of silently filling defaults. Concurrent\n`kind=concurrent` load-test artifacts are rejected explicitly because their\nlatency/throughput semantics are not batch WER/CER semantics.",
    ),
    (
        "There is no concurrent CLI or serialized load-test artifact yet. Those are tracked\nseparately so their public contract can be defined after this Python API is stable.",
        "`callasr concurrent` now exposes this foundation and writes a separate\n`kind=concurrent`, `schema_version=1` load-test artifact. This does not change or\nreuse the batch schema-v6 contract.",
    ),
    (
        "- streaming partial-result metrics;\n- concurrent-call benchmarks;\n",
        "- a real provider-backed streaming adapter and streaming CLI/artifact contract;\n- impairment pipelines inside concurrent load runs;\n",
    ),
    (
        "## Roadmap\n\n1. More local ASR adapters, including GigaAM Multilingual when its packaging path\n   is stable.\n2. Example datasets for first-user smoke runs.\n3. Streaming metrics: time to first partial, finalization latency, and partial\n   transcript stability.\n4. Additional critical-entity slices such as names and addresses after the\n   numeric contract is stable.\n5. Concurrency runs and a comparable public leaderboard format.\n",
        "## Roadmap\n\n1. Validate the streaming contract against a real self-hosted/production ASR\n   implementation before defining streaming serialization.\n2. More local ASR adapters, including GigaAM Multilingual when its packaging path\n   is stable.\n3. Add impairment composition to concurrent load runs only after the clean-load\n   artifact contract has practical usage.\n4. Additional critical-entity slices such as names and addresses after the\n   numeric contract is stable.\n5. Add reporting/leaderboard layers only when datasets and artifact identities\n   are comparable by construction.\n",
    ),
]
for old, new in replacements:
    if text.count(old) != 1:
        raise SystemExit(f"unexpected README occurrence for: {old[:70]!r}")
    text = text.replace(old, new, 1)

anchor = "## Critical numeric entities\n"
if text.count(anchor) != 1:
    raise SystemExit("concurrent CLI section anchor not found")
section = """## Run a concurrent load benchmark

Use `callasr concurrent` to measure throughput and per-request latency with a
fixed number of synchronous worker slots. The first load-test contract uses clean
source WAVs only; telephone impairments are intentionally not accepted by this
command yet.

For local `faster-whisper`:

```bash
uv run callasr concurrent dataset/dataset.jsonl \\
  --adapter faster-whisper \\
  --model large-v3 \\
  --concurrency 4 \\
  --device cuda \\
  --compute-type float16 \\
  --output runs/large-v3-load-c4.json
```

For an OpenAI-compatible endpoint:

```bash
CALLASR_API_KEY=your-secret-key \\
uv run callasr concurrent dataset/dataset.jsonl \\
  --adapter openai-compatible \\
  --model served-asr \\
  --base-url https://asr.example.com/v1 \\
  --concurrency 16 \\
  --timeout-seconds 60 \\
  --output runs/served-asr-load-c16.json
```

The command passes an adapter **factory** to the load runner. Each worker thread
lazily owns one adapter instance; a single model/client object is never shared
between worker threads. For local models this also means memory usage can scale
with concurrency because each active worker may load its own model instance.

Per-item latency times only `adapter.transcribe`. Aggregate wall time begins just
before the first work submission and ends after all successful worker work is
drained. Throughput speed factor is total source-audio seconds divided by that
wall time. Latency p50 and p95 use deterministic nearest-rank percentiles rather
than interpolated quantiles.

Concurrent results use a separate artifact family:

```text
kind: concurrent
schema_version: 1
```

The artifact records the dataset fingerprint, non-secret adapter configuration,
requested concurrency, aggregate throughput/latency metrics, and manifest-order
item results. API keys are never serialized. Writing uses the same same-directory
atomic replacement contract as batch artifacts, so a failed run/write does not
replace an existing completed result.

`callasr compare` intentionally compares batch accuracy artifacts only and rejects
concurrent load artifacts instead of mixing WER/CER and load-test semantics.

"""
text = text.replace(anchor, section + anchor, 1)
path.write_text(text, encoding="utf-8")
