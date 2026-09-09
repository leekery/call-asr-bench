from pathlib import Path

path = Path("README.md")
text = path.read_text(encoding="utf-8")
anchor = "## Current development limitations\n"
if text.count(anchor) != 1:
    raise SystemExit("unexpected README concurrent anchor")
section = """## Concurrent benchmark foundation

Current `main` also includes a Python-only concurrency foundation for synchronous
ASR adapters through `run_concurrent_benchmark`. It is separate from the
sequential `callasr run` workflow and does not change the batch schema-v6
artifact contract.

The concurrent runner accepts an adapter factory and a positive `concurrency`.
Each worker thread lazily creates and then reuses its own adapter instance; one
adapter object is never shared across worker threads. Scheduling is bounded to at
most `concurrency` in-flight dataset items, while returned item results remain in
manifest order even when workers finish out of order.

The first foundation intentionally benchmarks the clean source-WAV path only. It
reports per-item transcription latency plus aggregate source-audio seconds, total
observed wall time, throughput speed factor, and latency p50/p95/max. Item latency
times only `adapter.transcribe`; WAV loading and lazy adapter construction are
outside that per-item timing boundary.

Latency percentiles use an explicit nearest-rank definition: sort the N observed
latencies, then percentile `p` is rank `ceil(p * N / 100)`, clamped to ranks
1 through N. This avoids interpolation differences between statistics libraries.
If total wall time is zero, throughput speed factor is `None`.

Failures do not produce a partial-success result. The runner stops submitting new
items, cancels pending work where possible, drains already-running worker tasks,
and re-raises the original factory or adapter exception.

There is no concurrent CLI or serialized load-test artifact yet. Those are tracked
separately so their public contract can be defined after this Python API is stable.

"""
path.write_text(text.replace(anchor, section + anchor, 1), encoding="utf-8")
