# call-asr-bench

[![CI](https://github.com/leekery/call-asr-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/leekery/call-asr-bench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducible speech-to-text benchmarks for the audio that real phone calls
actually deliver.

Most ASR leaderboards use clean, wideband recordings and report one aggregate
WER. Voice agents often receive 8 kHz G.711 audio, so model quality can change
once the same utterances pass through a telephone channel. `call-asr-bench`
provides a small workflow for comparing clean and impaired audio with
reproducible inputs and a versioned JSON result artifact.

## What works today

The end-to-end runner supports:

- strict UTF-8 JSONL dataset manifests with relative WAV paths;
- uncompressed mono integer-PCM WAV loading;
- clean-audio runs and deterministic 8 kHz G.711 PCMU / PCMA runs;
- deterministic front-end gain and symmetric hard clipping;
- deterministic SNR-controlled Gaussian additive noise;
- deterministic frame-level packet loss with codec-correct silence substitution;
- deterministic frame jitter modeled as late G.711 packet loss against a fixed
  playout buffer;
- a model-agnostic ASR adapter boundary;
- local in-process `faster-whisper` inference;
- OpenAI-compatible `/audio/transcriptions` endpoints, including self-hosted
  servers such as vLLM;
- per-utterance and corpus WER / CER;
- digit-form phone-number and numeric-entity preservation accuracy;
- measured adapter time, real-time factor (RTF), and speed factor;
- the `callasr run` CLI;
- the `callasr concurrent` load-test CLI with worker-local adapter instances;
- deterministic Markdown comparison of saved schema-v1 through schema-v6 artifacts with dataset identity and front-end configuration checks;
- atomic schema-versioned JSON artifacts.

Gain and symmetric hard clipping are available through both `callasr run` and
the lower-level Python API.

## Install

Clone the repository first:

```bash
git clone https://github.com/leekery/call-asr-bench.git
cd call-asr-bench
```

For local `faster-whisper` inference, install its optional extra:

```bash
uv sync --extra faster-whisper
```

For an OpenAI-compatible HTTP endpoint, install the lightweight HTTP extra:

```bash
uv sync --extra openai-compatible
```

For the vLLM Realtime WebSocket streaming adapter, install its lightweight
transport extra:

```bash
uv sync --extra vllm-realtime
```

The examples below use `uv run callasr` so activating the virtual environment is
not required. The installed console entry point itself is `callasr`.

## Prepare a dataset

Create a UTF-8 JSONL manifest with one utterance per line. Audio paths are
resolved relative to the manifest file:

```json
{"id":"call-001","audio":"audio/call-001.wav","reference":"добрый день, мой номер +7 (916) 123-45-67","language":"ru"}
{"id":"call-002","audio":"audio/call-002.wav","reference":"your order code is 0042","language":"en"}
```

The fields are:

- `id`: non-empty string, unique inside the manifest;
- `audio`: non-empty path to an existing WAV file;
- `reference`: reference transcript; an empty string is valid;
- `language`: optional lowercase ISO 639-1 code such as `ru` or `en`.

Unknown fields are rejected. Blank lines are allowed. WAV input must be
uncompressed, mono, integer PCM. The benchmark loader keeps the source sample
rate; model-specific or transport-specific conversion belongs to the adapter.
The `faster-whisper` adapter converts its input to 16 kHz internally.

A minimal directory can look like this:

```text
dataset/
├── dataset.jsonl
└── audio/
    ├── call-001.wav
    └── call-002.wav
```

## Run a clean local baseline

Use `--codec none` for the source audio without the G.711 telephone transform:

```bash
uv run callasr run dataset/dataset.jsonl \
  --adapter faster-whisper \
  --model large-v3 \
  --codec none \
  --output runs/large-v3-clean.json
```

Packet loss must be zero when `--codec none` is selected. Zero is the default,
so the clean command does not need a packet-loss flag. Jitter is an encoded
G.711-frame impairment and is therefore available only with `pcmu` or `pcma`.

`faster-whisper` uses `--device auto` and `--compute-type default` by default.
They can be overridden explicitly, for example:

```bash
--device cuda --compute-type float16
```

## Benchmark an OpenAI-compatible endpoint

`openai-compatible` targets an API root that already includes its version path,
for example `http://localhost:8000/v1`. The adapter appends
`/audio/transcriptions`; it does not guess or add `/v1` itself.

For a local vLLM-style server that does not require authentication:

```bash
uv run callasr run dataset/dataset.jsonl \
  --adapter openai-compatible \
  --model openai/whisper-large-v3-turbo \
  --base-url http://localhost:8000/v1 \
  --codec none \
  --output runs/served-whisper-clean.json
```

For an authenticated endpoint, prefer an environment variable instead of
putting the key in shell history:

```bash
CALLASR_API_KEY=your-secret-key \
uv run callasr run dataset/dataset.jsonl \
  --adapter openai-compatible \
  --model served-asr \
  --base-url https://asr.example.com/v1 \
  --timeout-seconds 60 \
  --output runs/served-asr.json
```

API-key resolution is deterministic:

1. explicit `--api-key`;
2. `CALLASR_API_KEY`;
3. `OPENAI_API_KEY`;
4. no Authorization header.

The API key is never included in benchmark adapter metadata or result artifacts.
The base URL is also rejected if it contains URL user credentials, a query
string, or a fragment, which avoids accidentally persisting credentials through
the reproducibility metadata.

The HTTP adapter uploads each `AudioBuffer` as an in-memory mono PCM16 WAV at its
current sample rate. Samples are clipped to `[-1, 1]` at this transport boundary
before PCM16 conversion. The artifact records `upload_format=wav_pcm16`, the
non-secret base URL, response format, and timeout so this conversion is visible
and reproducible.

For a remote adapter, the timed adapter call includes WAV serialization, the HTTP
round trip, server inference, and response parsing. RTF therefore measures
observed endpoint latency/throughput from the benchmark client rather than only
the server's internal model compute time.

## Apply deterministic gain and clipping

Use `--gain-db` to model a fixed front-end level change and `--clip-threshold` to
model symmetric hard clipping before any acoustic noise or telephone-channel
processing. Gain uses the amplitude conversion `10 ** (gain_db / 20)` and is
applied first; clipping then limits samples to `[-threshold, +threshold]`.

```bash
uv run callasr run dataset/dataset.jsonl \
  --adapter faster-whisper \
  --model large-v3 \
  --codec none \
  --gain-db 6 \
  --clip-threshold 0.8 \
  --output runs/large-v3-front-end.json
```

There is no automatic normalization and no implicit clipping to `[-1, 1]` in
this impairment. Omit `--clip-threshold` to disable hard clipping. The defaults
`--gain-db 0` with no clip threshold preserve the earlier audio path exactly;
the transform is not invoked at all in that configuration.

## Add deterministic acoustic noise

Use `--snr-db` to add seeded zero-mean Gaussian noise before any telephone codec
processing. The value is an amplitude signal-to-noise ratio in decibels. Higher
values are cleaner; `0` means equal signal and noise RMS, and negative finite
values are allowed.

A noisy clean-audio run is valid:

```bash
uv run callasr run dataset/dataset.jsonl \
  --adapter faster-whisper \
  --model large-v3 \
  --codec none \
  --snr-db 15 \
  --seed 42 \
  --output runs/large-v3-noisy-clean.json
```

Omit `--snr-db` to disable additive noise. `nan`, `inf`, and `-inf` are rejected.
The run seed and manifest position derive an independent deterministic noise
seed for each utterance.

## Run a telephone-channel benchmark

The same dataset can be passed through front-end gain/clipping, additive noise,
G.711, packet loss, and late-frame jitter before transcription. The impairment
pipeline is independent of which ASR adapter receives the resulting audio.

```bash
uv run callasr run dataset/dataset.jsonl \
  --adapter faster-whisper \
  --model large-v3 \
  --codec pcmu \
  --packet-loss-rate 0.05 \
  --frame-duration-ms 20 \
  --snr-db 15 \
  --jitter-std-ms 8 \
  --playout-buffer-ms 20 \
  --seed 42 \
  --output runs/large-v3-pcmu-impaired.json
```

Use `--codec pcma` for G.711 A-law. `--jitter-std-ms` and
`--playout-buffer-ms` must be supplied together. The jitter model samples an
independent zero-mean Gaussian delay variation for each encoded frame; a frame
whose positive delay variation exceeds the fixed playout buffer is replaced by
the codec-correct silence value. This is a bounded late-arrival approximation,
not a complete RTP or adaptive jitter-buffer simulator.

The impairment order is fixed:

```text
source WAV
→ optional gain / hard clipping
→ optional additive noise
→ G.711 resample / encode
→ optional frame-level packet loss
→ optional jitter / late-frame loss
→ G.711 decode
→ ASR adapter
```

The random streams are deterministic and intentionally independent:

```text
packet loss: SeedSequence([run_seed, item_index])
noise:       SeedSequence([run_seed, item_index, 1])
jitter:      SeedSequence([run_seed, item_index, 2])
```

The packet-loss and additive-noise derivations are unchanged from their earlier
contracts, so enabling jitter does not perturb either existing random stream.
Runs without jitter keep the earlier telephone-channel call path.

The runner is sequential and stops on the first dataset, audio, channel, or
transcription failure. The requested output is replaced only after every item
succeeds. A failed run does not leave a partial result that looks like a
complete benchmark artifact.

## Run a concurrent load benchmark

Use `callasr concurrent` to measure throughput and per-request latency with a
fixed number of synchronous worker slots. The first load-test contract uses clean
source WAVs only; telephone impairments are intentionally not accepted by this
command yet.

For local `faster-whisper`:

```bash
uv run callasr concurrent dataset/dataset.jsonl \
  --adapter faster-whisper \
  --model large-v3 \
  --concurrency 4 \
  --device cuda \
  --compute-type float16 \
  --output runs/large-v3-load-c4.json
```

For an OpenAI-compatible endpoint:

```bash
CALLASR_API_KEY=your-secret-key \
uv run callasr concurrent dataset/dataset.jsonl \
  --adapter openai-compatible \
  --model served-asr \
  --base-url https://asr.example.com/v1 \
  --concurrency 16 \
  --timeout-seconds 60 \
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

## Critical numeric entities

WER can stay low while a phone number or code becomes unusable. Current `main`
therefore scores preservation of **digit-form** phone and numeric entities in
addition to WER/CER.

The v0.3 contract is deliberately conservative and representation-preserving:

- a phone-like span has an optional leading `+`, at least seven ASCII digits,
  and may contain spaces, parentheses, or hyphens between digits;
- phone formatting is removed for comparison, so `+7 (916) 123-45-67` and
  `+7 916 123 45 67` both canonicalize to `+79161234567`;
- other standalone digit runs are generic numeric entities;
- leading zeroes are semantic, so `0042` does not match `42`;
- number words are **not** converted to digits. For example, a reference
  `+7 (916) 123-45-67` does not silently match a hypothesis such as
  `плюс семь девятьсот шестнадцать ...`.

That final rule is intentional. It makes representation failures from models
that spell numbers out visible instead of depending on a hidden language-specific
number normalizer.

Reference and hypothesis entity sequences are aligned in transcript order with
exact-canonical longest-common-subsequence matching. Per-item accuracy is:

```text
numeric entity accuracy = matched reference entities / reference entity count
```

If the reference has no scorable digit-form entities, item accuracy is `null`
and that item is excluded from the corpus denominator. Hypothesis-only extra
numeric entities are still recorded in the artifact for diagnosis, but v0.3's
metric is a preservation/recall-style score and does not penalize those extras.

## Result artifact

Current `main` writes UTF-8 JSON with `schema_version` set to `6`. The main
sections are:

- `dataset`: resolved manifest path, item count, and path-independent dataset fingerprint;
- `adapter`: adapter name, model identifier, device, compute type, and decoding
  options;
- `channel`: codec, packet-loss rate, frame duration, run seed, `gain_db`,
  nullable `clip_threshold`, nullable `additive_noise_snr_db`, nullable
  `jitter_std_ms`, and nullable `playout_buffer_ms`;
- `summary`: corpus audio time, measured adapter time, WER, CER, RTF, speed
  factor, numeric-entity matches/reference count, and nullable micro-averaged
  numeric-entity accuracy;
- `items`: ordered per-utterance references, hypotheses, language tags,
  durations, timings, WER/CER, and a `numeric_entities` diagnostic object.

Each item records extracted reference and hypothesis entities with `kind`, the
original `surface`, and the comparison `canonical` value, plus `matches`,
`reference_count`, and nullable `accuracy`. A failed phone-number comparison can
therefore be inspected directly rather than inferred from one aggregate score.

For `openai-compatible`, `adapter.device` is `remote`, `compute_type` is
`server`, and `decoding_options` contains only non-secret endpoint metadata:
`base_url`, `response_format`, `upload_format`, and `timeout_seconds`.

Disabled optional impairments are represented by `null` channel fields. Jitter
parameters are always either both set or both `null`.

Published `v0.2.0` artifacts remain schema version 1, published `v0.3.0`
artifacts remain schema version 4, and published `v0.4.0` batch artifacts remain
schema version 6. Current `main` still writes batch schema version 6; published
tags and their artifacts are not rewritten.

Item audio paths are stored relative to the manifest when possible. JSON is
pretty-printed with two-space indentation and preserves non-ASCII text rather
than escaping Russian or other Unicode transcripts.

## Compare saved result artifacts

Use `callasr compare` to compare completed runs without loading audio or invoking
any ASR model:

```bash
uv run callasr compare \
  runs/large-v3-clean.json \
  runs/large-v3-pcmu-impaired.json \
  runs/served-asr.json
```

The command writes a deterministic Markdown table to stdout. It accepts known
call-asr-bench schema versions 1 through 6 and keeps the schema version visible
for every row. WER, CER, RTF, and speed-factor definitions are compatible across
those schema versions. Fields introduced later are not invented for older
artifacts: SNR, jitter, or numeric-entity accuracy render as `—` when the source
schema does not contain them. Numeric zero remains `0`, not `—`.

Rows preserve command-line input order. Model names and artifact filenames are
escaped so pipes or newlines cannot corrupt the Markdown table. Unknown schema
versions, malformed fields, unreadable files, and invalid JSON fail with an
artifact-path-qualified error instead of silently filling defaults. Concurrent
`kind=concurrent` load-test artifacts are rejected explicitly because their
latency/throughput semantics are not batch WER/CER semantics.

Schema v6 additionally records front-end gain/clipping. `callasr compare` shows
explicit `Gain dB` and `Clip` columns; schema-v1 through schema-v5 artifacts render
those fields as unavailable rather than assuming zero.

Schema-v5 artifacts add `dataset.fingerprint` as a versioned SHA-256 identity.
The fingerprint is order-sensitive and includes each item's `id`, reference text,
optional language, and the SHA-256 digest of the exact source WAV bytes. Absolute
manifest locations and audio pathnames are excluded, so moving or renaming an
otherwise identical dataset does not change its identity.

When `callasr compare` receives two or more artifacts with known schema-v5
fingerprints, all known fingerprints must match; a mismatch is a hard error rather
than a potentially misleading table. Schema-v1 through schema-v4 artifacts remain
readable, but they do not contain enough information to prove dataset identity.

### Metrics

**WER** is normalized word-level edit distance. Corpus WER is micro-averaged
from total word edits and total reference words rather than averaging each
utterance's WER equally.

**CER** is the same idea at character level after the project's text
normalization; spaces are removed for the character comparison.

Numeric-entity extraction and scoring is a separate metric pipeline. It does
not alter the WER/CER normalization contract.

**RTF** is:

```text
RTF = total adapter seconds / total input-audio seconds
```

Only the adapter call is timed. WAV loading, impairment processing, metric
calculation, and JSON serialization are excluded. Lower RTF is faster. For the
remote adapter, transport serialization and HTTP/server latency are intentionally
inside the adapter call and therefore inside RTF.

**speed factor** is the reciprocal view:

```text
speed factor = total input-audio seconds / total adapter seconds
```

A speed factor of `10` means the measured adapter processed audio at roughly 10x
real time for that run. It is `null` if the measured adapter time is zero.

## Errors and output safety

Expected dataset, audio, adapter, configuration, artifact-write, and comparison
errors are printed as a concise stderr message and return exit status `2`. Remote
endpoint errors report sanitized status/transport information and do not include
response bodies or API keys. Unexpected exceptions are not converted into user
errors, so developer defects retain a normal traceback.

Artifact writing uses a temporary file in the destination directory followed
by an atomic replace. Parent directories are created when necessary.

## Lower-level Python API

The audio and metric primitives remain available independently of the CLI:

```python
import numpy as np

from callasr import (
    AudioBuffer,
    apply_additive_noise,
    apply_gain_and_clip,
    score_numeric_entities,
    telephone_channel,
    word_error_rate,
)

source = AudioBuffer(
    samples=np.full(16_000, 0.1, dtype=np.float32),
    sample_rate=16_000,
)

front_end = apply_gain_and_clip(source, gain_db=3.0, clip_threshold=0.9)
noisy = apply_additive_noise(front_end, snr_db=15.0, seed=42)
phone_audio = telephone_channel(
    noisy,
    codec="pcmu",
    packet_loss_rate=0.05,
    frame_duration_ms=20,
    seed=42,
    jitter_std_ms=8.0,
    playout_buffer_ms=20.0,
    jitter_seed=43,
)
wer = word_error_rate(
    reference="добрый день чем могу помочь",
    hypothesis="добрый день чем могу вам помочь",
)
entities = score_numeric_entities(
    reference="мой номер +7 (916) 123-45-67",
    hypothesis="мой номер +7 916 123 45 67",
)

print(phone_audio.sample_rate)  # 8000
print(wer)  # 0.2
print(entities.accuracy)  # 1.0
```

## Streaming benchmark foundation

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

The streaming foundation itself remains separate from the batch CLI/artifact
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
CALLASR_VLLM_REALTIME_URL=http://localhost:8000/v1 \
CALLASR_VLLM_REALTIME_MODEL=mistralai/Voxtral-Mini-4B-Realtime-2602 \
CALLASR_VLLM_REALTIME_WAV=/absolute/path/to/example-16k.wav \
uv run --extra vllm-realtime pytest -q tests/test_vllm_realtime_live.py
```

Set `CALLASR_VLLM_REALTIME_API_KEY` as well when the server requires Bearer
authentication. The live test is skipped by default, so normal CI performs no
network access and downloads no model.

## Concurrent benchmark foundation

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

`callasr concurrent` now exposes this foundation and writes a separate
`kind=concurrent`, `schema_version=1` load-test artifact. This does not change or
reuse the batch schema-v6 contract.

## Current development limitations

The current runner does not provide:

- word-to-digit normalization for spoken numeric forms;
- critical-entity scoring for names or addresses;
- provider-specific remote features beyond the common transcription contract;
- a full RTP/adaptive jitter-buffer, packet reordering, duplication, or
  correlated network-delay simulation;
- a streaming CLI/artifact contract and paced full-duplex microphone timing;
- impairment pipelines inside concurrent load runs;
- GigaAM integration;
- automatic dataset downloading;
- a hosted leaderboard.

These remain follow-up areas after the runner and artifact contract are stable.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
```

The package supports Python 3.10 through 3.13. Default CI does not install the
OpenAI-compatible extra and does not make external transcription requests; HTTP
behavior is covered with an injected fake client.

## Roadmap

1. Validate the streaming contract against a real self-hosted/production ASR
   implementation before defining streaming serialization.
2. More local ASR adapters, including GigaAM Multilingual when its packaging path
   is stable.
3. Add impairment composition to concurrent load runs only after the clean-load
   artifact contract has practical usage.
4. Additional critical-entity slices such as names and addresses after the
   numeric contract is stable.
5. Add reporting/leaderboard layers only when datasets and artifact identities
   are comparable by construction.

## License

[MIT](LICENSE)