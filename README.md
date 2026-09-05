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
- atomic schema-versioned JSON artifacts.

The lower-level Python API also includes deterministic gain and hard-clipping
transforms. Gain/clipping is not yet wired into `callasr run`.

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

The same dataset can be passed through additive noise, G.711, packet loss, and
late-frame jitter before transcription. The impairment pipeline is independent
of which ASR adapter receives the resulting audio.

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

Current `main` writes UTF-8 JSON with `schema_version` set to `4`. The main
sections are:

- `dataset`: resolved manifest path and item count;
- `adapter`: adapter name, model identifier, device, compute type, and decoding
  options;
- `channel`: codec, packet-loss rate, frame duration, run seed, nullable
  `additive_noise_snr_db`, nullable `jitter_std_ms`, and nullable
  `playout_buffer_ms`;
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

The published `v0.2.0` tag remains the schema-version-1 release. Schema version
4 is the current unreleased development contract on `main`; the old release and
its artifacts are not rewritten.

Item audio paths are stored relative to the manifest when possible. JSON is
pretty-printed with two-space indentation and preserves non-ASCII text rather
than escaping Russian or other Unicode transcripts.

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

Expected dataset, audio, adapter, configuration, and artifact-write errors are
printed as a concise stderr message and return exit status `2`. Remote endpoint
errors report sanitized status/transport information and do not include response
bodies or API keys. Unexpected exceptions are not converted into user errors, so
developer defects retain a normal traceback.

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

## Current development limitations

The current runner does not provide:

- word-to-digit normalization for spoken numeric forms;
- critical-entity scoring for names or addresses;
- gain/clipping configuration through the runner or CLI;
- provider-specific remote features beyond the common transcription contract;
- a full RTP/adaptive jitter-buffer, packet reordering, duplication, or
  correlated network-delay simulation;
- streaming partial-result metrics;
- concurrent-call benchmarks;
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

1. More local ASR adapters, including GigaAM Multilingual when its packaging path
   is stable.
2. Comparison reports from saved benchmark artifacts.
3. Example datasets for first-user smoke runs.
4. Streaming metrics: time to first partial, finalization latency, and partial
   transcript stability.
5. Additional critical-entity slices such as names and addresses after the
   numeric contract is stable.
6. Concurrency runs and a comparable public leaderboard format.

## License

[MIT](LICENSE)
