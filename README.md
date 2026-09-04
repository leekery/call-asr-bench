# call-asr-bench

[![CI](https://github.com/leekery/call-asr-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/leekery/call-asr-bench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducible speech-to-text benchmarks for the audio that real phone calls
actually deliver.

Most ASR leaderboards use clean, wideband recordings and report one aggregate
WER. Voice agents often receive 8 kHz G.711 audio, so model quality can change
once the same utterances pass through a telephone channel. `call-asr-bench`
provides a small local workflow for comparing clean and impaired audio with
reproducible inputs and a versioned JSON result artifact.

## What works today

The end-to-end local runner supports:

- strict UTF-8 JSONL dataset manifests with relative WAV paths;
- uncompressed mono integer-PCM WAV loading;
- clean-audio runs and deterministic 8 kHz G.711 PCMU / PCMA runs;
- deterministic SNR-controlled Gaussian additive noise;
- deterministic frame-level packet loss with codec-correct silence substitution;
- a model-agnostic ASR adapter boundary and a local `faster-whisper` adapter;
- per-utterance and corpus WER / CER;
- measured adapter time, real-time factor (RTF), and speed factor;
- the `callasr run` CLI;
- atomic schema-versioned JSON artifacts.

## Install

Clone the repository and install the `faster-whisper` optional extra:

```bash
git clone https://github.com/leekery/call-asr-bench.git
cd call-asr-bench
uv sync --extra faster-whisper
```

The examples below use `uv run callasr` so activating the virtual environment is
not required. The installed console entry point itself is `callasr`.

## Prepare a dataset

Create a UTF-8 JSONL manifest with one utterance per line. Audio paths are
resolved relative to the manifest file:

```json
{"id":"call-001","audio":"audio/call-001.wav","reference":"добрый день","language":"ru"}
{"id":"call-002","audio":"audio/call-002.wav","reference":"your order is ready","language":"en"}
```

The fields are:

- `id`: non-empty string, unique inside the manifest;
- `audio`: non-empty path to an existing WAV file;
- `reference`: reference transcript; an empty string is valid;
- `language`: optional lowercase ISO 639-1 code such as `ru` or `en`.

Unknown fields are rejected. Blank lines are allowed. WAV input must be
uncompressed, mono, integer PCM. The benchmark loader keeps the source sample
rate; model-specific resampling belongs to the adapter. The `faster-whisper`
adapter converts its input to 16 kHz internally.

A minimal directory can look like this:

```text
dataset/
├── dataset.jsonl
└── audio/
    ├── call-001.wav
    └── call-002.wav
```

## Run a clean baseline

Use `--codec none` for the source audio without the G.711 telephone transform:

```bash
uv run callasr run dataset/dataset.jsonl \
  --adapter faster-whisper \
  --model large-v3 \
  --codec none \
  --output runs/large-v3-clean.json
```

Packet loss must be zero when `--codec none` is selected. Zero is the default,
so the clean command does not need a packet-loss flag.

`faster-whisper` uses `--device auto` and `--compute-type default` by default.
They can be overridden explicitly, for example:

```bash
--device cuda --compute-type float16
```

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

The same dataset can be passed through additive noise and then G.711 PCMU before
transcription:

```bash
uv run callasr run dataset/dataset.jsonl \
  --adapter faster-whisper \
  --model large-v3 \
  --codec pcmu \
  --packet-loss-rate 0.05 \
  --frame-duration-ms 20 \
  --snr-db 15 \
  --seed 42 \
  --output runs/large-v3-pcmu-noisy.json
```

Use `--codec pcma` for G.711 A-law. The impairment order is fixed:

```text
source WAV
→ optional additive noise
→ G.711 resample / encode
→ optional frame-level packet loss
→ G.711 decode
→ ASR adapter
```

Packet loss is applied at encoded-frame level. The packet-loss seed derivation
is unchanged from v0.2: the run-level seed is combined with each manifest
position, so repeated runs with the same dataset and seed use the same loss
masks without applying an identical mask to every utterance. Additive noise uses
a separate deterministic per-item random stream and therefore does not perturb
the packet-loss sequence.

The runner is sequential and stops on the first dataset, audio, channel, or
transcription failure. The requested output is replaced only after every item
succeeds. A failed run does not leave a partial result that looks like a
complete benchmark artifact.

## Result artifact

Current `main` writes UTF-8 JSON with `schema_version` set to `2`. The main
sections are:

- `dataset`: resolved manifest path and item count;
- `adapter`: adapter name, model identifier, device, compute type, and decoding
  options;
- `channel`: codec, packet-loss rate, frame duration, run seed, and nullable
  `additive_noise_snr_db`;
- `summary`: corpus audio time, measured adapter time, WER, CER, RTF, and speed
  factor;
- `items`: ordered per-utterance references, hypotheses, language tags,
  durations, timings, WER, and CER.

When additive noise is disabled, `channel.additive_noise_snr_db` is `null`.

The published `v0.2.0` tag remains the schema-version-1 release. Schema version
2 is the current unreleased development contract on `main`; the old release and
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

**RTF** is:

```text
RTF = total adapter seconds / total input-audio seconds
```

Only the adapter call is timed. WAV loading, impairment processing, and JSON
serialization are excluded. Lower RTF is faster.

**speed factor** is the reciprocal view:

```text
speed factor = total input-audio seconds / total adapter seconds
```

A speed factor of `10` means the measured adapter processed audio at roughly 10x
real time for that run. It is `null` if the measured adapter time is zero.

## Errors and output safety

Expected dataset, audio, adapter, configuration, and artifact-write errors are
printed as a concise stderr message and return exit status `2`. Unexpected
exceptions are not converted into user errors, so developer defects retain a
normal traceback.

Artifact writing uses a temporary file in the destination directory followed
by an atomic replace. Parent directories are created when necessary.

## Lower-level Python API

The audio primitives remain available independently of the CLI:

```python
import numpy as np

from callasr import AudioBuffer, apply_additive_noise, telephone_channel, word_error_rate

source = AudioBuffer(
    samples=np.full(16_000, 0.1, dtype=np.float32),
    sample_rate=16_000,
)

noisy = apply_additive_noise(source, snr_db=15.0, seed=42)
phone_audio = telephone_channel(
    noisy,
    codec="pcmu",
    packet_loss_rate=0.05,
    frame_duration_ms=20,
    seed=42,
)
score = word_error_rate(
    reference="добрый день чем могу помочь",
    hypothesis="добрый день чем могу вам помочь",
)

print(phone_audio.sample_rate)  # 8000
print(score)  # 0.2
```

## Current development limitations

The current local runner does not provide:

- streaming partial-result metrics;
- jitter simulation or richer channel profiles;
- concurrent-call benchmarks;
- remote ASR API adapters;
- GigaAM integration;
- automatic dataset downloading;
- a hosted leaderboard.

These remain follow-up areas after the local runner and artifact contract are
stable.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
```

The package supports Python 3.10 through 3.13.

## Roadmap

1. Additional telephone-channel profiles: jitter and related impairments.
2. More local and OpenAI-compatible ASR adapters.
3. Streaming metrics: time to first partial, finalization latency, and partial
   transcript stability.
4. Accuracy slices for numbers, names, addresses, Russian, and English.
5. Concurrency runs and a comparable public leaderboard format.

## License

[MIT](LICENSE)
