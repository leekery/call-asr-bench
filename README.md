# call-asr-bench

[![CI](https://github.com/leekery/call-asr-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/leekery/call-asr-bench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducible speech-to-text benchmarks for the audio that real phone calls
actually deliver.

Most ASR leaderboards use clean, wideband recordings and report one aggregate
WER. Voice agents receive 8 kHz companded audio and also care about latency,
partial-result stability, numbers, names, and concurrent calls.
`call-asr-bench` is being built around that gap.

## What works today

The first release provides a deterministic benchmark foundation:

- polyphase conversion from arbitrary sample rates to 8 kHz;
- ITU-T G.711 PCMU (μ-law) and PCMA (A-law) encode/decode;
- a composed telephone-channel transform;
- deterministic frame-level packet loss with codec-correct silence substitution;
- Unicode-aware WER for Russian and English transcripts;
- fixed codec vectors and deterministic tests.

Jitter, model adapters, streaming latency, datasets, and the leaderboard are
roadmap items, not simulated results presented as finished features.

## Quick start

```bash
git clone https://github.com/leekery/call-asr-bench.git
cd call-asr-bench
uv sync --extra dev
```

```python
import numpy as np

from callasr import AudioBuffer, telephone_channel, word_error_rate

source = AudioBuffer(
    samples=np.zeros(16_000, dtype=np.float32),
    sample_rate=16_000,
)

phone_audio = telephone_channel(
    source,
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

Both `pcmu` and `pcma` are supported. With the same input and seed, packet-loss
simulation is reproducible. A zero loss rate preserves the original telephone
channel behavior. Inputs are mono floating-point waveforms; codec payloads are
exposed as `numpy.uint8` arrays when lower-level control is needed:

```python
from callasr import apply_packet_loss, decode_g711, encode_g711, resample

telephone_audio = resample(source, target_sample_rate=8_000)
payload = encode_g711(telephone_audio.samples, codec="pcma")
impaired = apply_packet_loss(
    payload,
    codec="pcma",
    loss_rate=0.05,
    frame_duration_ms=20,
    seed=42,
)
decoded = decode_g711(impaired, codec="pcma")
```

## Development

```bash
uv run --extra dev pytest -v
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
```

The package supports Python 3.10 through 3.13.

## Roadmap

1. A benchmark manifest and CLI with reproducible run artifacts.
2. Jitter, noise, and configurable telephone channel profiles.
3. Adapters for local models and OpenAI-compatible ASR endpoints.
4. Streaming metrics: time to first partial, finalization latency, and partial
   transcript stability.
5. Accuracy slices for numbers, names, addresses, Russian, and English.
6. Concurrency runs and a comparable public leaderboard format.

## License

[MIT](LICENSE)
