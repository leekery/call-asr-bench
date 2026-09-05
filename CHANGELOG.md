# Changelog

All notable changes to `call-asr-bench` are documented here.

## 0.3.0 - 2026-09-05

### Added

- deterministic SNR-controlled additive-noise impairment with independent per-item seeds;
- deterministic G.711 frame jitter / late-packet-loss modeling with fixed playout-buffer configuration;
- low-level deterministic gain and symmetric hard-clipping waveform transform;
- OpenAI-compatible `/audio/transcriptions` adapter with optional `httpx` extra, sanitized errors, and non-secret endpoint metadata;
- digit-form phone-number and numeric-entity extraction, diagnostics, and corpus preservation accuracy for Russian/English benchmark transcripts;
- `callasr compare` for deterministic Markdown comparison of saved schema-v1 through schema-v4 artifacts without rerunning models;
- a tiny reproducible RU/EN synthetic non-speech smoke fixture pack with byte-reproducible WAV generator;
- repository-specific release/versioning policy and contributor extension guide.

### Changed

- current result artifacts use schema version 4;
- telephone benchmark runs can combine additive noise, G.711 packet loss, and late-frame jitter while preserving the earlier packet-loss/noise random streams;
- remote endpoint RTF includes WAV serialization, HTTP round-trip, server inference, and response parsing because those operations occur inside the adapter call;
- result artifacts now contain explainable per-item numeric-entity diagnostics while WER/CER normalization remains unchanged.

### Compatibility

- published `v0.2.0` artifacts remain schema version 1 and are not rewritten;
- `callasr compare` explicitly supports schemas 1, 2, 3, and 4 for the common WER/CER/RTF/speed metrics;
- later optional fields are shown as unavailable for older artifacts rather than synthesized as zero;
- API keys are not serialized into OpenAI-compatible adapter metadata or benchmark artifacts.

### Notes

- gain/clipping is available as a lower-level Python primitive but is not yet exposed through `callasr run`;
- the smoke fixture is synthetic non-speech and is only for end-to-end plumbing verification, not model-quality comparison;
- GigaAM Multilingual integration remains deferred because upstream multilingual packaging is not yet available through the stable published package path;
- streaming metrics, concurrency benchmarking, automatic dataset downloading, and hosted leaderboard infrastructure remain future work.

## 0.2.0 - 2026-09-04

### Added

- strict UTF-8 JSONL benchmark manifests with relative audio paths and validation;
- uncompressed mono integer-PCM WAV loading into `AudioBuffer`;
- WER and CER edit counts with corpus micro-aggregation and schema-versioned result models;
- a stable `ASRAdapter` protocol and optional `faster-whisper` integration;
- deterministic 16 kHz model input conversion and decoding metadata for `faster-whisper`;
- a sequential benchmark runner for clean, PCMU, and PCMA audio;
- deterministic per-item packet-loss seeds derived from the run seed;
- adapter-only timing, RTF, and speed-factor reporting;
- the `callasr run` command-line workflow;
- atomic UTF-8 JSON result artifacts with ordered per-item results;
- end-to-end documentation for clean and telephone-channel local benchmarks.

### Notes

Version 0.2.0 is a local sequential benchmark release. It does not include streaming metrics, jitter, additive noise, concurrency, remote ASR APIs, GigaAM integration, automatic dataset downloading, or a hosted leaderboard.
