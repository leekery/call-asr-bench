# Changelog

All notable changes to `call-asr-bench` are documented here.

## 0.5.0 - 2026-09-09

### Added

- a Python-only concurrent benchmark foundation for synchronous ASR adapters with bounded worker scheduling;
- lazy worker-local adapter factories so one adapter instance is never shared across concurrent worker threads;
- deterministic manifest-order concurrent results even when requests complete out of order;
- per-item adapter latency, aggregate wall time, throughput speed factor, and explicit nearest-rank p50/p95/max latency metrics;
- `callasr concurrent` for clean-source-WAV load tests using faster-whisper or OpenAI-compatible adapters;
- a separate `kind=concurrent`, schema-version-1 load-test artifact family with dataset fingerprints, non-secret adapter metadata, load configuration, aggregate metrics, and ordered item results;
- `VLLMRealtimeAdapter` for vLLM's self-hosted WebSocket `/v1/realtime` transcription protocol;
- deterministic 16 kHz PCM16/base64 vLLM Realtime transport with cumulative partial transcripts and authoritative final transcript handling;
- optional `vllm-realtime` transport dependencies and an opt-in live vLLM smoke test.

### Changed

- `callasr compare` explicitly rejects concurrent load-test artifacts instead of mixing load latency/throughput with batch WER/CER semantics;
- concurrent benchmark failures stop new submissions, cancel pending work where possible, drain already-running work, and re-raise the original factory/adapter error rather than returning partial-success summaries;
- the streaming foundation is now validated against a real self-hosted protocol while remaining separate from batch `callasr run` and batch artifact schema v6.

### Compatibility

- batch artifacts remain schema version 6; v0.5.0 does not introduce batch schema v7;
- concurrent artifacts use their own `kind=concurrent`, schema version 1 contract;
- published `v0.2.0`, `v0.3.0`, and `v0.4.0` artifacts remain immutable;
- existing WER/CER/RTF/entity metrics and impairment seed streams are unchanged;
- API keys for HTTP and WebSocket integrations are not serialized into artifacts or reproducibility metadata;
- the vLLM Realtime optional dependency is constrained to `websockets>=14,<17` so the package continues to support Python 3.10.

### Notes

- the first concurrent CLI intentionally benchmarks clean source WAVs only; impairment composition for load tests remains future work;
- the vLLM Realtime adapter requires mono 16 kHz input and rejects explicit language selection because the current upstream realtime session contract does not expose a language field;
- the first vLLM integration follows the documented file-style sequence and submits all audio before consuming transcript events, so time-to-first-partial is not a paced full-duplex microphone TTFT metric; finalization latency remains available;
- streaming CLI/serialization and paced full-duplex timing remain future work;
- GigaAM Multilingual integration remains tracked separately in issue #22 and is not part of v0.5.0.

## 0.4.0 - 2026-09-09

### Added

- path-independent, order-sensitive dataset fingerprints that include item metadata and exact source-WAV content hashes;
- comparison-time dataset identity checks for fingerprinted artifacts, preventing silent comparison of different known datasets;
- end-to-end front-end gain and symmetric hard-clipping configuration through `run_benchmark` and `callasr run`;
- explicit `Gain dB` and `Clip` columns in `callasr compare`;
- a separate Python-only `StreamingASRAdapter` protocol and immutable streaming update/result models;
- deterministic exact-sample audio framing for streaming benchmarks;
- streaming latency metrics for time to first non-empty partial, finalization after the last submitted frame, and total observed streaming wall time;
- normalized token-level partial-transcript stability with deterministic duplicate-state collapsing and revision scoring;
- strict streaming contract validation for incomplete frame consumption, invalid updates, finalization violations, and non-monotonic clocks.

### Changed

- current batch result artifacts use schema version 6;
- schema-v5 artifacts add `dataset.fingerprint`, while schema v6 additionally records `channel.gain_db` and nullable `channel.clip_threshold`;
- `callasr compare` accepts schemas 1 through 6 and exposes later configuration only when the source schema actually recorded it;
- the fixed impairment order is now source WAV → optional gain/clipping → optional additive noise → G.711 → packet loss → jitter → ASR;
- default gain/clipping configuration preserves the earlier audio call path and does not invoke the transform.

### Compatibility

- published `v0.2.0` artifacts remain schema version 1 and published `v0.3.0` artifacts remain schema version 4;
- existing packet-loss, additive-noise, and jitter seed derivations are unchanged;
- WER, CER, RTF, speed-factor, numeric-entity, and existing dataset semantics are unchanged by gain/clipping integration;
- schemas 1 through 4 remain comparable without claiming dataset identity that those artifacts did not record;
- schemas 1 through 5 render gain/clipping fields as unavailable rather than assuming zero.

### Notes

- the streaming foundation is intentionally Python-only in this release: there is no streaming CLI command, serialized streaming artifact schema, or provider-specific streaming adapter yet;
- streaming tests use injected fake clocks and adapters; default CI adds no sleeps, network access, model downloads, or threads;
- GigaAM Multilingual integration remains tracked separately in issue #22 and is not part of v0.4.0.

## 0.3.0 - 2026-09-08

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
- GigaAM Multilingual integration is deferred from v0.3.0 and remains tracked separately in issue #22;
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
