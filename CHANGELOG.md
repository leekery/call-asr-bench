# Changelog

All notable changes to `call-asr-bench` are documented here.

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
