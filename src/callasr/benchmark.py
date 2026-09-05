"""Benchmark result schema and sequential runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np

from callasr.adapters.base import ASRAdapter
from callasr.audio import apply_additive_noise, telephone_channel
from callasr.dataset import load_dataset_manifest
from callasr.io import load_wav
from callasr.metrics.entities import NumericEntityScore, score_numeric_entities
from callasr.metrics.wer import character_error_counts, micro_average, word_error_counts

JsonScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    path: str
    item_count: int


@dataclass(frozen=True, slots=True)
class AdapterInfo:
    name: str
    model: str
    device: str
    compute_type: str
    decoding_options: dict[str, JsonScalar]


@dataclass(frozen=True, slots=True)
class ChannelInfo:
    codec: Literal["none", "pcmu", "pcma"]
    packet_loss_rate: float
    frame_duration_ms: int
    seed: int
    additive_noise_snr_db: float | None = None
    jitter_std_ms: float | None = None
    playout_buffer_ms: float | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    total_audio_seconds: float
    adapter_seconds: float
    wer: float
    cer: float
    rtf: float
    speed_factor: float | None
    numeric_entity_matches: int = 0
    numeric_entity_reference_count: int = 0
    numeric_entity_accuracy: float | None = None


@dataclass(frozen=True, slots=True)
class ItemResult:
    id: str
    audio: str
    reference: str
    hypothesis: str
    language: str | None
    audio_seconds: float
    adapter_seconds: float
    wer: float
    cer: float
    numeric_entities: NumericEntityScore | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    schema_version: int = field(default=4, init=False)
    created_at: str
    dataset: DatasetInfo
    adapter: AdapterInfo
    channel: ChannelInfo
    summary: BenchmarkSummary
    items: tuple[ItemResult, ...]


def result_to_dict(result: BenchmarkResult) -> dict[str, object]:
    """Convert a benchmark result into a JSON-ready mapping."""

    payload = asdict(result)
    payload["items"] = list(payload["items"])
    return payload


def _packet_loss_seed(run_seed: int, item_index: int) -> int:
    sequence = np.random.SeedSequence([run_seed, item_index])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _additive_noise_seed(run_seed: int, item_index: int) -> int:
    sequence = np.random.SeedSequence([run_seed, item_index, 1])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _jitter_seed(run_seed: int, item_index: int) -> int:
    sequence = np.random.SeedSequence([run_seed, item_index, 2])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _validate_jitter_configuration(
    codec: Literal["none", "pcmu", "pcma"],
    jitter_std_ms: float | None,
    playout_buffer_ms: float | None,
) -> None:
    if (jitter_std_ms is None) != (playout_buffer_ms is None):
        raise ValueError("jitter_std_ms and playout_buffer_ms must be provided together")
    if jitter_std_ms is not None and codec == "none":
        raise ValueError("jitter requires codec 'pcmu' or 'pcma'")


def _artifact_audio_path(audio_path: Path, manifest_path: Path) -> str:
    try:
        return str(audio_path.relative_to(manifest_path.parent))
    except ValueError:
        return str(audio_path)


def run_benchmark(
    manifest_path: str | Path,
    adapter: ASRAdapter,
    *,
    codec: Literal["none", "pcmu", "pcma"] = "none",
    packet_loss_rate: float = 0.0,
    frame_duration_ms: int = 20,
    snr_db: float | None = None,
    jitter_std_ms: float | None = None,
    playout_buffer_ms: float | None = None,
    seed: int = 0,
) -> BenchmarkResult:
    """Run a validated dataset sequentially through an injected ASR adapter."""

    _validate_jitter_configuration(codec, jitter_std_ms, playout_buffer_ms)
    resolved_manifest = Path(manifest_path).expanduser().resolve()
    dataset_items = load_dataset_manifest(resolved_manifest)
    item_results: list[ItemResult] = []
    word_counts = []
    character_counts = []
    total_audio_seconds = 0.0
    total_adapter_seconds = 0.0
    numeric_entity_matches = 0
    numeric_entity_reference_count = 0

    for item_index, item in enumerate(dataset_items):
        audio = load_wav(item.audio)
        if snr_db is not None:
            audio = apply_additive_noise(
                audio,
                snr_db=snr_db,
                seed=_additive_noise_seed(seed, item_index),
            )
        if codec != "none":
            packet_seed = _packet_loss_seed(seed, item_index)
            if jitter_std_ms is None:
                audio = telephone_channel(
                    audio,
                    codec=codec,
                    packet_loss_rate=packet_loss_rate,
                    frame_duration_ms=frame_duration_ms,
                    seed=packet_seed,
                )
            else:
                audio = telephone_channel(
                    audio,
                    codec=codec,
                    packet_loss_rate=packet_loss_rate,
                    frame_duration_ms=frame_duration_ms,
                    seed=packet_seed,
                    jitter_std_ms=jitter_std_ms,
                    playout_buffer_ms=playout_buffer_ms,
                    jitter_seed=_jitter_seed(seed, item_index),
                )

        audio_seconds = audio.samples.size / audio.sample_rate
        started_at = perf_counter()
        transcription = adapter.transcribe(audio, language=item.language)
        adapter_seconds = perf_counter() - started_at

        item_word_counts = word_error_counts(item.reference, transcription.text)
        item_character_counts = character_error_counts(item.reference, transcription.text)
        numeric_entities = score_numeric_entities(item.reference, transcription.text)
        word_counts.append(item_word_counts)
        character_counts.append(item_character_counts)
        numeric_entity_matches += numeric_entities.matches
        numeric_entity_reference_count += numeric_entities.reference_count
        total_audio_seconds += audio_seconds
        total_adapter_seconds += adapter_seconds

        item_results.append(
            ItemResult(
                id=item.id,
                audio=_artifact_audio_path(item.audio, resolved_manifest),
                reference=item.reference,
                hypothesis=transcription.text,
                language=item.language,
                audio_seconds=audio_seconds,
                adapter_seconds=adapter_seconds,
                wer=item_word_counts.rate,
                cer=item_character_counts.rate,
                numeric_entities=numeric_entities,
            )
        )

    rtf = total_adapter_seconds / total_audio_seconds if total_audio_seconds else 0.0
    speed_factor = (
        None if total_adapter_seconds == 0.0 else total_audio_seconds / total_adapter_seconds
    )
    numeric_entity_accuracy = (
        None
        if numeric_entity_reference_count == 0
        else numeric_entity_matches / numeric_entity_reference_count
    )
    return BenchmarkResult(
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset=DatasetInfo(path=str(resolved_manifest), item_count=len(dataset_items)),
        adapter=AdapterInfo(
            name=adapter.name,
            model=adapter.model,
            device=adapter.device,
            compute_type=adapter.compute_type,
            decoding_options=dict(adapter.decoding_options),
        ),
        channel=ChannelInfo(
            codec=codec,
            packet_loss_rate=packet_loss_rate,
            frame_duration_ms=frame_duration_ms,
            seed=seed,
            additive_noise_snr_db=snr_db,
            jitter_std_ms=jitter_std_ms,
            playout_buffer_ms=playout_buffer_ms,
        ),
        summary=BenchmarkSummary(
            total_audio_seconds=total_audio_seconds,
            adapter_seconds=total_adapter_seconds,
            wer=micro_average(word_counts),
            cer=micro_average(character_counts),
            rtf=rtf,
            speed_factor=speed_factor,
            numeric_entity_matches=numeric_entity_matches,
            numeric_entity_reference_count=numeric_entity_reference_count,
            numeric_entity_accuracy=numeric_entity_accuracy,
        ),
        items=tuple(item_results),
    )
