from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from callasr.adapters.base import Transcription
from callasr.audio import AudioBuffer
from callasr.benchmark import (
    AdapterInfo,
    BenchmarkResult,
    BenchmarkSummary,
    ChannelInfo,
    DatasetInfo,
    ItemResult,
    result_to_dict,
    run_benchmark,
)
from callasr.dataset import DatasetItem
from callasr.metrics.wer import character_error_rate, word_error_rate


def _entities_api():
    try:
        from callasr.metrics.entities import (
            NumericEntity,
            NumericEntityScore,
            extract_numeric_entities,
            score_numeric_entities,
        )
    except ImportError as exc:
        pytest.fail(f"numeric-entity metric implementation is missing: {exc}")
    return NumericEntity, NumericEntityScore, extract_numeric_entities, score_numeric_entities


class FakeAdapter:
    name = "fake"
    model = "fake-model"
    device = "cpu"
    compute_type = "float32"
    decoding_options: ClassVar[dict[str, int]] = {"beam_size": 1}

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)

    def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcription:
        return Transcription(next(self.outputs))


def _item(tmp_path: Path, index: int, reference: str, language: str) -> DatasetItem:
    audio_path = tmp_path / f"audio-{index}.wav"
    audio_path.touch()
    return DatasetItem(
        id=f"id-{index}",
        audio=audio_path,
        reference=reference,
        language=language,
        line_number=index + 1,
    )


def test_extracts_russian_phone_and_generic_digit_entities() -> None:
    _, _, extract_numeric_entities, _ = _entities_api()

    entities = extract_numeric_entities("мой номер +7 (916) 123-45-67, код 0042")

    assert [(item.kind, item.surface, item.canonical) for item in entities] == [
        ("phone", "+7 (916) 123-45-67", "+79161234567"),
        ("number", "0042", "0042"),
    ]


def test_extracts_english_phone_and_number_in_transcript_order() -> None:
    _, _, extract_numeric_entities, _ = _entities_api()

    entities = extract_numeric_entities("call +1 (415) 555-2671 and enter 73")

    assert [(item.kind, item.canonical) for item in entities] == [
        ("phone", "+14155552671"),
        ("number", "73"),
    ]


def test_spoken_number_words_are_not_silently_converted() -> None:
    _, _, extract_numeric_entities, _ = _entities_api()

    russian = extract_numeric_entities(
        "плюс семь девятьсот шестнадцать сто двадцать три сорок пять шестьдесят семь"
    )
    english = extract_numeric_entities("plus one four one five five five five two six seven one")

    assert russian == ()
    assert english == ()


def test_phone_formatting_differences_match_after_canonicalization() -> None:
    _, _, _, score_numeric_entities = _entities_api()

    score = score_numeric_entities(
        "номер +7 (916) 123-45-67",
        "номер +7 916 123 45 67",
    )

    assert score.matches == 1
    assert score.reference_count == 1
    assert score.accuracy == 1.0
    assert score.reference[0].canonical == "+79161234567"
    assert score.hypothesis[0].canonical == "+79161234567"


def test_digit_reference_and_spoken_hypothesis_are_an_explicit_miss() -> None:
    _, _, _, score_numeric_entities = _entities_api()

    score = score_numeric_entities(
        "номер +7 (916) 123-45-67",
        "номер плюс семь девятьсот шестнадцать сто двадцать три сорок пять шестьдесят семь",
    )

    assert score.reference_count == 1
    assert score.matches == 0
    assert score.accuracy == 0.0
    assert score.hypothesis == ()


def test_ordered_lcs_alignment_tolerates_hypothesis_insertions() -> None:
    _, _, _, score_numeric_entities = _entities_api()

    score = score_numeric_entities("codes 123 and 456", "codes 999 then 123 and 456")

    assert [item.canonical for item in score.reference] == ["123", "456"]
    assert [item.canonical for item in score.hypothesis] == ["999", "123", "456"]
    assert score.matches == 2
    assert score.reference_count == 2
    assert score.accuracy == 1.0


def test_ordered_alignment_penalizes_reordered_reference_entities() -> None:
    _, _, _, score_numeric_entities = _entities_api()

    score = score_numeric_entities("codes 123 and 456", "codes 456 and 123")

    assert score.matches == 1
    assert score.reference_count == 2
    assert score.accuracy == 0.5


def test_leading_zeroes_are_semantic_in_generic_digit_entities() -> None:
    _, _, _, score_numeric_entities = _entities_api()

    exact = score_numeric_entities("код 0042", "code 0042")
    changed = score_numeric_entities("код 0042", "code 42")

    assert exact.accuracy == 1.0
    assert changed.accuracy == 0.0


def test_no_reference_entities_has_null_accuracy_and_keeps_hypothesis_extras() -> None:
    _, _, _, score_numeric_entities = _entities_api()

    score = score_numeric_entities("no critical number here", "hallucinated 42")

    assert score.reference == ()
    assert [item.canonical for item in score.hypothesis] == ["42"]
    assert score.matches == 0
    assert score.reference_count == 0
    assert score.accuracy is None


def test_current_result_schema_is_v4_and_entity_fields_are_json_ready() -> None:
    NumericEntity, NumericEntityScore, _, _ = _entities_api()
    entity_score = NumericEntityScore(
        reference=(NumericEntity(kind="number", surface="0042", canonical="0042"),),
        hypothesis=(NumericEntity(kind="number", surface="0043", canonical="0043"),),
        matches=0,
        reference_count=1,
        accuracy=0.0,
    )
    result = BenchmarkResult(
        created_at="2026-09-05T20:00:00+00:00",
        dataset=DatasetInfo(path="/tmp/dataset.jsonl", item_count=1),
        adapter=AdapterInfo(
            name="fake",
            model="fake-model",
            device="cpu",
            compute_type="float32",
            decoding_options={"beam_size": 1},
        ),
        channel=ChannelInfo(
            codec="none",
            packet_loss_rate=0.0,
            frame_duration_ms=20,
            seed=0,
        ),
        summary=BenchmarkSummary(
            total_audio_seconds=1.0,
            adapter_seconds=0.1,
            wer=0.0,
            cer=0.0,
            rtf=0.1,
            speed_factor=10.0,
            numeric_entity_matches=0,
            numeric_entity_reference_count=1,
            numeric_entity_accuracy=0.0,
        ),
        items=(
            ItemResult(
                id="id-0",
                audio="audio.wav",
                reference="code 0042",
                hypothesis="code 0043",
                language="en",
                audio_seconds=1.0,
                adapter_seconds=0.1,
                wer=0.5,
                cer=0.25,
                numeric_entities=entity_score,
            ),
        ),
    )

    payload = result_to_dict(result)

    assert payload["schema_version"] == 4
    assert payload["summary"]["numeric_entity_reference_count"] == 1
    assert payload["summary"]["numeric_entity_accuracy"] == 0.0
    assert payload["items"][0]["numeric_entities"]["reference"][0] == {
        "kind": "number",
        "surface": "0042",
        "canonical": "0042",
    }
    assert payload["items"][0]["numeric_entities"]["hypothesis"][0]["canonical"] == "0043"


def test_runner_reports_per_item_and_micro_numeric_entity_accuracy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    items = (
        _item(tmp_path, 0, "номер +7 (916) 123-45-67", "ru"),
        _item(tmp_path, 1, "enter code 0042", "en"),
        _item(tmp_path, 2, "no entity", "en"),
    )
    monkeypatch.setattr("callasr.benchmark.load_dataset_manifest", lambda path: items)
    monkeypatch.setattr(
        "callasr.benchmark.load_wav",
        lambda path: AudioBuffer(np.zeros(8_000, dtype=np.float32), 8_000),
    )
    monkeypatch.setattr("callasr.benchmark.perf_counter", lambda: 1.0)
    adapter = FakeAdapter(
        [
            "номер +7 916 123 45 67",
            "enter code 0043",
            "hallucinated 99",
        ]
    )

    result = run_benchmark(tmp_path / "dataset.jsonl", adapter, codec="none")

    assert result.items[0].numeric_entities is not None
    assert result.items[0].numeric_entities.accuracy == 1.0
    assert result.items[1].numeric_entities is not None
    assert result.items[1].numeric_entities.accuracy == 0.0
    assert result.items[2].numeric_entities is not None
    assert result.items[2].numeric_entities.accuracy is None
    assert result.summary.numeric_entity_matches == 1
    assert result.summary.numeric_entity_reference_count == 2
    assert result.summary.numeric_entity_accuracy == 0.5


def test_entity_scoring_does_not_change_existing_wer_or_cer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reference = "call code 0042 now"
    hypothesis = "call code 0043 now"
    item = _item(tmp_path, 0, reference, "en")
    monkeypatch.setattr("callasr.benchmark.load_dataset_manifest", lambda path: (item,))
    monkeypatch.setattr(
        "callasr.benchmark.load_wav",
        lambda path: AudioBuffer(np.zeros(8_000, dtype=np.float32), 8_000),
    )
    monkeypatch.setattr("callasr.benchmark.perf_counter", lambda: 1.0)

    result = run_benchmark(
        tmp_path / "dataset.jsonl",
        FakeAdapter([hypothesis]),
        codec="none",
    )

    assert result.items[0].wer == word_error_rate(reference, hypothesis)
    assert result.items[0].cer == character_error_rate(reference, hypothesis)


def test_numeric_entity_api_is_exported_from_top_level_package() -> None:
    try:
        from callasr import extract_numeric_entities, score_numeric_entities
    except ImportError as exc:
        pytest.fail(f"public numeric-entity API is missing: {exc}")
    _, _, direct_extract, direct_score = _entities_api()

    assert extract_numeric_entities is direct_extract
    assert score_numeric_entities is direct_score
