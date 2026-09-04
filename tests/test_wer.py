import pytest

from callasr import (
    ErrorCounts,
    character_error_counts,
    character_error_rate,
    micro_average,
    normalize_text,
    word_error_counts,
    word_error_rate,
)


def test_normalize_text_handles_russian_english_and_unicode() -> None:
    text = "  Привет, МИР!  Hello—WORLD. Номер \uff11\uff12\uff13 ₽  "

    assert normalize_text(text) == "привет мир hello world номер 123"


def test_word_error_rate_ignores_case_and_punctuation() -> None:
    assert word_error_rate("Добрый день, Alex!", "добрый ДЕНЬ alex") == 0.0


@pytest.mark.parametrize(
    ("hypothesis", "expected"),
    [
        ("one two five four", 0.25),
        ("one two four", 0.25),
        ("one two three extra four", 0.25),
    ],
)
def test_word_error_rate_counts_edit_operations(hypothesis: str, expected: float) -> None:
    assert word_error_rate("one two three four", hypothesis) == expected


def test_word_error_rate_supports_empty_transcripts() -> None:
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "one two") == 2.0
    assert word_error_rate("one two", "") == 1.0


def test_word_error_rate_can_exceed_one() -> None:
    assert word_error_rate("слово", "раз два три") == 3.0


def test_word_error_counts_exposes_edits_and_reference_units() -> None:
    counts = word_error_counts("one two three four", "one two five four")

    assert counts == ErrorCounts(edits=1, reference_units=4)
    assert counts.rate == 0.25


def test_character_error_rate_uses_normalized_text_without_spaces() -> None:
    counts = character_error_counts("A b!", "acb")

    assert counts == ErrorCounts(edits=1, reference_units=2)
    assert character_error_rate("A b!", "acb") == 0.5


def test_empty_reference_keeps_existing_denominator_contract() -> None:
    counts = word_error_counts("", "one two")

    assert counts == ErrorCounts(edits=2, reference_units=0)
    assert counts.rate == 2.0


def test_micro_average_weights_by_reference_units() -> None:
    counts = [ErrorCounts(1, 1), ErrorCounts(0, 9)]

    assert micro_average(counts) == pytest.approx(0.1)


def test_micro_average_counts_empty_reference_as_one_denominator_unit() -> None:
    counts = [ErrorCounts(2, 0), ErrorCounts(0, 2)]

    assert micro_average(counts) == pytest.approx(2 / 3)


def test_micro_average_of_empty_collection_is_zero() -> None:
    assert micro_average([]) == 0.0
