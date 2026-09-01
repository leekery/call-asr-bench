import pytest

from callasr import normalize_text, word_error_rate


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
