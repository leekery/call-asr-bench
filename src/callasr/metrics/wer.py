"""Unicode-aware word error rate."""

from __future__ import annotations

import unicodedata


def normalize_text(text: str) -> str:
    """Normalize case, compatibility characters, punctuation, and whitespace."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    without_punctuation = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in normalized
    )
    return " ".join(without_punctuation.split())


def _word_edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous_row = list(range(len(hypothesis) + 1))
    for reference_index, reference_word in enumerate(reference, start=1):
        current_row = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis, start=1):
            deletion = previous_row[hypothesis_index] + 1
            insertion = current_row[hypothesis_index - 1] + 1
            substitution = previous_row[hypothesis_index - 1] + (reference_word != hypothesis_word)
            current_row.append(min(deletion, insertion, substitution))
        previous_row = current_row
    return previous_row[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return normalized word-level Levenshtein distance."""

    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()
    edit_distance = _word_edit_distance(reference_words, hypothesis_words)
    return edit_distance / max(1, len(reference_words))
