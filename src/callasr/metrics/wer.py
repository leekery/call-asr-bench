"""Unicode-aware word and character error rates."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ErrorCounts:
    """Levenshtein edits paired with the raw reference-unit count."""

    edits: int
    reference_units: int

    @property
    def denominator(self) -> int:
        """Return the effective denominator used by the metric contract."""

        return max(1, self.reference_units)

    @property
    def rate(self) -> float:
        """Return the normalized error rate for these counts."""

        return self.edits / self.denominator


def normalize_text(text: str) -> str:
    """Normalize case, compatibility characters, punctuation, and whitespace."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    without_punctuation = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in normalized
    )
    return " ".join(without_punctuation.split())


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous_row = list(range(len(hypothesis) + 1))
    for reference_index, reference_unit in enumerate(reference, start=1):
        current_row = [reference_index]
        for hypothesis_index, hypothesis_unit in enumerate(hypothesis, start=1):
            deletion = previous_row[hypothesis_index] + 1
            insertion = current_row[hypothesis_index - 1] + 1
            substitution = previous_row[hypothesis_index - 1] + (reference_unit != hypothesis_unit)
            current_row.append(min(deletion, insertion, substitution))
        previous_row = current_row
    return previous_row[-1]


def word_error_counts(reference: str, hypothesis: str) -> ErrorCounts:
    """Return word-level edit and reference-unit counts."""

    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()
    return ErrorCounts(
        edits=_edit_distance(reference_words, hypothesis_words),
        reference_units=len(reference_words),
    )


def character_error_counts(reference: str, hypothesis: str) -> ErrorCounts:
    """Return character-level counts after normalization and space removal."""

    reference_characters = list(normalize_text(reference).replace(" ", ""))
    hypothesis_characters = list(normalize_text(hypothesis).replace(" ", ""))
    return ErrorCounts(
        edits=_edit_distance(reference_characters, hypothesis_characters),
        reference_units=len(reference_characters),
    )


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return normalized word-level Levenshtein distance."""

    return word_error_counts(reference, hypothesis).rate


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Return normalized character-level Levenshtein distance."""

    return character_error_counts(reference, hypothesis).rate


def micro_average(counts: Iterable[ErrorCounts]) -> float:
    """Micro-average error counts using each item's effective denominator."""

    total_edits = 0
    total_denominator = 0
    for item in counts:
        total_edits += item.edits
        total_denominator += item.denominator
    if total_denominator == 0:
        return 0.0
    return total_edits / total_denominator
