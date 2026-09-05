"""Representation-preserving scoring for digit-form critical entities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_PHONE_PATTERN = re.compile(r"(?<!\w)\+?(?:[0-9][\s()\-]*){6,}[0-9](?!\w)")
_NUMBER_PATTERN = re.compile(r"(?<!\w)[0-9]+(?!\w)")


@dataclass(frozen=True, slots=True)
class NumericEntity:
    """A digit-form entity with its original and comparison representations."""

    kind: Literal["phone", "number"]
    surface: str
    canonical: str


@dataclass(frozen=True, slots=True)
class NumericEntityScore:
    """Ordered exact-match score for reference and hypothesis entities."""

    reference: tuple[NumericEntity, ...]
    hypothesis: tuple[NumericEntity, ...]
    matches: int
    reference_count: int
    accuracy: float | None


def _canonical_phone(surface: str) -> str:
    stripped = surface.strip()
    prefix = "+" if stripped.startswith("+") else ""
    digits = "".join(character for character in stripped if "0" <= character <= "9")
    return prefix + digits


def extract_numeric_entities(text: str) -> tuple[NumericEntity, ...]:
    """Extract digit-form phone and numeric entities without word-to-digit conversion."""

    located: list[tuple[int, NumericEntity]] = []
    phone_ranges: list[tuple[int, int]] = []

    for match in _PHONE_PATTERN.finditer(text):
        surface = match.group(0).strip()
        phone_ranges.append(match.span())
        located.append(
            (
                match.start(),
                NumericEntity(
                    kind="phone",
                    surface=surface,
                    canonical=_canonical_phone(surface),
                ),
            )
        )

    for match in _NUMBER_PATTERN.finditer(text):
        start, end = match.span()
        if any(start < phone_end and end > phone_start for phone_start, phone_end in phone_ranges):
            continue
        surface = match.group(0)
        located.append(
            (
                start,
                NumericEntity(kind="number", surface=surface, canonical=surface),
            )
        )

    located.sort(key=lambda item: item[0])
    return tuple(entity for _, entity in located)


def _lcs_length(reference: tuple[str, ...], hypothesis: tuple[str, ...]) -> int:
    previous = [0] * (len(hypothesis) + 1)
    for reference_value in reference:
        current = [0]
        for hypothesis_index, hypothesis_value in enumerate(hypothesis, start=1):
            if reference_value == hypothesis_value:
                current.append(previous[hypothesis_index - 1] + 1)
            else:
                current.append(max(previous[hypothesis_index], current[-1]))
        previous = current
    return previous[-1]


def score_numeric_entities(reference: str, hypothesis: str) -> NumericEntityScore:
    """Score preservation of reference digit-form entities with ordered LCS matching."""

    reference_entities = extract_numeric_entities(reference)
    hypothesis_entities = extract_numeric_entities(hypothesis)
    matches = _lcs_length(
        tuple(entity.canonical for entity in reference_entities),
        tuple(entity.canonical for entity in hypothesis_entities),
    )
    reference_count = len(reference_entities)
    accuracy = None if reference_count == 0 else matches / reference_count
    return NumericEntityScore(
        reference=reference_entities,
        hypothesis=hypothesis_entities,
        matches=matches,
        reference_count=reference_count,
        accuracy=accuracy,
    )
