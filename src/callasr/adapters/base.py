"""Stable ASR adapter protocol used by benchmark runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from callasr.audio import AudioBuffer

AdapterOption = str | int | float | bool | None


class AdapterError(ValueError):
    """Raised for user-actionable ASR adapter configuration errors."""


@dataclass(frozen=True, slots=True)
class Transcription:
    """Immutable text returned by an ASR adapter."""

    text: str


@runtime_checkable
class ASRAdapter(Protocol):
    """Model-agnostic interface consumed by the benchmark runner."""

    name: str
    model: str
    device: str
    compute_type: str

    @property
    def decoding_options(self) -> dict[str, AdapterOption]: ...

    def transcribe(
        self,
        audio: AudioBuffer,
        language: str | None = None,
    ) -> Transcription: ...
