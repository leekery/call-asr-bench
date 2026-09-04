"""Optional faster-whisper integration."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from callasr.adapters.base import AdapterError, AdapterOption, Transcription
from callasr.audio import AudioBuffer, resample

_MODEL_SAMPLE_RATE = 16_000
_BEAM_SIZE = 5
_TEMPERATURE = 0.0


class FasterWhisperAdapter:
    """Run faster-whisper behind the stable call-asr-bench adapter boundary."""

    name = "faster-whisper"

    def __init__(
        self,
        model: str,
        *,
        device: str = "auto",
        compute_type: str = "default",
    ) -> None:
        try:
            faster_whisper = import_module("faster_whisper")
        except ModuleNotFoundError as exc:
            if exc.name != "faster_whisper":
                raise
            raise AdapterError(
                "faster-whisper is not installed; run `uv sync --extra faster-whisper`"
            ) from exc

        self.model = model
        self.device = device
        self.compute_type = compute_type
        self._model: Any = faster_whisper.WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
        )

    @property
    def decoding_options(self) -> dict[str, AdapterOption]:
        """Return deterministic decoding options recorded in benchmark metadata."""

        return {"beam_size": _BEAM_SIZE, "temperature": _TEMPERATURE}

    def transcribe(
        self,
        audio: AudioBuffer,
        language: str | None = None,
    ) -> Transcription:
        """Transcribe mono audio after converting it to faster-whisper's 16 kHz input."""

        model_audio = resample(audio, _MODEL_SAMPLE_RATE)
        segments, _ = self._model.transcribe(
            model_audio.samples,
            language=language,
            **self.decoding_options,
        )
        text = "".join(segment.text for segment in segments).strip()
        return Transcription(text=text)
