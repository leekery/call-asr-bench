"""ASR model adapters."""

from callasr.adapters.base import AdapterError, ASRAdapter, Transcription
from callasr.adapters.faster_whisper import FasterWhisperAdapter
from callasr.adapters.openai_compatible import OpenAICompatibleAdapter

__all__ = [
    "ASRAdapter",
    "AdapterError",
    "FasterWhisperAdapter",
    "OpenAICompatibleAdapter",
    "Transcription",
]
