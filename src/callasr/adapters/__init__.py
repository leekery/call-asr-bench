"""ASR model adapters."""

from callasr.adapters.base import AdapterError, ASRAdapter, Transcription
from callasr.adapters.faster_whisper import FasterWhisperAdapter

__all__ = ["ASRAdapter", "AdapterError", "FasterWhisperAdapter", "Transcription"]
