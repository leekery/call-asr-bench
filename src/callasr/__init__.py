"""Public API for call-asr-bench."""

from callasr.audio import (
    AudioBuffer,
    apply_additive_noise,
    apply_gain_and_clip,
    resample,
    telephone_channel,
)
from callasr.codecs.g711 import decode_g711, encode_g711
from callasr.impairments import apply_jitter_loss, apply_packet_loss
from callasr.metrics.wer import (
    ErrorCounts,
    character_error_counts,
    character_error_rate,
    micro_average,
    normalize_text,
    word_error_counts,
    word_error_rate,
)

__all__ = [
    "AudioBuffer",
    "ErrorCounts",
    "apply_additive_noise",
    "apply_gain_and_clip",
    "apply_jitter_loss",
    "apply_packet_loss",
    "character_error_counts",
    "character_error_rate",
    "decode_g711",
    "encode_g711",
    "micro_average",
    "normalize_text",
    "resample",
    "telephone_channel",
    "word_error_counts",
    "word_error_rate",
]
