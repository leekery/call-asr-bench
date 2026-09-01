"""Audio containers and deterministic sample-rate conversion."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import resample_poly

from callasr.codecs.g711 import Codec, decode_g711, encode_g711

TELEPHONE_SAMPLE_RATE = 8_000


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    """An immutable mono waveform with an explicit sample rate."""

    samples: NDArray[np.float32]
    sample_rate: int

    def __init__(self, samples: ArrayLike, sample_rate: int) -> None:
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")

        waveform = np.asarray(samples)
        if waveform.ndim != 1:
            raise ValueError("samples must contain mono audio")
        if waveform.size == 0:
            raise ValueError("samples must not be empty")
        if not np.isfinite(waveform).all():
            raise ValueError("samples must contain only finite values")

        owned_samples = np.array(waveform, dtype=np.float32, copy=True)
        owned_samples.setflags(write=False)
        object.__setattr__(self, "samples", owned_samples)
        object.__setattr__(self, "sample_rate", sample_rate)


def resample(audio: AudioBuffer, target_sample_rate: int) -> AudioBuffer:
    """Resample mono audio with a deterministic polyphase filter."""

    if (
        not isinstance(target_sample_rate, int)
        or isinstance(target_sample_rate, bool)
        or target_sample_rate <= 0
    ):
        raise ValueError("target_sample_rate must be a positive integer")

    if target_sample_rate == audio.sample_rate:
        return AudioBuffer(audio.samples, audio.sample_rate)

    common_factor = gcd(audio.sample_rate, target_sample_rate)
    up = target_sample_rate // common_factor
    down = audio.sample_rate // common_factor
    converted = resample_poly(audio.samples, up, down)
    return AudioBuffer(converted, target_sample_rate)


def telephone_channel(audio: AudioBuffer, codec: Codec = "pcmu") -> AudioBuffer:
    """Apply 8 kHz resampling and a G.711 encode/decode round trip."""

    downsampled = resample(audio, TELEPHONE_SAMPLE_RATE)
    payload = encode_g711(downsampled.samples, codec)
    decoded = decode_g711(payload, codec)
    return AudioBuffer(decoded, TELEPHONE_SAMPLE_RATE)
