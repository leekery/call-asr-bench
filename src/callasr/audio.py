"""Audio containers, deterministic resampling, and waveform impairments."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import resample_poly

from callasr.codecs.g711 import Codec, decode_g711, encode_g711
from callasr.impairments import apply_jitter_loss, apply_packet_loss

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


def apply_additive_noise(
    audio: AudioBuffer,
    *,
    snr_db: float,
    seed: int = 0,
) -> AudioBuffer:
    """Add deterministic Gaussian noise at a requested signal-to-noise ratio."""

    if not isinstance(snr_db, Real) or isinstance(snr_db, bool) or not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    signal = audio.samples.astype(np.float64)
    signal_rms = float(np.sqrt(np.mean(np.square(signal))))
    if signal_rms == 0.0:
        raise ValueError("cannot define SNR for silent audio")

    raw_noise = np.random.default_rng(int(seed)).standard_normal(signal.size)
    raw_noise_rms = float(np.sqrt(np.mean(np.square(raw_noise))))
    target_noise_rms = signal_rms * (10.0 ** (-float(snr_db) / 20.0))
    noise = raw_noise * (target_noise_rms / raw_noise_rms)
    return AudioBuffer(signal + noise, audio.sample_rate)


def apply_gain_and_clip(
    audio: AudioBuffer,
    *,
    gain_db: float = 0.0,
    clip_threshold: float | None = None,
) -> AudioBuffer:
    """Apply amplitude gain followed by optional symmetric hard clipping."""

    if not isinstance(gain_db, Real) or isinstance(gain_db, bool) or not np.isfinite(gain_db):
        raise ValueError("gain_db must be finite")
    if clip_threshold is not None and (
        not isinstance(clip_threshold, Real)
        or isinstance(clip_threshold, bool)
        or not np.isfinite(clip_threshold)
        or clip_threshold <= 0.0
    ):
        raise ValueError("clip_threshold must be a finite positive number")

    gain = 10.0 ** (float(gain_db) / 20.0)
    transformed = audio.samples.astype(np.float64) * gain
    if clip_threshold is not None:
        threshold = float(clip_threshold)
        transformed = np.clip(transformed, -threshold, threshold)
    return AudioBuffer(transformed, audio.sample_rate)


def telephone_channel(
    audio: AudioBuffer,
    codec: Codec = "pcmu",
    *,
    packet_loss_rate: float = 0.0,
    frame_duration_ms: int = 20,
    seed: int = 0,
    jitter_std_ms: float | None = None,
    playout_buffer_ms: float | None = None,
    jitter_seed: int = 0,
) -> AudioBuffer:
    """Apply 8 kHz G.711 transport with optional deterministic loss and jitter."""

    if (jitter_std_ms is None) != (playout_buffer_ms is None):
        raise ValueError("jitter_std_ms and playout_buffer_ms must be provided together")

    downsampled = resample(audio, TELEPHONE_SAMPLE_RATE)
    payload = encode_g711(downsampled.samples, codec)
    payload = apply_packet_loss(
        payload,
        codec=codec,
        loss_rate=packet_loss_rate,
        frame_duration_ms=frame_duration_ms,
        seed=seed,
    )
    if jitter_std_ms is not None and playout_buffer_ms is not None:
        payload = apply_jitter_loss(
            payload,
            codec=codec,
            jitter_std_ms=jitter_std_ms,
            playout_buffer_ms=playout_buffer_ms,
            frame_duration_ms=frame_duration_ms,
            seed=jitter_seed,
        )
    decoded = decode_g711(payload, codec)
    return AudioBuffer(decoded, TELEPHONE_SAMPLE_RATE)
