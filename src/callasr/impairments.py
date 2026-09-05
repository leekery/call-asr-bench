"""Deterministic network impairments for encoded telephone audio."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from callasr.codecs.g711 import Codec

_G711_SAMPLES_PER_MILLISECOND = 8
_SILENCE_BYTES: dict[str, int] = {"pcmu": 0xFF, "pcma": 0xD5}


def apply_packet_loss(
    payload: ArrayLike,
    *,
    codec: Codec,
    loss_rate: float,
    frame_duration_ms: int = 20,
    seed: int = 0,
) -> NDArray[np.uint8]:
    """Apply deterministic frame-level packet loss to a G.711 payload."""

    encoded = np.asarray(payload)
    if encoded.ndim != 1:
        raise ValueError("payload must be one-dimensional")
    if encoded.dtype != np.uint8:
        raise ValueError("payload must have dtype uint8")
    if codec not in _SILENCE_BYTES:
        raise ValueError("codec must be 'pcmu' or 'pcma'")
    if (
        not isinstance(loss_rate, Real)
        or isinstance(loss_rate, bool)
        or not np.isfinite(loss_rate)
        or not 0.0 <= loss_rate <= 1.0
    ):
        raise ValueError("loss_rate must be between 0.0 and 1.0")
    if (
        not isinstance(frame_duration_ms, Integral)
        or isinstance(frame_duration_ms, bool)
        or frame_duration_ms <= 0
    ):
        raise ValueError("frame_duration_ms must be a positive integer")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    impaired = np.array(encoded, dtype=np.uint8, copy=True)
    if impaired.size == 0 or loss_rate == 0.0:
        return impaired

    frame_size = int(frame_duration_ms) * _G711_SAMPLES_PER_MILLISECOND
    frame_count = (impaired.size + frame_size - 1) // frame_size
    dropped_frames = np.random.default_rng(int(seed)).random(frame_count) < loss_rate

    for frame_index in np.flatnonzero(dropped_frames):
        start = int(frame_index) * frame_size
        impaired[start : start + frame_size] = _SILENCE_BYTES[codec]

    return impaired


def apply_jitter_loss(
    payload: ArrayLike,
    *,
    codec: Codec,
    jitter_std_ms: float,
    playout_buffer_ms: float,
    frame_duration_ms: int = 20,
    seed: int = 0,
) -> NDArray[np.uint8]:
    """Replace G.711 frames that arrive beyond a deterministic jitter deadline."""

    encoded = np.asarray(payload)
    if encoded.ndim != 1:
        raise ValueError("payload must be one-dimensional")
    if encoded.dtype != np.uint8:
        raise ValueError("payload must have dtype uint8")
    if codec not in _SILENCE_BYTES:
        raise ValueError("codec must be 'pcmu' or 'pcma'")
    if (
        not isinstance(jitter_std_ms, Real)
        or isinstance(jitter_std_ms, bool)
        or not np.isfinite(jitter_std_ms)
        or jitter_std_ms < 0.0
    ):
        raise ValueError("jitter_std_ms must be a finite non-negative number")
    if (
        not isinstance(playout_buffer_ms, Real)
        or isinstance(playout_buffer_ms, bool)
        or not np.isfinite(playout_buffer_ms)
        or playout_buffer_ms < 0.0
    ):
        raise ValueError("playout_buffer_ms must be a finite non-negative number")
    if (
        not isinstance(frame_duration_ms, Integral)
        or isinstance(frame_duration_ms, bool)
        or frame_duration_ms <= 0
    ):
        raise ValueError("frame_duration_ms must be a positive integer")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    impaired = np.array(encoded, dtype=np.uint8, copy=True)
    if impaired.size == 0 or jitter_std_ms == 0.0:
        return impaired

    frame_size = int(frame_duration_ms) * _G711_SAMPLES_PER_MILLISECOND
    frame_count = (impaired.size + frame_size - 1) // frame_size
    delay_variation_ms = np.random.default_rng(int(seed)).normal(
        0.0,
        float(jitter_std_ms),
        frame_count,
    )
    late_frames = delay_variation_ms > float(playout_buffer_ms)

    for frame_index in np.flatnonzero(late_frames):
        start = int(frame_index) * frame_size
        impaired[start : start + frame_size] = _SILENCE_BYTES[codec]

    return impaired
