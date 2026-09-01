"""ITU-T G.711 PCMU and PCMA companding."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

Codec = Literal["pcmu", "pcma"]

_MU_LAW_SEGMENT_ENDS = np.array(
    [0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32
)
_A_LAW_SEGMENT_ENDS = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF], dtype=np.int32)


def _validated_codec(codec: str) -> Codec:
    if codec not in {"pcmu", "pcma"}:
        raise ValueError("codec must be 'pcmu' or 'pcma'")
    return cast(Codec, codec)


def _float_to_pcm16(samples: ArrayLike) -> NDArray[np.int32]:
    waveform = np.asarray(samples)
    if waveform.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    if not np.isfinite(waveform).all():
        raise ValueError("samples must contain only finite values")

    scaled = np.rint(np.clip(waveform, -1.0, 1.0) * 32768.0)
    return np.clip(scaled, -32768, 32767).astype(np.int32)


def _encode_mu_law(pcm: NDArray[np.int32]) -> NDArray[np.uint8]:
    pcm_14_bit = pcm >> 2
    negative = pcm_14_bit < 0
    magnitude = np.where(negative, -pcm_14_bit, pcm_14_bit)
    magnitude = np.minimum(magnitude, 8159) + 33

    segment = np.searchsorted(_MU_LAW_SEGMENT_ENDS, magnitude, side="left")
    shift = np.minimum(segment, 7) + 1
    mantissa = (magnitude >> shift) & 0x0F
    value = (segment << 4) | mantissa
    value = np.where(segment >= 8, 0x7F, value)
    mask = np.where(negative, 0x7F, 0xFF)
    return (value ^ mask).astype(np.uint8)


def _decode_mu_law(payload: NDArray[np.uint8]) -> NDArray[np.int32]:
    value = (~payload.astype(np.int32)) & 0xFF
    magnitude = ((value & 0x0F) << 3) + 0x84
    magnitude <<= (value & 0x70) >> 4
    return np.where(value & 0x80, 0x84 - magnitude, magnitude - 0x84)


def _encode_a_law(pcm: NDArray[np.int32]) -> NDArray[np.uint8]:
    pcm_13_bit = pcm >> 3
    negative = pcm_13_bit < 0
    magnitude = np.where(negative, -pcm_13_bit - 1, pcm_13_bit)

    segment = np.searchsorted(_A_LAW_SEGMENT_ENDS, magnitude, side="left")
    small_mantissa = (magnitude >> 1) & 0x0F
    large_mantissa = (magnitude >> np.maximum(segment, 1)) & 0x0F
    mantissa = np.where(segment < 2, small_mantissa, large_mantissa)
    value = (segment << 4) | mantissa
    mask = np.where(negative, 0x55, 0xD5)
    return (value ^ mask).astype(np.uint8)


def _decode_a_law(payload: NDArray[np.uint8]) -> NDArray[np.int32]:
    value = payload.astype(np.int32) ^ 0x55
    segment = (value & 0x70) >> 4
    magnitude = (value & 0x0F) << 4
    magnitude = np.where(segment == 0, magnitude + 8, magnitude + 0x108)
    magnitude = np.where(segment > 1, magnitude << (segment - 1), magnitude)
    return np.where(value & 0x80, magnitude, -magnitude)


def encode_g711(samples: ArrayLike, codec: str) -> NDArray[np.uint8]:
    """Encode normalized mono samples as a PCMU or PCMA byte array."""

    selected_codec = _validated_codec(codec)
    pcm = _float_to_pcm16(samples)
    if selected_codec == "pcmu":
        return _encode_mu_law(pcm)
    return _encode_a_law(pcm)


def decode_g711(payload: ArrayLike, codec: str) -> NDArray[np.float32]:
    """Decode a PCMU or PCMA byte array into normalized mono samples."""

    selected_codec = _validated_codec(codec)
    encoded = np.asarray(payload)
    if encoded.ndim != 1:
        raise ValueError("payload must be one-dimensional")
    if encoded.dtype != np.uint8:
        raise ValueError("payload must have dtype uint8")

    pcm = _decode_mu_law(encoded) if selected_codec == "pcmu" else _decode_a_law(encoded)
    return (pcm.astype(np.float32) / 32768.0).astype(np.float32, copy=False)
