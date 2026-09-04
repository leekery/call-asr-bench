"""WAV decoding into benchmark audio buffers."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from callasr.audio import AudioBuffer

_SUPPORTED_SAMPLE_WIDTHS = frozenset({1, 2, 3, 4})


class AudioError(ValueError):
    """An audio input error with path context."""

    def __init__(self, path: Path, message: str) -> None:
        self.path = path
        super().__init__(f"{path}: {message}")


def _decode_pcm(payload: bytes, sample_width: int) -> NDArray[np.float32]:
    if sample_width == 1:
        values = np.frombuffer(payload, dtype=np.uint8).astype(np.float32)
        return (values - 128.0) / 128.0

    if sample_width == 2:
        values = np.frombuffer(payload, dtype="<i2").astype(np.float32)
        return values / 32_768.0

    if sample_width == 3:
        octets = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8_388_608.0

    values = np.frombuffer(payload, dtype="<i4").astype(np.float64)
    return (values / 2_147_483_648.0).astype(np.float32)


def load_wav(path: str | Path) -> AudioBuffer:
    """Load an uncompressed mono integer-PCM WAV without resampling it."""

    audio_path = Path(path).expanduser().resolve()
    try:
        with wave.open(str(audio_path), "rb") as reader:
            if reader.getcomptype() != "NONE":
                raise AudioError(audio_path, "only uncompressed PCM WAV is supported")

            channels = reader.getnchannels()
            if channels != 1:
                raise AudioError(audio_path, "WAV audio must be mono")

            sample_width = reader.getsampwidth()
            if sample_width not in _SUPPORTED_SAMPLE_WIDTHS:
                raise AudioError(
                    audio_path,
                    f"unsupported PCM sample width: {sample_width} bytes",
                )

            sample_rate = reader.getframerate()
            frame_count = reader.getnframes()
            if frame_count == 0:
                raise AudioError(audio_path, "WAV audio must not be empty")

            payload = reader.readframes(frame_count)
    except AudioError:
        raise
    except wave.Error as exc:
        raise AudioError(
            audio_path,
            f"only uncompressed PCM WAV is supported ({exc})",
        ) from exc
    except OSError as exc:
        raise AudioError(audio_path, f"cannot read WAV: {exc}") from exc

    expected_bytes = frame_count * sample_width
    if len(payload) != expected_bytes:
        raise AudioError(audio_path, "WAV data is truncated")

    samples = _decode_pcm(payload, sample_width)
    return AudioBuffer(samples, sample_rate)
