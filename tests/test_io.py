import wave
from pathlib import Path

import numpy as np
import pytest


def _pcm_bytes(sample_width: int) -> tuple[bytes, list[float]]:
    if sample_width == 1:
        return bytes([0, 128, 255]), [-1.0, 0.0, 127.0 / 128.0]
    if sample_width == 2:
        values = np.array([-32768, 0, 32767], dtype="<i2")
        return values.tobytes(), [-1.0, 0.0, 32767.0 / 32768.0]
    if sample_width == 3:
        values = (-8_388_608, 0, 8_388_607)
        payload = b"".join(value.to_bytes(3, "little", signed=True) for value in values)
        return payload, [-1.0, 0.0, 8_388_607.0 / 8_388_608.0]
    if sample_width == 4:
        values = np.array([-2_147_483_648, 0, 2_147_483_647], dtype="<i4")
        return values.tobytes(), [-1.0, 0.0, 2_147_483_647.0 / 2_147_483_648.0]
    raise AssertionError("unsupported test width")


def _write_wav(
    path: Path,
    payload: bytes,
    *,
    sample_width: int,
    channels: int = 1,
    sample_rate: int = 16_000,
) -> Path:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        writer.writeframes(payload)
    return path


@pytest.mark.parametrize("sample_width", [1, 2, 3, 4])
def test_load_wav_normalizes_supported_integer_pcm_widths(
    tmp_path: Path, sample_width: int
) -> None:
    from callasr.io import load_wav

    payload, expected = _pcm_bytes(sample_width)
    path = _write_wav(tmp_path / f"pcm{sample_width}.wav", payload, sample_width=sample_width)

    audio = load_wav(path)

    assert audio.sample_rate == 16_000
    assert audio.samples.dtype == np.float32
    assert not audio.samples.flags.writeable
    np.testing.assert_allclose(audio.samples, expected, rtol=0.0, atol=1e-7)


def test_load_wav_preserves_source_sample_rate(tmp_path: Path) -> None:
    from callasr.io import load_wav

    payload, _ = _pcm_bytes(2)
    path = _write_wav(
        tmp_path / "sample.wav",
        payload,
        sample_width=2,
        sample_rate=44_100,
    )

    audio = load_wav(path)

    assert audio.sample_rate == 44_100
    assert audio.samples.shape == (3,)


def test_load_wav_rejects_stereo_audio(tmp_path: Path) -> None:
    from callasr.io import AudioError, load_wav

    stereo = np.array([-100, 100, -200, 200], dtype="<i2").tobytes()
    path = _write_wav(
        tmp_path / "stereo.wav",
        stereo,
        sample_width=2,
        channels=2,
    )

    with pytest.raises(AudioError, match="mono") as exc_info:
        load_wav(path)

    assert str(path.resolve()) in str(exc_info.value)


def test_load_wav_rejects_compressed_encoding(tmp_path: Path) -> None:
    from callasr.io import AudioError, load_wav

    payload, _ = _pcm_bytes(2)
    path = _write_wav(tmp_path / "compressed.wav", payload, sample_width=2)
    wav_bytes = bytearray(path.read_bytes())
    wav_bytes[20:22] = (6).to_bytes(2, "little")
    path.write_bytes(wav_bytes)

    with pytest.raises(AudioError, match="uncompressed PCM"):
        load_wav(path)


def test_load_wav_rejects_empty_audio(tmp_path: Path) -> None:
    from callasr.io import AudioError, load_wav

    path = _write_wav(tmp_path / "empty.wav", b"", sample_width=2)

    with pytest.raises(AudioError, match="empty"):
        load_wav(path)
