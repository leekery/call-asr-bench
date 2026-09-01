import numpy as np
import pytest

from callasr.codecs.g711 import decode_g711, encode_g711

PCM_VALUES = np.array([-32768, -32124, -1000, -1, 0, 1, 1000, 32124, 32767])
FLOAT_SAMPLES = PCM_VALUES.astype(np.float64) / 32768.0


@pytest.mark.parametrize(
    ("codec", "expected"),
    [
        ("pcmu", [0x00, 0x00, 0x4E, 0x7E, 0xFF, 0xFF, 0xCE, 0x80, 0x80]),
        ("pcma", [0x2A, 0x2A, 0x7A, 0x55, 0xD5, 0xD5, 0xFA, 0xAA, 0xAA]),
    ],
)
def test_encode_g711_matches_known_code_words(codec: str, expected: list[int]) -> None:
    payload = encode_g711(FLOAT_SAMPLES, codec=codec)

    assert payload.dtype == np.uint8
    assert payload.tolist() == expected


@pytest.mark.parametrize(
    ("codec", "payload", "decoded_pcm"),
    [
        ("pcmu", [0x00, 0x4E, 0x7E, 0xFF, 0xCE, 0x80], [-32124, -988, -8, 0, 988, 32124]),
        ("pcma", [0x2A, 0x7A, 0x55, 0xD5, 0xFA, 0xAA], [-32256, -1008, -8, 8, 1008, 32256]),
    ],
)
def test_decode_g711_matches_known_pcm_values(
    codec: str, payload: list[int], decoded_pcm: list[int]
) -> None:
    samples = decode_g711(np.array(payload, dtype=np.uint8), codec=codec)

    assert samples.dtype == np.float32
    np.testing.assert_array_equal(samples, np.array(decoded_pcm, dtype=np.float32) / 32768.0)


@pytest.mark.parametrize("codec", ["pcmu", "pcma"])
def test_g711_round_trip_preserves_speech_scale_waveform(codec: str) -> None:
    phase = np.linspace(0, 8 * np.pi, 8_000, endpoint=False)
    samples = (0.8 * np.sin(phase)).astype(np.float32)

    decoded = decode_g711(encode_g711(samples, codec=codec), codec=codec)

    assert decoded.shape == samples.shape
    assert np.mean(np.abs(decoded - samples)) < 0.01


@pytest.mark.parametrize("codec", ["pcmu", "pcma"])
def test_encode_g711_clips_samples_to_pcm_range(codec: str) -> None:
    clipped = encode_g711(np.array([-2.0, 2.0]), codec=codec)
    boundary = encode_g711(np.array([-1.0, 1.0]), codec=codec)

    np.testing.assert_array_equal(clipped, boundary)


def test_encode_g711_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="codec"):
        encode_g711(np.array([0.0]), codec="mp3")
    with pytest.raises(ValueError, match="one-dimensional"):
        encode_g711(np.zeros((2, 2)), codec="pcmu")
    with pytest.raises(ValueError, match="finite"):
        encode_g711(np.array([np.inf]), codec="pcmu")


def test_decode_g711_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="codec"):
        decode_g711(np.array([0xFF], dtype=np.uint8), codec="wav")
    with pytest.raises(ValueError, match="one-dimensional"):
        decode_g711(np.zeros((2, 2), dtype=np.uint8), codec="pcma")
    with pytest.raises(ValueError, match="uint8"):
        decode_g711(np.array([0xFF], dtype=np.int16), codec="pcmu")
