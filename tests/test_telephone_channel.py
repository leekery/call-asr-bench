import numpy as np

from callasr import AudioBuffer, decode_g711, encode_g711, resample, telephone_channel


def _speech_like_audio(sample_rate: int = 16_000) -> AudioBuffer:
    time = np.arange(sample_rate // 10, dtype=np.float32) / sample_rate
    samples = 0.6 * np.sin(2 * np.pi * 220 * time) + 0.2 * np.sin(2 * np.pi * 660 * time)
    return AudioBuffer(samples, sample_rate)


def test_telephone_channel_outputs_8khz_audio() -> None:
    result = telephone_channel(_speech_like_audio(), codec="pcmu")

    assert result.sample_rate == 8_000
    assert result.samples.shape == (800,)
    assert result.samples.dtype == np.float32


def test_telephone_channel_matches_explicit_pipeline() -> None:
    source = _speech_like_audio()

    result = telephone_channel(source, codec="pcma")
    downsampled = resample(source, 8_000)
    expected = decode_g711(encode_g711(downsampled.samples, "pcma"), "pcma")

    np.testing.assert_array_equal(result.samples, expected)


def test_telephone_channel_is_deterministic() -> None:
    source = _speech_like_audio(sample_rate=44_100)

    first = telephone_channel(source, codec="pcmu")
    second = telephone_channel(source, codec="pcmu")

    np.testing.assert_array_equal(first.samples, second.samples)


def test_pcmu_and_pcma_produce_distinct_channel_signals() -> None:
    source = _speech_like_audio()

    pcmu = telephone_channel(source, codec="pcmu")
    pcma = telephone_channel(source, codec="pcma")

    assert np.any(pcmu.samples != pcma.samples)
