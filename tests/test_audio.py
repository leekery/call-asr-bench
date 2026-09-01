import numpy as np
import pytest

from callasr.audio import AudioBuffer, resample


def test_audio_buffer_owns_immutable_float32_samples() -> None:
    source = np.array([0.25, -0.5], dtype=np.float64)

    audio = AudioBuffer(source, sample_rate=16_000)
    source[0] = 1.0

    assert audio.samples.dtype == np.float32
    assert audio.samples.tolist() == pytest.approx([0.25, -0.5])
    assert not audio.samples.flags.writeable


@pytest.mark.parametrize(
    ("samples", "sample_rate", "message"),
    [
        (np.zeros((2, 2)), 16_000, "mono"),
        (np.array([], dtype=np.float32), 16_000, "empty"),
        (np.array([np.nan], dtype=np.float32), 16_000, "finite"),
        (np.array([0.0], dtype=np.float32), 0, "positive"),
    ],
)
def test_audio_buffer_rejects_invalid_audio(
    samples: np.ndarray, sample_rate: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AudioBuffer(samples, sample_rate)


def test_resample_changes_rate_and_expected_length() -> None:
    time = np.arange(1_600, dtype=np.float32) / 16_000
    source = AudioBuffer(np.sin(2 * np.pi * 440 * time), sample_rate=16_000)

    result = resample(source, target_sample_rate=8_000)

    assert result.sample_rate == 8_000
    assert result.samples.shape == (800,)
    assert result.samples.dtype == np.float32


def test_resample_is_deterministic() -> None:
    source = AudioBuffer(np.linspace(-1.0, 1.0, 257), sample_rate=44_100)

    first = resample(source, target_sample_rate=8_000)
    second = resample(source, target_sample_rate=8_000)

    np.testing.assert_array_equal(first.samples, second.samples)


def test_resample_rejects_non_positive_rate() -> None:
    source = AudioBuffer(np.zeros(10), sample_rate=16_000)

    with pytest.raises(ValueError, match="positive"):
        resample(source, target_sample_rate=-1)
