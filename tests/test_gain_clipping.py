from __future__ import annotations

import numpy as np
import pytest

from callasr.audio import AudioBuffer


def _api():
    try:
        from callasr.audio import apply_gain_and_clip
    except ImportError as exc:
        pytest.fail(f"gain/clipping implementation is missing: {exc}")
    return apply_gain_and_clip


@pytest.mark.parametrize(
    ("gain_db", "expected_factor"),
    [
        (6.0, 10.0 ** (6.0 / 20.0)),
        (-6.0, 10.0 ** (-6.0 / 20.0)),
    ],
)
def test_gain_db_uses_amplitude_conversion(gain_db: float, expected_factor: float) -> None:
    apply_gain_and_clip = _api()
    source = AudioBuffer(np.array([0.25, -0.5], dtype=np.float32), 16_000)

    result = apply_gain_and_clip(source, gain_db=gain_db)

    np.testing.assert_allclose(
        result.samples,
        source.samples * expected_factor,
        rtol=1e-6,
        atol=1e-7,
    )
    assert result.sample_rate == source.sample_rate
    assert result.samples.shape == source.samples.shape


def test_gain_is_applied_before_symmetric_hard_clipping() -> None:
    apply_gain_and_clip = _api()
    source = AudioBuffer(
        np.array([0.25, -0.25, 0.75, -0.75], dtype=np.float32),
        8_000,
    )

    result = apply_gain_and_clip(
        source,
        gain_db=6.020599913279624,
        clip_threshold=0.8,
    )

    np.testing.assert_allclose(
        result.samples,
        np.array([0.5, -0.5, 0.8, -0.8], dtype=np.float32),
        rtol=1e-6,
        atol=1e-7,
    )


def test_no_clip_threshold_does_not_implicitly_limit_to_unit_range() -> None:
    apply_gain_and_clip = _api()
    source = AudioBuffer(np.array([0.75, -0.75], dtype=np.float32), 16_000)

    result = apply_gain_and_clip(source, gain_db=6.020599913279624)

    assert result.samples[0] == pytest.approx(1.5)
    assert result.samples[1] == pytest.approx(-1.5)


def test_default_transform_is_noop_copy_and_input_remains_unchanged() -> None:
    apply_gain_and_clip = _api()
    samples = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    source = AudioBuffer(samples, 44_100)
    original = source.samples.copy()

    result = apply_gain_and_clip(source)

    np.testing.assert_array_equal(result.samples, original)
    np.testing.assert_array_equal(source.samples, original)
    assert result is not source
    assert not result.samples.flags.writeable
    assert result.sample_rate == 44_100


@pytest.mark.parametrize("gain_db", [float("nan"), float("inf"), float("-inf"), True])
def test_invalid_gain_is_rejected(gain_db: object) -> None:
    apply_gain_and_clip = _api()
    source = AudioBuffer(np.ones(4, dtype=np.float32), 8_000)

    with pytest.raises(ValueError, match="gain_db"):
        apply_gain_and_clip(source, gain_db=gain_db)


@pytest.mark.parametrize(
    "clip_threshold",
    [0.0, -0.1, float("nan"), float("inf"), float("-inf"), True],
)
def test_invalid_clip_threshold_is_rejected(clip_threshold: object) -> None:
    apply_gain_and_clip = _api()
    source = AudioBuffer(np.ones(4, dtype=np.float32), 8_000)

    with pytest.raises(ValueError, match="clip_threshold"):
        apply_gain_and_clip(source, clip_threshold=clip_threshold)


def test_gain_and_clip_is_exported_from_public_api() -> None:
    try:
        from callasr import apply_gain_and_clip as exported
    except ImportError as exc:
        pytest.fail(f"public gain/clipping export is missing: {exc}")
    from callasr.audio import apply_gain_and_clip as direct

    assert exported is direct
