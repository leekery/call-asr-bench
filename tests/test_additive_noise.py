from __future__ import annotations

import numpy as np
import pytest

from callasr.audio import AudioBuffer


def _api():
    try:
        from callasr.audio import apply_additive_noise
    except ImportError as exc:
        pytest.fail(f"additive-noise implementation is missing: {exc}")
    return apply_additive_noise


def _signal() -> AudioBuffer:
    time = np.arange(16_000, dtype=np.float64) / 16_000.0
    samples = 0.1 * np.sin(2.0 * np.pi * 440.0 * time)
    return AudioBuffer(samples, 16_000)


def test_same_seed_is_exactly_deterministic_and_input_is_unchanged() -> None:
    apply_additive_noise = _api()
    source = _signal()
    before = source.samples.copy()

    first = apply_additive_noise(source, snr_db=20.0, seed=42)
    second = apply_additive_noise(source, snr_db=20.0, seed=42)

    assert np.array_equal(first.samples, second.samples)
    assert np.array_equal(source.samples, before)
    assert first.sample_rate == source.sample_rate
    assert first.samples.shape == source.samples.shape
    assert not first.samples.flags.writeable


def test_different_seeds_produce_different_noise() -> None:
    apply_additive_noise = _api()
    source = _signal()

    first = apply_additive_noise(source, snr_db=10.0, seed=1)
    second = apply_additive_noise(source, snr_db=10.0, seed=2)

    assert not np.array_equal(first.samples, second.samples)


@pytest.mark.parametrize("snr_db", [20.0, 0.0, -5.0])
def test_measured_snr_matches_requested_value(snr_db: float) -> None:
    apply_additive_noise = _api()
    source = _signal()

    impaired = apply_additive_noise(source, snr_db=snr_db, seed=7)
    signal = source.samples.astype(np.float64)
    noise = impaired.samples.astype(np.float64) - signal
    signal_rms = np.sqrt(np.mean(np.square(signal)))
    noise_rms = np.sqrt(np.mean(np.square(noise)))
    measured = 20.0 * np.log10(signal_rms / noise_rms)

    assert measured == pytest.approx(snr_db, abs=1e-3)


@pytest.mark.parametrize("snr_db", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_snr_is_rejected(snr_db: float) -> None:
    apply_additive_noise = _api()
    with pytest.raises(ValueError, match="snr_db must be finite"):
        apply_additive_noise(_signal(), snr_db=snr_db)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_seed_is_rejected(seed: object) -> None:
    apply_additive_noise = _api()
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        apply_additive_noise(_signal(), snr_db=20.0, seed=seed)  # type: ignore[arg-type]


def test_silent_input_is_rejected() -> None:
    apply_additive_noise = _api()
    silent = AudioBuffer(np.zeros(160, dtype=np.float32), 8_000)

    with pytest.raises(ValueError, match="cannot define SNR for silent audio"):
        apply_additive_noise(silent, snr_db=10.0)


def test_additive_noise_is_exported_from_public_api() -> None:
    try:
        from callasr import apply_additive_noise as exported
    except ImportError as exc:
        pytest.fail(f"public additive-noise export is missing: {exc}")
    from callasr.audio import apply_additive_noise as direct

    assert exported is direct
