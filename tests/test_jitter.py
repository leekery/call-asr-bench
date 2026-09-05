from __future__ import annotations

import numpy as np
import pytest


def _api():
    try:
        from callasr.impairments import apply_jitter_loss
    except ImportError as exc:
        pytest.fail(f"jitter-loss implementation is missing: {exc}")
    return apply_jitter_loss


def _expected(
    payload: np.ndarray,
    *,
    codec: str,
    jitter_std_ms: float,
    playout_buffer_ms: float,
    frame_duration_ms: int,
    seed: int,
) -> np.ndarray:
    frame_size = frame_duration_ms * 8
    frame_count = (payload.size + frame_size - 1) // frame_size
    delays = np.random.default_rng(seed).normal(0.0, jitter_std_ms, frame_count)
    late_frames = delays > playout_buffer_ms
    silence = {"pcmu": 0xFF, "pcma": 0xD5}[codec]
    result = payload.copy()
    for frame_index in np.flatnonzero(late_frames):
        start = int(frame_index) * frame_size
        result[start : start + frame_size] = silence
    return result


def test_zero_jitter_is_noop_copy() -> None:
    apply_jitter_loss = _api()
    payload = np.arange(200, dtype=np.uint8)

    result = apply_jitter_loss(
        payload,
        codec="pcmu",
        jitter_std_ms=0.0,
        playout_buffer_ms=0.0,
        seed=42,
    )

    assert np.array_equal(result, payload)
    assert result is not payload
    result[0] = 123
    assert payload[0] == 0


def test_jitter_matches_gaussian_late_frame_contract_and_partial_tail() -> None:
    apply_jitter_loss = _api()
    frame_size = 20 * 8
    payload = np.arange(frame_size * 2 + 37, dtype=np.uint8)

    result = apply_jitter_loss(
        payload,
        codec="pcmu",
        jitter_std_ms=12.5,
        playout_buffer_ms=3.0,
        frame_duration_ms=20,
        seed=7,
    )

    expected = _expected(
        payload,
        codec="pcmu",
        jitter_std_ms=12.5,
        playout_buffer_ms=3.0,
        frame_duration_ms=20,
        seed=7,
    )
    assert np.array_equal(result, expected)
    assert result.shape == payload.shape


def test_pcma_uses_alaw_silence_byte() -> None:
    apply_jitter_loss = _api()
    payload = np.full(160 * 16, 0x11, dtype=np.uint8)

    result = apply_jitter_loss(
        payload,
        codec="pcma",
        jitter_std_ms=10.0,
        playout_buffer_ms=0.0,
        frame_duration_ms=20,
        seed=1,
    )

    expected = _expected(
        payload,
        codec="pcma",
        jitter_std_ms=10.0,
        playout_buffer_ms=0.0,
        frame_duration_ms=20,
        seed=1,
    )
    assert np.array_equal(result, expected)
    assert np.any(result == 0xD5)


def test_same_seed_is_exactly_deterministic() -> None:
    apply_jitter_loss = _api()
    payload = np.arange(160 * 32, dtype=np.uint8)

    first = apply_jitter_loss(
        payload,
        codec="pcmu",
        jitter_std_ms=25.0,
        playout_buffer_ms=5.0,
        seed=99,
    )
    second = apply_jitter_loss(
        payload,
        codec="pcmu",
        jitter_std_ms=25.0,
        playout_buffer_ms=5.0,
        seed=99,
    )

    assert np.array_equal(first, second)


def test_different_seeds_can_produce_different_late_masks() -> None:
    apply_jitter_loss = _api()
    payload = np.arange(160 * 64, dtype=np.uint8)

    first = apply_jitter_loss(
        payload,
        codec="pcmu",
        jitter_std_ms=20.0,
        playout_buffer_ms=0.0,
        seed=1,
    )
    second = apply_jitter_loss(
        payload,
        codec="pcmu",
        jitter_std_ms=20.0,
        playout_buffer_ms=0.0,
        seed=2,
    )

    assert not np.array_equal(first, second)


@pytest.mark.parametrize(
    ("payload", "kwargs", "message"),
    [
        (np.zeros((2, 2), dtype=np.uint8), {}, "one-dimensional"),
        (np.zeros(10, dtype=np.int16), {}, "dtype uint8"),
        (np.zeros(10, dtype=np.uint8), {"codec": "bad"}, "codec"),
        (np.zeros(10, dtype=np.uint8), {"jitter_std_ms": -1.0}, "jitter_std_ms"),
        (np.zeros(10, dtype=np.uint8), {"jitter_std_ms": float("nan")}, "jitter_std_ms"),
        (np.zeros(10, dtype=np.uint8), {"playout_buffer_ms": -1.0}, "playout_buffer_ms"),
        (np.zeros(10, dtype=np.uint8), {"playout_buffer_ms": float("inf")}, "playout_buffer_ms"),
        (np.zeros(10, dtype=np.uint8), {"frame_duration_ms": 0}, "frame_duration_ms"),
        (np.zeros(10, dtype=np.uint8), {"frame_duration_ms": 1.5}, "frame_duration_ms"),
        (np.zeros(10, dtype=np.uint8), {"seed": -1}, "seed"),
        (np.zeros(10, dtype=np.uint8), {"seed": True}, "seed"),
    ],
)
def test_jitter_rejects_invalid_inputs(
    payload: np.ndarray,
    kwargs: dict[str, object],
    message: str,
) -> None:
    apply_jitter_loss = _api()
    options: dict[str, object] = {
        "codec": "pcmu",
        "jitter_std_ms": 10.0,
        "playout_buffer_ms": 5.0,
        "frame_duration_ms": 20,
        "seed": 0,
    }
    options.update(kwargs)

    with pytest.raises(ValueError, match=message):
        apply_jitter_loss(payload, **options)


def test_jitter_loss_is_exported_from_public_api() -> None:
    try:
        from callasr import apply_jitter_loss as exported
    except ImportError as exc:
        pytest.fail(f"public jitter-loss export is missing: {exc}")
    from callasr.impairments import apply_jitter_loss as direct

    assert exported is direct
