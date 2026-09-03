import numpy as np
import pytest

import callasr
from callasr import apply_packet_loss


def test_packet_loss_is_part_of_public_api() -> None:
    assert callable(getattr(callasr, "apply_packet_loss", None))


def test_zero_packet_loss_preserves_payload_without_aliasing() -> None:
    payload = np.arange(24, dtype=np.uint8)

    result = apply_packet_loss(payload, codec="pcmu", loss_rate=0.0)

    np.testing.assert_array_equal(result, payload)
    assert result is not payload
    assert not np.shares_memory(result, payload)


@pytest.mark.parametrize(("codec", "silence_byte"), [("pcmu", 0xFF), ("pcma", 0xD5)])
def test_full_packet_loss_replaces_every_frame_with_codec_silence(
    codec: str, silence_byte: int
) -> None:
    payload = np.arange(19, dtype=np.uint8)

    result = apply_packet_loss(
        payload,
        codec=codec,
        loss_rate=1.0,
        frame_duration_ms=1,
    )

    np.testing.assert_array_equal(result, np.full(19, silence_byte, dtype=np.uint8))


def test_seeded_packet_loss_drops_complete_frames_reproducibly() -> None:
    payload = np.arange(32, dtype=np.uint8)
    expected = np.concatenate(
        [
            np.arange(8, dtype=np.uint8),
            np.full(24, 0xFF, dtype=np.uint8),
        ]
    )

    first = apply_packet_loss(
        payload,
        codec="pcmu",
        loss_rate=0.5,
        frame_duration_ms=1,
        seed=0,
    )
    second = apply_packet_loss(
        payload,
        codec="pcmu",
        loss_rate=0.5,
        frame_duration_ms=1,
        seed=0,
    )

    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)


@pytest.mark.parametrize(
    ("payload", "kwargs", "message"),
    [
        (np.zeros((2, 2), dtype=np.uint8), {}, "one-dimensional"),
        (np.zeros(8, dtype=np.int16), {}, "uint8"),
        (np.zeros(8, dtype=np.uint8), {"codec": "mp3"}, "codec"),
        (np.zeros(8, dtype=np.uint8), {"loss_rate": -0.1}, "loss_rate"),
        (np.zeros(8, dtype=np.uint8), {"loss_rate": 1.1}, "loss_rate"),
        (np.zeros(8, dtype=np.uint8), {"loss_rate": np.nan}, "loss_rate"),
        (np.zeros(8, dtype=np.uint8), {"frame_duration_ms": 0}, "frame_duration_ms"),
    ],
)
def test_packet_loss_rejects_invalid_inputs(
    payload: np.ndarray, kwargs: dict[str, object], message: str
) -> None:
    options: dict[str, object] = {"codec": "pcmu", "loss_rate": 0.1}
    options.update(kwargs)

    with pytest.raises(ValueError, match=message):
        apply_packet_loss(payload, **options)
