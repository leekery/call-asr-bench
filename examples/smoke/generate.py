"""Generate deterministic non-speech WAV files for the smoke fixture."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

_SAMPLE_RATE = 16_000
_SAMPLE_COUNT = 4_000
_AMPLITUDE = 8_192
_FIXTURES = {
    "tone-ru.wav": 440,
    "tone-en.wav": 660,
}


def _wav_bytes(frequency_hz: int) -> bytes:
    pcm = bytearray()
    half_cycle = _SAMPLE_RATE // 2
    for sample_index in range(_SAMPLE_COUNT):
        phase = (sample_index * frequency_hz) % _SAMPLE_RATE
        value = _AMPLITUDE if phase < half_cycle else -_AMPLITUDE
        pcm.extend(struct.pack("<h", value))

    data = bytes(pcm)
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVE"
        + b"fmt "
        + struct.pack(
            "<IHHIIHH",
            16,
            1,
            1,
            _SAMPLE_RATE,
            _SAMPLE_RATE * 2,
            2,
            16,
        )
        + b"data"
        + struct.pack("<I", len(data))
    )
    return header + data


def generate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for filename, frequency_hz in _FIXTURES.items():
        (output / filename).write_bytes(_wav_bytes(frequency_hz))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
