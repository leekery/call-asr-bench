from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from callasr.adapters.base import Transcription
from callasr.audio import AudioBuffer, telephone_channel
from callasr.benchmark import (
    AdapterInfo,
    BenchmarkResult,
    BenchmarkSummary,
    ChannelInfo,
    DatasetInfo,
    ItemResult,
    result_to_dict,
    run_benchmark,
)
from callasr.dataset import DatasetItem


class FakeAdapter:
    name = "fake"
    model = "fake-model"
    device = "cpu"
    compute_type = "float32"
    decoding_options: ClassVar[dict[str, int]] = {"beam_size": 1}

    def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcription:
        return Transcription("hello")


def _item(tmp_path: Path) -> DatasetItem:
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    return DatasetItem(
        id="id-0",
        audio=audio_path,
        reference="hello",
        language="en",
        line_number=1,
    )


def _result() -> BenchmarkResult:
    return BenchmarkResult(
        created_at="2026-09-05T20:00:00+00:00",
        dataset=DatasetInfo(path="/tmp/dataset.jsonl", item_count=1),
        adapter=AdapterInfo(
            name="fake",
            model="fake-model",
            device="cpu",
            compute_type="float32",
            decoding_options={"beam_size": 1},
        ),
        channel=ChannelInfo(
            codec="pcmu",
            packet_loss_rate=0.05,
            frame_duration_ms=20,
            seed=42,
            additive_noise_snr_db=15.0,
            jitter_std_ms=8.0,
            playout_buffer_ms=20.0,
        ),
        summary=BenchmarkSummary(
            total_audio_seconds=1.0,
            adapter_seconds=0.25,
            wer=0.0,
            cer=0.0,
            rtf=0.25,
            speed_factor=4.0,
        ),
        items=(
            ItemResult(
                id="id-0",
                audio="audio.wav",
                reference="hello",
                hypothesis="hello",
                language="en",
                audio_seconds=1.0,
                adapter_seconds=0.25,
                wer=0.0,
                cer=0.0,
            ),
        ),
    )


def test_current_schema_records_jitter_configuration() -> None:
    payload = result_to_dict(_result())

    assert payload["schema_version"] == 4
    assert payload["channel"]["jitter_std_ms"] == 8.0
    assert payload["channel"]["playout_buffer_ms"] == 20.0


def test_telephone_channel_applies_packet_loss_before_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = AudioBuffer(np.full(160, 0.1, dtype=np.float32), 8_000)
    events: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        "callasr.audio.encode_g711",
        lambda samples, codec: np.arange(samples.size, dtype=np.uint8),
    )

    def packet_loss(payload, *, codec, loss_rate, frame_duration_ms, seed):
        events.append(("packet", codec, loss_rate, frame_duration_ms, seed))
        result = payload.copy()
        result[0] = 0xAA
        return result

    def jitter(
        payload,
        *,
        codec,
        jitter_std_ms,
        playout_buffer_ms,
        frame_duration_ms,
        seed,
    ):
        assert payload[0] == 0xAA
        events.append(("jitter", codec, jitter_std_ms, playout_buffer_ms, frame_duration_ms, seed))
        result = payload.copy()
        result[1] = 0xBB
        return result

    def decode(payload, codec):
        assert payload[1] == 0xBB
        return np.zeros(payload.size, dtype=np.float32)

    monkeypatch.setattr("callasr.audio.apply_packet_loss", packet_loss)
    monkeypatch.setattr("callasr.audio.apply_jitter_loss", jitter)
    monkeypatch.setattr("callasr.audio.decode_g711", decode)

    result = telephone_channel(
        source,
        codec="pcmu",
        packet_loss_rate=0.05,
        frame_duration_ms=20,
        seed=11,
        jitter_std_ms=8.0,
        playout_buffer_ms=20.0,
        jitter_seed=22,
    )

    assert result.sample_rate == 8_000
    assert events == [
        ("packet", "pcmu", 0.05, 20, 11),
        ("jitter", "pcmu", 8.0, 20.0, 20, 22),
    ]


def test_runner_preserves_existing_seed_streams_and_adds_independent_jitter_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    item = _item(tmp_path)
    monkeypatch.setattr("callasr.benchmark.load_dataset_manifest", lambda path: (item,))
    monkeypatch.setattr(
        "callasr.benchmark.load_wav",
        lambda path: AudioBuffer(np.full(8_000, 0.1, dtype=np.float32), 8_000),
    )
    seen: dict[str, object] = {}

    def noise(audio: AudioBuffer, *, snr_db: float, seed: int) -> AudioBuffer:
        seen["noise"] = (snr_db, seed)
        return audio

    def channel(
        audio: AudioBuffer,
        codec: str,
        *,
        packet_loss_rate: float,
        frame_duration_ms: int,
        seed: int,
        jitter_std_ms: float | None,
        playout_buffer_ms: float | None,
        jitter_seed: int,
    ) -> AudioBuffer:
        seen["channel"] = (
            codec,
            packet_loss_rate,
            frame_duration_ms,
            seed,
            jitter_std_ms,
            playout_buffer_ms,
            jitter_seed,
        )
        return audio

    monkeypatch.setattr("callasr.benchmark.apply_additive_noise", noise)
    monkeypatch.setattr("callasr.benchmark.telephone_channel", channel)
    monkeypatch.setattr("callasr.benchmark.perf_counter", lambda: 1.0)

    result = run_benchmark(
        tmp_path / "dataset.jsonl",
        FakeAdapter(),
        codec="pcmu",
        packet_loss_rate=0.05,
        snr_db=15.0,
        jitter_std_ms=8.0,
        playout_buffer_ms=20.0,
        seed=42,
    )

    packet_seed = int(np.random.SeedSequence([42, 0]).generate_state(1, dtype=np.uint32)[0])
    noise_seed = int(np.random.SeedSequence([42, 0, 1]).generate_state(1, dtype=np.uint32)[0])
    jitter_seed = int(np.random.SeedSequence([42, 0, 2]).generate_state(1, dtype=np.uint32)[0])
    assert seen["noise"] == (15.0, noise_seed)
    assert seen["channel"] == ("pcmu", 0.05, 20, packet_seed, 8.0, 20.0, jitter_seed)
    assert result.channel.jitter_std_ms == 8.0
    assert result.channel.playout_buffer_ms == 20.0


def test_runner_without_jitter_preserves_current_behavior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    item = _item(tmp_path)
    monkeypatch.setattr("callasr.benchmark.load_dataset_manifest", lambda path: (item,))
    monkeypatch.setattr(
        "callasr.benchmark.load_wav",
        lambda path: AudioBuffer(np.full(8_000, 0.1, dtype=np.float32), 8_000),
    )
    seen: dict[str, object] = {}

    def channel(audio: AudioBuffer, codec: str, **kwargs: object) -> AudioBuffer:
        seen.update(kwargs)
        return audio

    monkeypatch.setattr("callasr.benchmark.telephone_channel", channel)
    monkeypatch.setattr("callasr.benchmark.perf_counter", lambda: 1.0)

    result = run_benchmark(
        tmp_path / "dataset.jsonl",
        FakeAdapter(),
        codec="pcmu",
        seed=7,
    )

    assert result.channel.jitter_std_ms is None
    assert result.channel.playout_buffer_ms is None
    assert "jitter_std_ms" not in seen
    assert "playout_buffer_ms" not in seen
    assert "jitter_seed" not in seen


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"codec": "none", "jitter_std_ms": 8.0, "playout_buffer_ms": 20.0}, "codec"),
        ({"codec": "pcmu", "jitter_std_ms": 8.0}, "must be provided together"),
        ({"codec": "pcmu", "playout_buffer_ms": 20.0}, "must be provided together"),
    ],
)
def test_runner_rejects_invalid_jitter_configuration(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_benchmark(tmp_path / "dataset.jsonl", FakeAdapter(), **kwargs)


def test_cli_accepts_jitter_flags_and_passes_them_to_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from callasr import cli

    seen: dict[str, object] = {}
    monkeypatch.setattr(cli, "FasterWhisperAdapter", lambda *args, **kwargs: object())

    def run(*args: object, **kwargs: object) -> BenchmarkResult:
        seen.update(kwargs)
        return _result()

    monkeypatch.setattr(cli, "run_benchmark", run)
    monkeypatch.setattr(cli, "write_result_artifact", lambda result, path: None)

    status = cli.main(
        [
            "run",
            "dataset.jsonl",
            "--adapter",
            "faster-whisper",
            "--model",
            "large-v3",
            "--codec",
            "pcmu",
            "--jitter-std-ms",
            "8",
            "--playout-buffer-ms",
            "20",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert status == 0
    assert seen["jitter_std_ms"] == 8.0
    assert seen["playout_buffer_ms"] == 20.0


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--codec", "none", "--jitter-std-ms", "8", "--playout-buffer-ms", "20"],
        ["--codec", "pcmu", "--jitter-std-ms", "8"],
        ["--codec", "pcmu", "--playout-buffer-ms", "20"],
    ],
)
def test_cli_rejects_invalid_jitter_configuration_before_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra_args: list[str],
) -> None:
    from callasr import cli

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("adapter must not be constructed")

    monkeypatch.setattr(cli, "FasterWhisperAdapter", forbidden)
    argv = [
        "run",
        "dataset.jsonl",
        "--adapter",
        "faster-whisper",
        "--model",
        "large-v3",
        *extra_args,
        "--output",
        str(tmp_path / "result.json"),
    ]

    assert cli.main(argv) == 2


@pytest.mark.parametrize("flag", ["--jitter-std-ms", "--playout-buffer-ms"])
def test_cli_rejects_negative_jitter_values(flag: str, tmp_path: Path) -> None:
    from callasr import cli

    with pytest.raises(SystemExit, match="2"):
        cli.main(
            [
                "run",
                "dataset.jsonl",
                "--adapter",
                "faster-whisper",
                "--model",
                "large-v3",
                "--codec",
                "pcmu",
                flag,
                "-1",
                "--output",
                str(tmp_path / "result.json"),
            ]
        )
