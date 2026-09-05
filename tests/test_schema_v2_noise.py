from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from callasr.adapters.base import Transcription
from callasr.audio import AudioBuffer
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

    def __init__(self) -> None:
        self.calls: list[AudioBuffer] = []

    def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcription:
        self.calls.append(audio)
        return Transcription("hello")


def _item(tmp_path: Path, index: int = 0) -> DatasetItem:
    audio_path = tmp_path / f"audio-{index}.wav"
    audio_path.touch()
    return DatasetItem(
        id=f"id-{index}",
        audio=audio_path,
        reference="hello",
        language="en",
        line_number=index + 1,
    )


def _result(additive_noise_snr_db: float | None) -> BenchmarkResult:
    return BenchmarkResult(
        created_at="2026-09-04T18:00:00+00:00",
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
            additive_noise_snr_db=additive_noise_snr_db,
            seed=42,
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
                audio="audio-0.wav",
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


def test_current_schema_keeps_nullable_additive_noise_metadata() -> None:
    payload = result_to_dict(_result(None))

    assert payload["schema_version"] == 4
    assert payload["channel"]["additive_noise_snr_db"] is None


def test_runner_applies_noise_before_g711_and_preserves_packet_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    items = (_item(tmp_path),)
    monkeypatch.setattr("callasr.benchmark.load_dataset_manifest", lambda path: items)
    source = AudioBuffer(np.full(8_000, 0.25, dtype=np.float32), 8_000)
    monkeypatch.setattr("callasr.benchmark.load_wav", lambda path: source)
    events: list[tuple[object, ...]] = []

    def noise(audio: AudioBuffer, *, snr_db: float, seed: int) -> AudioBuffer:
        events.append(("noise", snr_db, seed))
        return AudioBuffer(np.full(audio.samples.shape, 0.5, dtype=np.float32), audio.sample_rate)

    def channel(
        audio: AudioBuffer,
        codec: str,
        *,
        packet_loss_rate: float,
        frame_duration_ms: int,
        seed: int,
    ) -> AudioBuffer:
        events.append(("channel", codec, packet_loss_rate, frame_duration_ms, seed))
        assert np.all(audio.samples == 0.5)
        return audio

    monkeypatch.setattr("callasr.benchmark.apply_additive_noise", noise)
    monkeypatch.setattr("callasr.benchmark.telephone_channel", channel)
    times = iter([0.0, 0.25])
    monkeypatch.setattr("callasr.benchmark.perf_counter", lambda: next(times))

    result = run_benchmark(
        tmp_path / "dataset.jsonl",
        FakeAdapter(),
        codec="pcmu",
        packet_loss_rate=0.05,
        snr_db=15.0,
        seed=42,
    )

    expected_packet_seed = int(
        np.random.SeedSequence([42, 0]).generate_state(1, dtype=np.uint32)[0]
    )
    expected_noise_seed = int(
        np.random.SeedSequence([42, 0, 1]).generate_state(1, dtype=np.uint32)[0]
    )
    assert events == [
        ("noise", 15.0, expected_noise_seed),
        ("channel", "pcmu", 0.05, 20, expected_packet_seed),
    ]
    assert result.channel.additive_noise_snr_db == 15.0


def test_clean_run_can_use_noise_without_telephone_channel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    items = (_item(tmp_path),)
    monkeypatch.setattr("callasr.benchmark.load_dataset_manifest", lambda path: items)
    source = AudioBuffer(np.full(16_000, 0.1, dtype=np.float32), 16_000)
    monkeypatch.setattr("callasr.benchmark.load_wav", lambda path: source)
    seen: list[int] = []

    def noise(audio: AudioBuffer, *, snr_db: float, seed: int) -> AudioBuffer:
        seen.append(seed)
        return audio

    def forbidden_channel(*args: object, **kwargs: object) -> AudioBuffer:
        raise AssertionError("telephone channel called for codec=none")

    monkeypatch.setattr("callasr.benchmark.apply_additive_noise", noise)
    monkeypatch.setattr("callasr.benchmark.telephone_channel", forbidden_channel)
    monkeypatch.setattr("callasr.benchmark.perf_counter", lambda: 1.0)

    result = run_benchmark(
        tmp_path / "dataset.jsonl",
        FakeAdapter(),
        codec="none",
        snr_db=-5.0,
        seed=7,
    )

    expected = int(np.random.SeedSequence([7, 0, 1]).generate_state(1, dtype=np.uint32)[0])
    assert seen == [expected]
    assert result.channel.additive_noise_snr_db == -5.0


def test_runner_without_snr_does_not_apply_noise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    items = (_item(tmp_path),)
    monkeypatch.setattr("callasr.benchmark.load_dataset_manifest", lambda path: items)
    monkeypatch.setattr(
        "callasr.benchmark.load_wav",
        lambda path: AudioBuffer(np.ones(800, dtype=np.float32), 8_000),
    )

    def forbidden_noise(*args: object, **kwargs: object) -> AudioBuffer:
        raise AssertionError("noise called without snr_db")

    monkeypatch.setattr("callasr.benchmark.apply_additive_noise", forbidden_noise)
    monkeypatch.setattr("callasr.benchmark.perf_counter", lambda: 1.0)

    result = run_benchmark(tmp_path / "dataset.jsonl", FakeAdapter(), codec="none")

    assert result.channel.additive_noise_snr_db is None


def test_cli_accepts_snr_db_and_passes_it_to_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from callasr import cli

    seen: dict[str, object] = {}

    monkeypatch.setattr(cli, "FasterWhisperAdapter", lambda *args, **kwargs: object())

    def run(*args: object, **kwargs: object) -> BenchmarkResult:
        seen.update(kwargs)
        return _result(-5.0)

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
            "--snr-db",
            "-5",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert status == 0
    assert seen["snr_db"] == -5.0


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_cli_rejects_non_finite_snr(
    value: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
                f"--snr-db={value}",
                "--output",
                str(tmp_path / "result.json"),
            ]
        )

    assert "must be finite" in capsys.readouterr().err
