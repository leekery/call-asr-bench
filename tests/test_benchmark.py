from callasr.benchmark import (
    AdapterInfo,
    BenchmarkResult,
    BenchmarkSummary,
    ChannelInfo,
    DatasetInfo,
    ItemResult,
    result_to_dict,
)


def test_result_schema_serializes_to_versioned_json_ready_mapping() -> None:
    result = BenchmarkResult(
        created_at="2026-09-04T10:00:00+00:00",
        dataset=DatasetInfo(path="/tmp/dataset.jsonl", item_count=1),
        adapter=AdapterInfo(
            name="faster-whisper",
            model="large-v3",
            device="auto",
            compute_type="default",
            decoding_options={"beam_size": 5, "temperature": 0.0},
        ),
        channel=ChannelInfo(
            codec="pcmu",
            packet_loss_rate=0.05,
            frame_duration_ms=20,
            seed=42,
        ),
        summary=BenchmarkSummary(
            total_audio_seconds=2.0,
            adapter_seconds=0.5,
            wer=0.25,
            cer=0.1,
            rtf=0.25,
            speed_factor=4.0,
        ),
        items=(
            ItemResult(
                id="call-001",
                audio="audio/call-001.wav",
                reference="добрый день",
                hypothesis="добрый вечер",
                language="ru",
                audio_seconds=2.0,
                adapter_seconds=0.5,
                wer=0.5,
                cer=0.25,
            ),
        ),
    )

    payload = result_to_dict(result)

    assert payload["schema_version"] == 4
    assert payload["dataset"] == {"path": "/tmp/dataset.jsonl", "item_count": 1}
    assert payload["adapter"]["decoding_options"]["temperature"] == 0.0
    assert payload["channel"]["codec"] == "pcmu"
    assert payload["channel"]["additive_noise_snr_db"] is None
    assert payload["channel"]["jitter_std_ms"] is None
    assert payload["channel"]["playout_buffer_ms"] is None
    assert payload["summary"]["speed_factor"] == 4.0
    assert payload["items"][0]["id"] == "call-001"
    assert isinstance(payload["items"], list)
