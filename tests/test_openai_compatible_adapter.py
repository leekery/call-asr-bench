from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from callasr.adapters.base import AdapterError, ASRAdapter
from callasr.audio import AudioBuffer
from callasr.benchmark import result_to_dict, run_benchmark
from callasr.dataset import DatasetItem


def _api():
    try:
        import callasr.adapters.openai_compatible as module
        from callasr.adapters.openai_compatible import OpenAICompatibleAdapter
    except ImportError as exc:
        pytest.fail(f"OpenAI-compatible adapter implementation is missing: {exc}")
    return module, OpenAICompatibleAdapter


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: object | None = None,
        json_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self.payload = {"text": "hello"} if payload is None else payload
        self.json_error = json_error
        self.json_calls = 0

    def json(self) -> object:
        self.json_calls += 1
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class FakeClient:
    def __init__(
        self,
        response: FakeResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or FakeResponse()
        self.error = error
        self.calls: list[dict[str, object]] = []

    def post(
        self,
        url: str,
        *,
        data: dict[str, str],
        files: dict[str, tuple[str, bytes, str]],
        headers: dict[str, str],
        timeout: float,
    ) -> FakeResponse:
        self.calls.append(
            {
                "url": url,
                "data": dict(data),
                "files": dict(files),
                "headers": dict(headers),
                "timeout": timeout,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response


def test_adapter_metadata_is_reproducible_without_api_key() -> None:
    _, OpenAICompatibleAdapter = _api()
    client = FakeClient()
    secret = "sk-super-secret"

    adapter = OpenAICompatibleAdapter(
        "served-asr",
        base_url="http://localhost:8000/v1/",
        api_key=secret,
        timeout_seconds=12.5,
        client=client,
    )

    assert isinstance(adapter, ASRAdapter)
    assert adapter.name == "openai-compatible"
    assert adapter.model == "served-asr"
    assert adapter.device == "remote"
    assert adapter.compute_type == "server"
    assert adapter.decoding_options == {
        "base_url": "http://localhost:8000/v1",
        "response_format": "json",
        "upload_format": "wav_pcm16",
        "timeout_seconds": 12.5,
    }
    assert secret not in json.dumps(adapter.decoding_options)


def test_transcribe_posts_pcm16_wav_model_language_and_bearer_key() -> None:
    _, OpenAICompatibleAdapter = _api()
    response = FakeResponse(payload={"text": "привет"})
    client = FakeClient(response)
    adapter = OpenAICompatibleAdapter(
        "served-asr",
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
        timeout_seconds=30.0,
        client=client,
    )
    audio = AudioBuffer(np.array([2.0, -2.0, 0.5], dtype=np.float32), 8_000)

    transcription = adapter.transcribe(audio, language="ru")

    assert transcription.text == "привет"
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["url"] == "http://localhost:8000/v1/audio/transcriptions"
    assert call["data"] == {
        "model": "served-asr",
        "response_format": "json",
        "language": "ru",
    }
    assert call["headers"] == {"Authorization": "Bearer token-abc123"}
    assert call["timeout"] == 30.0

    filename, wav_bytes, content_type = call["files"]["file"]
    assert filename == "audio.wav"
    assert content_type == "audio/wav"
    with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 8_000
        assert reader.getnframes() == 3
        pcm = np.frombuffer(reader.readframes(3), dtype="<i2")
    assert pcm.tolist() == [32767, -32767, 16384]


def test_language_and_authorization_are_omitted_when_not_configured() -> None:
    _, OpenAICompatibleAdapter = _api()
    client = FakeClient(FakeResponse(payload={"text": "hello"}))
    adapter = OpenAICompatibleAdapter(
        "served-asr",
        base_url="https://asr.example.test/v1",
        client=client,
    )

    adapter.transcribe(AudioBuffer(np.zeros(8, dtype=np.float32), 16_000))

    call = client.calls[0]
    assert call["data"] == {"model": "served-asr", "response_format": "json"}
    assert call["headers"] == {}


@pytest.mark.parametrize(
    "base_url",
    [
        "not-a-url",
        "ftp://example.test/v1",
        "http://user:password@example.test/v1",
        "https://example.test/v1?api_key=secret",
        "https://example.test/v1#fragment",
    ],
)
def test_invalid_or_secret_bearing_base_urls_are_rejected(base_url: str) -> None:
    _, OpenAICompatibleAdapter = _api()
    client = FakeClient()

    with pytest.raises(AdapterError, match="base_url"):
        OpenAICompatibleAdapter("served-asr", base_url=base_url, client=client)

    assert client.calls == []


@pytest.mark.parametrize("timeout_seconds", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_timeout_is_rejected(timeout_seconds: object) -> None:
    _, OpenAICompatibleAdapter = _api()

    with pytest.raises(AdapterError, match="timeout"):
        OpenAICompatibleAdapter(
            "served-asr",
            base_url="http://localhost:8000/v1",
            timeout_seconds=timeout_seconds,
            client=FakeClient(),
        )


def test_missing_optional_http_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, OpenAICompatibleAdapter = _api()

    def missing(name: str):
        assert name == "httpx"
        raise ModuleNotFoundError("No module named 'httpx'", name="httpx")

    monkeypatch.setattr(module, "import_module", missing)

    with pytest.raises(AdapterError, match=r"uv sync --extra openai-compatible"):
        OpenAICompatibleAdapter("served-asr", base_url="http://localhost:8000/v1")


def test_transitive_missing_dependency_is_not_misreported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, OpenAICompatibleAdapter = _api()

    def missing(name: str):
        assert name == "httpx"
        raise ModuleNotFoundError("No module named 'anyio'", name="anyio")

    monkeypatch.setattr(module, "import_module", missing)

    with pytest.raises(ModuleNotFoundError, match="anyio"):
        OpenAICompatibleAdapter("served-asr", base_url="http://localhost:8000/v1")


def test_non_success_status_is_mapped_without_reading_or_leaking_body() -> None:
    _, OpenAICompatibleAdapter = _api()
    response = FakeResponse(status_code=503, payload={"error": "sk-secret-in-body"})
    client = FakeClient(response)
    adapter = OpenAICompatibleAdapter(
        "served-asr",
        base_url="http://localhost:8000/v1",
        api_key="sk-secret-key",
        client=client,
    )

    with pytest.raises(AdapterError, match="HTTP 503") as caught:
        adapter.transcribe(AudioBuffer(np.zeros(8, dtype=np.float32), 8_000))

    assert response.json_calls == 0
    assert "sk-secret" not in str(caught.value)


def test_invalid_json_and_missing_text_are_actionable() -> None:
    _, OpenAICompatibleAdapter = _api()
    audio = AudioBuffer(np.zeros(8, dtype=np.float32), 8_000)

    invalid_json = OpenAICompatibleAdapter(
        "served-asr",
        base_url="http://localhost:8000/v1",
        client=FakeClient(FakeResponse(json_error=ValueError("raw secret"))),
    )
    with pytest.raises(AdapterError, match="invalid JSON") as invalid:
        invalid_json.transcribe(audio)
    assert "raw secret" not in str(invalid.value)

    missing_text = OpenAICompatibleAdapter(
        "served-asr",
        base_url="http://localhost:8000/v1",
        client=FakeClient(FakeResponse(payload={"text": 123})),
    )
    with pytest.raises(AdapterError, match="string 'text'"):
        missing_text.transcribe(audio)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (TimeoutError("sk-timeout-secret"), "timed out"),
        (OSError("sk-transport-secret"), "request failed"),
    ],
)
def test_transport_failures_are_sanitized(error: Exception, message: str) -> None:
    _, OpenAICompatibleAdapter = _api()
    adapter = OpenAICompatibleAdapter(
        "served-asr",
        base_url="http://localhost:8000/v1",
        api_key="sk-secret-key",
        client=FakeClient(error=error),
    )

    with pytest.raises(AdapterError, match=message) as caught:
        adapter.transcribe(AudioBuffer(np.zeros(8, dtype=np.float32), 8_000))

    assert "sk-" not in str(caught.value)


def test_runner_artifact_contains_endpoint_metadata_but_not_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, OpenAICompatibleAdapter = _api()
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    item = DatasetItem(
        id="call-1",
        audio=audio_path,
        reference="hello 42",
        language="en",
        line_number=1,
    )
    monkeypatch.setattr("callasr.benchmark.load_dataset_manifest", lambda path: (item,))
    monkeypatch.setattr(
        "callasr.benchmark.load_wav",
        lambda path: AudioBuffer(np.zeros(8_000, dtype=np.float32), 8_000),
    )
    monkeypatch.setattr("callasr.benchmark.perf_counter", lambda: 1.0)
    secret = "sk-never-serialize"
    adapter = OpenAICompatibleAdapter(
        "served-asr",
        base_url="http://localhost:8000/v1",
        api_key=secret,
        client=FakeClient(FakeResponse(payload={"text": "hello 42"})),
    )

    result = run_benchmark(tmp_path / "dataset.jsonl", adapter)
    serialized = json.dumps(result_to_dict(result))

    assert result.adapter.name == "openai-compatible"
    assert result.adapter.device == "remote"
    assert result.adapter.decoding_options["base_url"] == "http://localhost:8000/v1"
    assert secret not in serialized
