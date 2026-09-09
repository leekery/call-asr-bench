from __future__ import annotations

import base64
import importlib
import json
from collections.abc import Iterable
from typing import ClassVar

import numpy as np
import pytest

from callasr.audio import AudioBuffer
from callasr.streaming import StreamingASRAdapter, frame_audio, run_streaming_benchmark


def _api():
    try:
        module = importlib.import_module("callasr.adapters.vllm_realtime")
    except ImportError as exc:
        pytest.fail(f"vLLM Realtime adapter implementation is missing: {exc}")
    return module


def _frame(values: list[float], sample_rate: int = 16_000) -> AudioBuffer:
    return AudioBuffer(np.asarray(values, dtype=np.float32), sample_rate=sample_rate)


class FakeConnection:
    def __init__(self, incoming: list[object]) -> None:
        self.incoming = iter(incoming)
        self.sent: list[dict[str, object]] = []
        self.recv_timeouts: list[float] = []
        self.entered = False
        self.exited = False

    def __enter__(self) -> FakeConnection:
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.exited = True

    def send(self, message: str) -> None:
        payload = json.loads(message)
        assert isinstance(payload, dict)
        self.sent.append(payload)

    def recv(self, timeout: float | None = None) -> object:
        assert timeout is not None
        self.recv_timeouts.append(timeout)
        return next(self.incoming)


class FakeConnectFactory:
    def __init__(self, connection: FakeConnection | None = None) -> None:
        self.connection = connection
        self.calls: list[dict[str, object]] = []
        self.error: Exception | None = None

    def __call__(
        self,
        uri: str,
        *,
        additional_headers: dict[str, str],
        open_timeout: float,
        close_timeout: float,
    ) -> FakeConnection:
        self.calls.append(
            {
                "uri": uri,
                "additional_headers": dict(additional_headers),
                "open_timeout": open_timeout,
                "close_timeout": close_timeout,
            }
        )
        if self.error is not None:
            raise self.error
        assert self.connection is not None
        return self.connection


def _event(event_type: str, **fields: object) -> str:
    return json.dumps({"type": event_type, **fields})


def test_adapter_metadata_protocol_and_http_to_ws_normalization() -> None:
    module = _api()
    connection = FakeConnection([_event("session.created"), _event("transcription.done", text="")])
    factory = FakeConnectFactory(connection)

    adapter = module.VLLMRealtimeAdapter(
        "mistralai/Voxtral-Mini-4B-Realtime-2602",
        base_url="http://localhost:8000/v1/",
        api_key="secret-key",
        timeout_seconds=12.5,
        connect_factory=factory,
    )

    assert isinstance(adapter, StreamingASRAdapter)
    assert adapter.name == "vllm-realtime"
    assert adapter.model == "mistralai/Voxtral-Mini-4B-Realtime-2602"
    assert adapter.device == "remote"
    assert adapter.compute_type == "server"
    assert adapter.decoding_options == {
        "base_url": "http://localhost:8000/v1",
        "websocket_endpoint": "ws://localhost:8000/v1/realtime",
        "input_format": "pcm16",
        "sample_rate_hz": 16000,
        "timeout_seconds": 12.5,
    }
    assert "secret-key" not in repr(adapter.decoding_options)

    assert list(adapter.stream([_frame([0.0])]))[-1].is_final is True
    assert factory.calls == [
        {
            "uri": "ws://localhost:8000/v1/realtime",
            "additional_headers": {"Authorization": "Bearer secret-key"},
            "open_timeout": 12.5,
            "close_timeout": 12.5,
        }
    ]


def test_https_base_url_uses_wss_endpoint() -> None:
    module = _api()
    connection = FakeConnection([_event("session.created"), _event("transcription.done", text="ok")])
    factory = FakeConnectFactory(connection)
    adapter = module.VLLMRealtimeAdapter(
        "model",
        base_url="https://asr.example/v1",
        connect_factory=factory,
    )

    list(adapter.stream([_frame([0.0])]))

    assert factory.calls[0]["uri"] == "wss://asr.example/v1/realtime"
    assert factory.calls[0]["additional_headers"] == {}


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        " ws://example.test/v1",
        "ws://example.test/v1",
        "ftp://example.test/v1",
        "http://user:pass@example.test/v1",
        "http://example.test/v1?token=secret",
        "http://example.test/v1#fragment",
    ],
)
def test_invalid_base_urls_are_rejected(base_url: str) -> None:
    module = _api()

    with pytest.raises(module.VLLMRealtimeError, match="base_url"):
        module.VLLMRealtimeAdapter("model", base_url=base_url, connect_factory=FakeConnectFactory())


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_timeout_is_rejected(timeout: object) -> None:
    module = _api()

    with pytest.raises(module.VLLMRealtimeError, match="timeout_seconds"):
        module.VLLMRealtimeAdapter(
            "model",
            base_url="http://localhost:8000/v1",
            timeout_seconds=timeout,
            connect_factory=FakeConnectFactory(),
        )


def test_language_is_rejected_before_connection() -> None:
    module = _api()
    factory = FakeConnectFactory()
    adapter = module.VLLMRealtimeAdapter(
        "model",
        base_url="http://localhost:8000/v1",
        connect_factory=factory,
    )

    with pytest.raises(module.VLLMRealtimeError, match="language"):
        list(adapter.stream([_frame([0.0])], language="ru"))

    assert factory.calls == []


def test_non_16khz_audio_is_rejected_before_connection() -> None:
    module = _api()
    factory = FakeConnectFactory()
    adapter = module.VLLMRealtimeAdapter(
        "model",
        base_url="http://localhost:8000/v1",
        connect_factory=factory,
    )

    with pytest.raises(module.VLLMRealtimeError, match="16000"):
        list(adapter.stream([_frame([0.0], sample_rate=8_000)]))

    assert factory.calls == []


def test_exact_vllm_event_sequence_pcm16_transport_and_cumulative_deltas() -> None:
    module = _api()
    connection = FakeConnection(
        [
            _event("session.created"),
            _event("session.updated", model="model"),
            _event("transcription.delta", delta="hel"),
            _event("transcription.delta", delta=""),
            _event("transcription.delta", delta="lo"),
            _event("transcription.done", text="hello!"),
        ]
    )
    factory = FakeConnectFactory(connection)
    adapter = module.VLLMRealtimeAdapter(
        "model",
        base_url="http://localhost:8000/v1",
        connect_factory=factory,
    )
    frames = [
        _frame([-2.0, -1.0, -0.5, 0.0]),
        _frame([0.5, 1.0, 2.0]),
    ]

    updates = list(adapter.stream(frames))

    assert [(item.text, item.is_final) for item in updates] == [
        ("hel", False),
        ("hello", False),
        ("hello!", True),
    ]
    assert connection.sent[0] == {"type": "session.update", "model": "model"}
    assert connection.sent[1] == {"type": "input_audio_buffer.commit"}
    assert connection.sent[-1] == {"type": "input_audio_buffer.commit", "final": True}
    append_events = connection.sent[2:-1]
    assert [event["type"] for event in append_events] == [
        "input_audio_buffer.append",
        "input_audio_buffer.append",
    ]

    first_pcm = np.frombuffer(base64.b64decode(append_events[0]["audio"]), dtype="<i2")
    second_pcm = np.frombuffer(base64.b64decode(append_events[1]["audio"]), dtype="<i2")
    np.testing.assert_array_equal(first_pcm, [-32767, -32767, -16384, 0])
    np.testing.assert_array_equal(second_pcm, [16384, 32767, 32767])
    assert connection.entered is True
    assert connection.exited is True
    assert all(timeout == 60.0 for timeout in connection.recv_timeouts)


def test_server_error_is_sanitized_and_does_not_echo_secret_or_body() -> None:
    module = _api()
    secret = "super-secret-9bd1"
    connection = FakeConnection(
        [
            _event("session.created"),
            _event("error", error={"message": f"bad request {secret}"}),
        ]
    )
    adapter = module.VLLMRealtimeAdapter(
        "model",
        base_url="http://localhost:8000/v1",
        api_key=secret,
        connect_factory=FakeConnectFactory(connection),
    )

    with pytest.raises(module.VLLMRealtimeError) as captured:
        list(adapter.stream([_frame([0.0])]))

    assert "server returned an error" in str(captured.value)
    assert secret not in str(captured.value)
    assert "bad request" not in str(captured.value)


def test_connection_failure_is_sanitized() -> None:
    module = _api()
    secret = "socket-secret-a2f0"
    factory = FakeConnectFactory()
    factory.error = OSError(f"connection failed with {secret}")
    adapter = module.VLLMRealtimeAdapter(
        "model",
        base_url="http://localhost:8000/v1",
        api_key=secret,
        connect_factory=factory,
    )

    with pytest.raises(module.VLLMRealtimeError) as captured:
        list(adapter.stream([_frame([0.0])]))

    assert "connection failed" in str(captured.value)
    assert secret not in str(captured.value)


@pytest.mark.parametrize(
    "incoming",
    [
        ["not-json"],
        [_event("wrong.first.event")],
        [_event("session.created"), json.dumps({"type": "transcription.delta", "delta": 123})],
        [_event("session.created"), json.dumps({"type": "transcription.done", "text": 123})],
    ],
)
def test_malformed_protocol_messages_are_actionable(incoming: list[object]) -> None:
    module = _api()
    adapter = module.VLLMRealtimeAdapter(
        "model",
        base_url="http://localhost:8000/v1",
        connect_factory=FakeConnectFactory(FakeConnection(incoming)),
    )

    with pytest.raises(module.VLLMRealtimeError, match="protocol"):
        list(adapter.stream([_frame([0.0])]))


def test_socket_close_before_done_is_actionable() -> None:
    module = _api()
    connection = FakeConnection([_event("session.created"), _event("transcription.delta", delta="hi")])
    adapter = module.VLLMRealtimeAdapter(
        "model",
        base_url="http://localhost:8000/v1",
        connect_factory=FakeConnectFactory(connection),
    )

    with pytest.raises(module.VLLMRealtimeError, match="before transcription.done"):
        list(adapter.stream([_frame([0.0])]))


def test_missing_optional_dependency_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _api()

    def missing(name: str):
        assert name == "websockets.sync.client"
        error = ModuleNotFoundError("No module named 'websockets'")
        error.name = "websockets"
        raise error

    monkeypatch.setattr(module, "import_module", missing)

    with pytest.raises(module.VLLMRealtimeError, match="uv sync --extra vllm-realtime"):
        module.VLLMRealtimeAdapter("model", base_url="http://localhost:8000/v1")


def test_transitive_missing_dependency_is_not_misreported(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _api()

    def missing(name: str):
        assert name == "websockets.sync.client"
        error = ModuleNotFoundError("No module named 'ssl_helper'")
        error.name = "ssl_helper"
        raise error

    monkeypatch.setattr(module, "import_module", missing)

    with pytest.raises(ModuleNotFoundError, match="ssl_helper"):
        module.VLLMRealtimeAdapter("model", base_url="http://localhost:8000/v1")


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def test_adapter_runs_end_to_end_through_streaming_benchmark() -> None:
    module = _api()
    connection = FakeConnection(
        [
            _event("session.created"),
            _event("transcription.delta", delta="hello"),
            _event("transcription.done", text="hello world"),
        ]
    )
    adapter = module.VLLMRealtimeAdapter(
        "model",
        base_url="http://localhost:8000/v1",
        connect_factory=FakeConnectFactory(connection),
    )
    audio = AudioBuffer(np.zeros(640, dtype=np.float32), sample_rate=16_000)
    clock = SequenceClock([0.0, 0.01, 0.02, 0.20, 0.40, 0.50])

    result = run_streaming_benchmark(
        audio,
        adapter,
        frame_duration_ms=20,
        clock=clock,
    )

    assert result.frame_count == 2
    assert result.final_text == "hello world"
    assert result.time_to_first_partial_seconds == pytest.approx(0.20)
    assert result.finalization_latency_seconds == pytest.approx(0.38)
    assert result.total_streaming_wall_seconds == pytest.approx(0.50)
    assert result.partial_stability == pytest.approx(0.5)
