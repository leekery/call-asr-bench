from __future__ import annotations

import asyncio
import base64
import importlib
import json
from collections.abc import AsyncIterator

import numpy as np
import pytest

from callasr.audio import AudioBuffer
from callasr.paced_streaming import PacedStreamingASRAdapter
from callasr.streaming import StreamingASRAdapter


def _api():
    return importlib.import_module("callasr.adapters.vllm_realtime_paced")


def _frame(values: list[float], sample_rate: int = 16_000) -> AudioBuffer:
    return AudioBuffer(np.asarray(values, dtype=np.float32), sample_rate=sample_rate)


def _event(event_type: str, **fields: object) -> str:
    return json.dumps({"type": event_type, **fields})


class FakeAsyncConnection:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.entered = False
        self.exited = False
        self.recv_calls = 0
        self.first_append = asyncio.Event()
        self.allow_second = asyncio.Event()
        self.final_commit = asyncio.Event()
        self.delta_before_second_append = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.exited = True

    async def send(self, message: str) -> None:
        payload = json.loads(message)
        assert isinstance(payload, dict)
        self.sent.append(payload)
        if payload.get("type") == "input_audio_buffer.append":
            append_count = sum(
                item.get("type") == "input_audio_buffer.append" for item in self.sent
            )
            if append_count == 1:
                self.first_append.set()
        if payload == {"type": "input_audio_buffer.commit", "final": True}:
            self.final_commit.set()

    async def recv(self) -> object:
        self.recv_calls += 1
        if self.recv_calls == 1:
            return _event("session.created")
        if self.recv_calls == 2:
            await self.first_append.wait()
            append_count = sum(
                item.get("type") == "input_audio_buffer.append" for item in self.sent
            )
            self.delta_before_second_append = append_count == 1
            self.allow_second.set()
            return _event("transcription.delta", delta="hel")
        if self.recv_calls == 3:
            await self.final_commit.wait()
            return _event("transcription.delta", delta="lo")
        if self.recv_calls == 4:
            return _event("transcription.done", text="hello!")
        raise AssertionError("unexpected recv")


class FakeConnectFactory:
    def __init__(self, connection) -> None:
        self.connection = connection
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        uri: str,
        *,
        additional_headers: dict[str, str],
        open_timeout: float,
        close_timeout: float,
    ):
        self.calls.append(
            {
                "uri": uri,
                "additional_headers": dict(additional_headers),
                "open_timeout": open_timeout,
                "close_timeout": close_timeout,
            }
        )
        return self.connection


def test_metadata_and_protocol_are_paced_only() -> None:
    module = _api()
    connection = FakeAsyncConnection()
    adapter = module.VLLMPacedRealtimeAdapter(
        "model",
        base_url="https://asr.example/v1/",
        api_key="secret",
        timeout_seconds=12.5,
        connect_factory=FakeConnectFactory(connection),
    )

    assert isinstance(adapter, PacedStreamingASRAdapter)
    assert not isinstance(adapter, StreamingASRAdapter)
    assert adapter.decoding_options == {
        "base_url": "https://asr.example/v1",
        "websocket_endpoint": "wss://asr.example/v1/realtime",
        "input_format": "pcm16",
        "sample_rate_hz": 16000,
        "timeout_seconds": 12.5,
    }
    assert "secret" not in repr(adapter.decoding_options)


def test_send_and_receive_are_interleaved_and_deltas_are_cumulative() -> None:
    module = _api()
    connection = FakeAsyncConnection()
    factory = FakeConnectFactory(connection)
    adapter = module.VLLMPacedRealtimeAdapter(
        "model",
        base_url="http://localhost:8000/v1",
        api_key="secret-key",
        connect_factory=factory,
    )

    async def frames() -> AsyncIterator[AudioBuffer]:
        yield _frame([-2.0, -1.0, -0.5, 0.0])
        await connection.allow_second.wait()
        yield _frame([0.5, 1.0, 2.0])

    async def collect():
        return [update async for update in adapter.stream_duplex(frames())]

    updates = asyncio.run(collect())

    assert connection.delta_before_second_append is True
    assert [(item.text, item.is_final) for item in updates] == [
        ("hel", False),
        ("hello", False),
        ("hello!", True),
    ]
    assert connection.sent[0] == {"type": "session.update", "model": "model"}
    assert connection.sent[1] == {"type": "input_audio_buffer.commit"}
    assert connection.sent[-1] == {"type": "input_audio_buffer.commit", "final": True}
    append_events = [
        item for item in connection.sent if item.get("type") == "input_audio_buffer.append"
    ]
    assert len(append_events) == 2
    first_pcm = np.frombuffer(base64.b64decode(append_events[0]["audio"]), dtype="<i2")
    second_pcm = np.frombuffer(base64.b64decode(append_events[1]["audio"]), dtype="<i2")
    np.testing.assert_array_equal(first_pcm, [-32767, -32767, -16384, 0])
    np.testing.assert_array_equal(second_pcm, [16384, 32767, 32767])
    assert factory.calls[0] == {
        "uri": "ws://localhost:8000/v1/realtime",
        "additional_headers": {"Authorization": "Bearer secret-key"},
        "open_timeout": 60.0,
        "close_timeout": 60.0,
    }
    assert connection.entered is True
    assert connection.exited is True


def test_done_is_not_exposed_until_sender_finishes() -> None:
    module = _api()

    class EarlyDoneConnection:
        def __init__(self) -> None:
            self.sent: list[dict[str, object]] = []
            self.recv_calls = 0
            self.first_append = asyncio.Event()
            self.done_returned = asyncio.Event()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def send(self, message: str) -> None:
            payload = json.loads(message)
            self.sent.append(payload)
            if payload.get("type") == "input_audio_buffer.append":
                self.first_append.set()

        async def recv(self) -> object:
            self.recv_calls += 1
            if self.recv_calls == 1:
                return _event("session.created")
            await self.first_append.wait()
            self.done_returned.set()
            return _event("transcription.done", text="done")

    async def scenario() -> None:
        connection = EarlyDoneConnection()
        gate = asyncio.Event()
        adapter = module.VLLMPacedRealtimeAdapter(
            "model",
            base_url="http://localhost:8000/v1",
            connect_factory=FakeConnectFactory(connection),
        )

        async def frames():
            yield _frame([0.0])
            await gate.wait()
            yield _frame([0.0])

        async def collect():
            return [update async for update in adapter.stream_duplex(frames())]

        task = asyncio.create_task(collect())
        await connection.done_returned.wait()
        await asyncio.sleep(0)
        assert not task.done()
        gate.set()
        updates = await task
        assert [(item.text, item.is_final) for item in updates] == [("done", True)]
        assert connection.sent[-1] == {"type": "input_audio_buffer.commit", "final": True}

    asyncio.run(scenario())


def test_receive_failure_cancels_and_drains_sender() -> None:
    module = _api()

    class ErrorConnection:
        def __init__(self) -> None:
            self.recv_calls = 0
            self.first_append = asyncio.Event()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def send(self, message: str) -> None:
            payload = json.loads(message)
            if payload.get("type") == "input_audio_buffer.append":
                self.first_append.set()

        async def recv(self) -> object:
            self.recv_calls += 1
            if self.recv_calls == 1:
                return _event("session.created")
            await self.first_append.wait()
            return _event("error", error={"message": "secret body"})

    async def scenario() -> None:
        connection = ErrorConnection()
        cancelled = asyncio.Event()
        adapter = module.VLLMPacedRealtimeAdapter(
            "model",
            base_url="http://localhost:8000/v1",
            connect_factory=FakeConnectFactory(connection),
        )

        async def frames():
            try:
                yield _frame([0.0])
                await asyncio.Future()
            finally:
                cancelled.set()

        with pytest.raises(module.VLLMRealtimeError, match="server returned an error"):
            _ = [update async for update in adapter.stream_duplex(frames())]
        assert cancelled.is_set()

    asyncio.run(scenario())


def test_invalid_frame_failure_is_propagated_from_sender() -> None:
    module = _api()

    class WaitingConnection:
        def __init__(self) -> None:
            self.recv_calls = 0
            self.never = asyncio.Event()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def send(self, message: str) -> None:
            return None

        async def recv(self) -> object:
            self.recv_calls += 1
            if self.recv_calls == 1:
                return _event("session.created")
            await self.never.wait()
            raise AssertionError("unreachable")

    adapter = module.VLLMPacedRealtimeAdapter(
        "model",
        base_url="http://localhost:8000/v1",
        connect_factory=FakeConnectFactory(WaitingConnection()),
    )

    async def frames():
        yield _frame([0.0], sample_rate=8_000)

    async def collect():
        return [update async for update in adapter.stream_duplex(frames())]

    with pytest.raises(module.VLLMRealtimeError, match="16000"):
        asyncio.run(collect())


def test_language_is_rejected_before_connection() -> None:
    module = _api()

    class NeverConnection:
        pass

    factory = FakeConnectFactory(NeverConnection())
    adapter = module.VLLMPacedRealtimeAdapter(
        "model",
        base_url="http://localhost:8000/v1",
        connect_factory=factory,
    )

    async def frames():
        yield _frame([0.0])

    async def collect():
        return [update async for update in adapter.stream_duplex(frames(), language="ru")]

    with pytest.raises(module.VLLMRealtimeError, match="language"):
        asyncio.run(collect())
    assert factory.calls == []


def test_server_error_is_sanitized() -> None:
    module = _api()
    secret = "super-secret-123"

    class ErrorConnection:
        def __init__(self) -> None:
            self.recv_calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def send(self, message: str) -> None:
            return None

        async def recv(self) -> object:
            self.recv_calls += 1
            if self.recv_calls == 1:
                return _event("session.created")
            return _event("error", error={"message": f"bad {secret}"})

    adapter = module.VLLMPacedRealtimeAdapter(
        "model",
        base_url="http://localhost:8000/v1",
        api_key=secret,
        connect_factory=FakeConnectFactory(ErrorConnection()),
    )

    async def frames():
        yield _frame([0.0])
        await asyncio.Future()

    async def collect():
        return [update async for update in adapter.stream_duplex(frames())]

    with pytest.raises(module.VLLMRealtimeError) as captured:
        asyncio.run(collect())
    assert "server returned an error" in str(captured.value)
    assert secret not in str(captured.value)
    assert "bad" not in str(captured.value)


def test_missing_optional_dependency_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _api()

    def missing(name: str):
        assert name == "websockets.asyncio.client"
        error = ModuleNotFoundError("No module named 'websockets'")
        error.name = "websockets"
        raise error

    monkeypatch.setattr(module, "import_module", missing)

    with pytest.raises(module.VLLMRealtimeError, match="uv sync --extra vllm-realtime"):
        module.VLLMPacedRealtimeAdapter("model", base_url="http://localhost:8000/v1")
