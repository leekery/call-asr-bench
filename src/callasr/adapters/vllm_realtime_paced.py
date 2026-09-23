"""Async paced adapter for vLLM's Realtime WebSocket API."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, AsyncIterator
from importlib import import_module
from typing import Protocol

from callasr.adapters.base import AdapterOption
from callasr.adapters.vllm_realtime import (
    VLLMRealtimeError,
    _decode_event,
    _normalize_base_url,
    _pcm16_base64,
    _validate_timeout,
)
from callasr.audio import AudioBuffer
from callasr.streaming import StreamingUpdate


class _AsyncRealtimeConnection(Protocol):
    async def send(self, message: str) -> None: ...

    async def recv(self) -> object: ...


class _AsyncConnectionContext(Protocol):
    async def __aenter__(self) -> _AsyncRealtimeConnection: ...

    async def __aexit__(self, exc_type, exc, traceback) -> None: ...


class _AsyncConnectFactory(Protocol):
    def __call__(
        self,
        uri: str,
        *,
        additional_headers: dict[str, str],
        open_timeout: float,
        close_timeout: float,
    ) -> _AsyncConnectionContext: ...


def _default_async_connect_factory() -> _AsyncConnectFactory:
    try:
        module = import_module("websockets.asyncio.client")
    except ModuleNotFoundError as exc:
        if exc.name != "websockets":
            raise
        raise VLLMRealtimeError(
            "vLLM Realtime support requires the optional 'vllm-realtime' dependencies; "
            "install them with `uv sync --extra vllm-realtime`"
        ) from exc

    connect = getattr(module, "connect", None)
    if not callable(connect):
        raise VLLMRealtimeError(
            "installed websockets package does not expose asyncio client connect"
        )
    return connect


async def _send(connection: _AsyncRealtimeConnection, payload: dict[str, object]) -> None:
    try:
        await connection.send(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    except Exception as exc:
        raise VLLMRealtimeError("vLLM Realtime connection failed while sending audio") from exc


async def _recv(
    connection: _AsyncRealtimeConnection,
    timeout_seconds: float | None,
) -> dict[str, object]:
    try:
        if timeout_seconds is None:
            raw = await connection.recv()
        else:
            raw = await asyncio.wait_for(connection.recv(), timeout=timeout_seconds)
    except (StopAsyncIteration, EOFError) as exc:
        raise VLLMRealtimeError(
            "vLLM Realtime connection closed before transcription.done"
        ) from exc
    except asyncio.TimeoutError as exc:
        raise VLLMRealtimeError("vLLM Realtime timed out while waiting for a server event") from exc
    except Exception as exc:
        raise VLLMRealtimeError("vLLM Realtime connection failed while receiving events") from exc
    return _decode_event(raw)


async def _timeout_after(timeout_seconds: float) -> None:
    """Wait for a response timeout after the paced sender has completed."""

    await asyncio.sleep(timeout_seconds)


async def _cancel_and_drain(task: asyncio.Task[None] | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _recv_while_sender_runs(
    connection: _AsyncRealtimeConnection,
    timeout_seconds: float,
    sender_task: asyncio.Task[None],
) -> dict[str, object]:
    receive_task = asyncio.create_task(_recv(connection, None))
    timeout_task: asyncio.Task[None] | None = None
    try:
        done, _ = await asyncio.wait(
            {receive_task, sender_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if sender_task in done:
            try:
                await sender_task
            except BaseException:
                if not receive_task.done():
                    receive_task.cancel()
                await asyncio.gather(receive_task, return_exceptions=True)
                raise

        if receive_task.done():
            return await receive_task

        # The sender has now exhausted the paced input and sent final commit.
        # Start the response timeout here, rather than at the beginning of a
        # potentially long audio submission.
        timeout_task = asyncio.create_task(_timeout_after(timeout_seconds))
        done, _ = await asyncio.wait(
            {receive_task, timeout_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if receive_task in done:
            await _cancel_and_drain(timeout_task)
            return await receive_task
        raise VLLMRealtimeError("vLLM Realtime timed out waiting for a server event")
    except BaseException:
        if not receive_task.done():
            receive_task.cancel()
        if timeout_task is not None and not timeout_task.done():
            timeout_task.cancel()
        await asyncio.gather(receive_task, return_exceptions=True)
        if timeout_task is not None:
            await asyncio.gather(timeout_task, return_exceptions=True)
        raise


class VLLMPacedRealtimeAdapter:
    """Async full-duplex adapter for paced vLLM Realtime transcription."""

    name = "vllm-realtime"
    device = "remote"
    compute_type = "server"

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        connect_factory: _AsyncConnectFactory | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise VLLMRealtimeError("model must be a non-empty string")
        if api_key is not None and (not isinstance(api_key, str) or not api_key):
            raise VLLMRealtimeError("api_key must be a non-empty string when provided")

        normalized_base_url, websocket_endpoint = _normalize_base_url(base_url)
        self.model = model
        self.base_url = normalized_base_url
        self.websocket_endpoint = websocket_endpoint
        self.timeout_seconds = _validate_timeout(timeout_seconds)
        self._api_key = api_key
        self._connect_factory = (
            _default_async_connect_factory() if connect_factory is None else connect_factory
        )

    @property
    def decoding_options(self) -> dict[str, AdapterOption]:
        return {
            "base_url": self.base_url,
            "websocket_endpoint": self.websocket_endpoint,
            "input_format": "pcm16",
            "sample_rate_hz": 16_000,
            "timeout_seconds": self.timeout_seconds,
        }

    async def _send_frames(
        self,
        connection: _AsyncRealtimeConnection,
        frames: AsyncIterable[AudioBuffer],
    ) -> None:
        sent_count = 0
        async for frame in frames:
            if not isinstance(frame, AudioBuffer):
                raise VLLMRealtimeError("vLLM Realtime frames must be AudioBuffer values")
            if frame.sample_rate != 16_000:
                raise VLLMRealtimeError("vLLM Realtime requires 16000 Hz audio frames")
            await _send(
                connection,
                {
                    "type": "input_audio_buffer.append",
                    "audio": _pcm16_base64(frame),
                },
            )
            sent_count += 1
        if sent_count == 0:
            raise VLLMRealtimeError("vLLM Realtime requires at least one audio frame")
        await _send(connection, {"type": "input_audio_buffer.commit", "final": True})

    async def stream_duplex(
        self,
        frames: AsyncIterable[AudioBuffer],
        language: str | None = None,
    ) -> AsyncIterator[StreamingUpdate]:
        if language is not None:
            raise VLLMRealtimeError(
                "vLLM Realtime does not currently expose a language setting; language must be None"
            )

        headers = {} if self._api_key is None else {"Authorization": f"Bearer {self._api_key}"}
        sender_task: asyncio.Task[None] | None = None
        try:
            context = self._connect_factory(
                self.websocket_endpoint,
                additional_headers=headers,
                open_timeout=self.timeout_seconds,
                close_timeout=self.timeout_seconds,
            )
            async with context as connection:
                first_event = await _recv(connection, self.timeout_seconds)
                first_type = first_event["type"]
                if first_type == "error":
                    raise VLLMRealtimeError("vLLM Realtime server returned an error")
                if first_type != "session.created":
                    raise VLLMRealtimeError(
                        "vLLM Realtime protocol expected session.created as the first event"
                    )

                await _send(connection, {"type": "session.update", "model": self.model})
                await _send(connection, {"type": "input_audio_buffer.commit"})
                sender_task = asyncio.create_task(self._send_frames(connection, frames))

                accumulated = ""
                try:
                    while True:
                        event = await _recv_while_sender_runs(
                            connection,
                            self.timeout_seconds,
                            sender_task,
                        )
                        event_type = event["type"]
                        if event_type == "error":
                            raise VLLMRealtimeError("vLLM Realtime server returned an error")
                        if event_type == "transcription.delta":
                            delta = event.get("delta")
                            if not isinstance(delta, str):
                                raise VLLMRealtimeError(
                                    "vLLM Realtime protocol transcription.delta requires "
                                    "string delta"
                                )
                            if delta:
                                accumulated += delta
                                yield StreamingUpdate(text=accumulated, is_final=False)
                            continue
                        if event_type == "transcription.done":
                            text = event.get("text")
                            if not isinstance(text, str):
                                raise VLLMRealtimeError(
                                    "vLLM Realtime protocol transcription.done requires string text"
                                )
                            await sender_task
                            yield StreamingUpdate(text=text, is_final=True)
                            return
                finally:
                    await _cancel_and_drain(sender_task)
        except VLLMRealtimeError:
            raise
        except Exception as exc:
            raise VLLMRealtimeError("vLLM Realtime connection failed") from exc
