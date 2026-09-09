"""Streaming ASR adapter for vLLM's Realtime WebSocket API."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable
from importlib import import_module
from math import isfinite
from numbers import Real
from typing import Protocol
from urllib.parse import urlparse, urlunparse

import numpy as np

from callasr.adapters.base import AdapterOption
from callasr.audio import AudioBuffer
from callasr.streaming import StreamingError, StreamingUpdate


class VLLMRealtimeError(StreamingError):
    """A user-actionable vLLM Realtime adapter or protocol error."""


class _RealtimeConnection(Protocol):
    def __enter__(self) -> _RealtimeConnection: ...

    def __exit__(self, exc_type, exc, traceback) -> None: ...

    def send(self, message: str) -> None: ...

    def recv(self, timeout: float | None = None) -> object: ...


class _ConnectFactory(Protocol):
    def __call__(
        self,
        uri: str,
        *,
        additional_headers: dict[str, str],
        open_timeout: float,
        close_timeout: float,
    ) -> _RealtimeConnection: ...


def _normalize_base_url(base_url: str) -> tuple[str, str]:
    if not isinstance(base_url, str) or not base_url or base_url != base_url.strip():
        raise VLLMRealtimeError("base_url must be an absolute HTTP(S) URL")

    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise VLLMRealtimeError("base_url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise VLLMRealtimeError("base_url must not contain userinfo or credentials")
    if parsed.query or parsed.fragment or parsed.params:
        raise VLLMRealtimeError("base_url must not contain query parameters or a fragment")

    path = parsed.path.rstrip("/")
    normalized = urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    websocket_scheme = "wss" if parsed.scheme == "https" else "ws"
    websocket_path = f"{path}/realtime" if path else "/realtime"
    endpoint = urlunparse((websocket_scheme, parsed.netloc, websocket_path, "", "", ""))
    return normalized, endpoint


def _default_connect_factory() -> _ConnectFactory:
    try:
        module = import_module("websockets.sync.client")
    except ModuleNotFoundError as exc:
        if exc.name != "websockets":
            raise
        raise VLLMRealtimeError(
            "vLLM Realtime support requires the optional 'vllm-realtime' dependencies; "
            "install them with `uv sync --extra vllm-realtime`"
        ) from exc

    connect = getattr(module, "connect", None)
    if not callable(connect):
        raise VLLMRealtimeError("installed websockets package does not expose sync client connect")
    return connect


def _validate_timeout(timeout_seconds: object) -> float:
    if (
        not isinstance(timeout_seconds, Real)
        or isinstance(timeout_seconds, bool)
        or not isfinite(timeout_seconds)
        or timeout_seconds <= 0.0
    ):
        raise VLLMRealtimeError("timeout_seconds must be a finite positive number")
    return float(timeout_seconds)


def _pcm16_base64(frame: AudioBuffer) -> str:
    clipped = np.clip(frame.samples.astype(np.float64, copy=False), -1.0, 1.0)
    pcm16 = np.rint(clipped * 32767.0).astype("<i2", copy=False)
    return base64.b64encode(pcm16.tobytes()).decode("ascii")


def _decode_event(raw: object) -> dict[str, object]:
    if not isinstance(raw, str):
        raise VLLMRealtimeError("vLLM Realtime protocol message must be JSON text")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VLLMRealtimeError("vLLM Realtime protocol returned invalid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
        raise VLLMRealtimeError("vLLM Realtime protocol message is missing a string type")
    return payload


def _send(connection: _RealtimeConnection, payload: dict[str, object]) -> None:
    try:
        connection.send(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    except Exception as exc:
        raise VLLMRealtimeError("vLLM Realtime connection failed while sending audio") from exc


def _recv(connection: _RealtimeConnection, timeout_seconds: float) -> dict[str, object]:
    try:
        raw = connection.recv(timeout=timeout_seconds)
    except (StopIteration, EOFError) as exc:
        raise VLLMRealtimeError(
            "vLLM Realtime connection closed before transcription.done"
        ) from exc
    except Exception as exc:
        raise VLLMRealtimeError("vLLM Realtime connection failed while receiving events") from exc
    return _decode_event(raw)


class VLLMRealtimeAdapter:
    """Synchronous adapter for vLLM's file-style Realtime transcription sequence."""

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
        connect_factory: _ConnectFactory | None = None,
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
            _default_connect_factory() if connect_factory is None else connect_factory
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

    def stream(
        self,
        frames: Iterable[AudioBuffer],
        language: str | None = None,
    ) -> Iterable[StreamingUpdate]:
        if language is not None:
            raise VLLMRealtimeError(
                "vLLM Realtime does not currently expose a language setting; language must be None"
            )

        materialized_frames = tuple(frames)
        if not materialized_frames:
            raise VLLMRealtimeError("vLLM Realtime requires at least one audio frame")
        for frame in materialized_frames:
            if not isinstance(frame, AudioBuffer):
                raise VLLMRealtimeError("vLLM Realtime frames must be AudioBuffer values")
            if frame.sample_rate != 16_000:
                raise VLLMRealtimeError("vLLM Realtime requires 16000 Hz audio frames")

        headers = {} if self._api_key is None else {"Authorization": f"Bearer {self._api_key}"}
        try:
            context = self._connect_factory(
                self.websocket_endpoint,
                additional_headers=headers,
                open_timeout=self.timeout_seconds,
                close_timeout=self.timeout_seconds,
            )
            with context as connection:
                first_event = _recv(connection, self.timeout_seconds)
                first_type = first_event["type"]
                if first_type == "error":
                    raise VLLMRealtimeError("vLLM Realtime server returned an error")
                if first_type != "session.created":
                    raise VLLMRealtimeError(
                        "vLLM Realtime protocol expected session.created as the first event"
                    )

                _send(connection, {"type": "session.update", "model": self.model})
                _send(connection, {"type": "input_audio_buffer.commit"})
                for frame in materialized_frames:
                    _send(
                        connection,
                        {
                            "type": "input_audio_buffer.append",
                            "audio": _pcm16_base64(frame),
                        },
                    )
                _send(connection, {"type": "input_audio_buffer.commit", "final": True})

                accumulated = ""
                while True:
                    event = _recv(connection, self.timeout_seconds)
                    event_type = event["type"]
                    if event_type == "error":
                        raise VLLMRealtimeError("vLLM Realtime server returned an error")
                    if event_type == "transcription.delta":
                        delta = event.get("delta")
                        if not isinstance(delta, str):
                            raise VLLMRealtimeError(
                                "vLLM Realtime protocol transcription.delta requires string delta"
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
                        yield StreamingUpdate(text=text, is_final=True)
                        return
        except VLLMRealtimeError:
            raise
        except Exception as exc:
            raise VLLMRealtimeError("vLLM Realtime connection failed") from exc
