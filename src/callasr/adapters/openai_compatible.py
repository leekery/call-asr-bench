"""OpenAI-compatible HTTP speech-to-text adapter."""

from __future__ import annotations

import io
import wave
from importlib import import_module
from math import isfinite
from numbers import Real
from typing import Protocol
from urllib.parse import urlsplit

import numpy as np

from callasr.adapters.base import AdapterError, AdapterOption, Transcription
from callasr.audio import AudioBuffer


class _Response(Protocol):
    status_code: int

    def json(self) -> object: ...


class _HttpClient(Protocol):
    def post(
        self,
        url: str,
        *,
        data: dict[str, str],
        files: dict[str, tuple[str, bytes, str]],
        headers: dict[str, str],
        timeout: float,
    ) -> _Response: ...


class _HttpxClient:
    def __init__(self, httpx_module: object) -> None:
        self._httpx = httpx_module

    def post(
        self,
        url: str,
        *,
        data: dict[str, str],
        files: dict[str, tuple[str, bytes, str]],
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        httpx = self._httpx
        try:
            return httpx.post(
                url,
                data=data,
                files=files,
                headers=headers,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise TimeoutError from exc
        except httpx.HTTPError as exc:
            raise OSError from exc


def _normalize_base_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url or base_url != base_url.strip():
        raise AdapterError("base_url must be a valid HTTP(S) API root")
    try:
        parsed = urlsplit(base_url)
    except ValueError as exc:
        raise AdapterError("base_url must be a valid HTTP(S) API root") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise AdapterError("base_url must be a valid HTTP(S) API root without credentials/query")
    return base_url.rstrip("/")


def _audio_to_pcm16_wav(audio: AudioBuffer) -> bytes:
    clipped = np.clip(audio.samples.astype(np.float64), -1.0, 1.0)
    pcm16 = np.rint(clipped * 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(audio.sample_rate)
        writer.writeframes(pcm16.tobytes())
    return output.getvalue()


class OpenAICompatibleAdapter:
    """ASR adapter for OpenAI-compatible audio transcription endpoints."""

    name = "openai-compatible"
    device = "remote"
    compute_type = "server"

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        client: _HttpClient | None = None,
    ) -> None:
        if not isinstance(model, str) or not model:
            raise AdapterError("model must be a non-empty string")
        if (
            not isinstance(timeout_seconds, Real)
            or isinstance(timeout_seconds, bool)
            or not isfinite(timeout_seconds)
            or timeout_seconds <= 0.0
        ):
            raise AdapterError("timeout_seconds must be a finite positive number")

        self.model = model
        self.base_url = _normalize_base_url(base_url)
        self.timeout_seconds = float(timeout_seconds)
        self._api_key = api_key if api_key else None
        self._client = client if client is not None else self._build_default_client()

    def _build_default_client(self) -> _HttpClient:
        try:
            httpx = import_module("httpx")
        except ModuleNotFoundError as exc:
            if exc.name != "httpx":
                raise
            raise AdapterError(
                "OpenAI-compatible adapter requires optional dependency 'httpx'; "
                "install it with 'uv sync --extra openai-compatible'"
            ) from exc
        return _HttpxClient(httpx)

    @property
    def decoding_options(self) -> dict[str, AdapterOption]:
        return {
            "base_url": self.base_url,
            "response_format": "json",
            "upload_format": "wav_pcm16",
            "timeout_seconds": self.timeout_seconds,
        }

    def transcribe(
        self,
        audio: AudioBuffer,
        language: str | None = None,
    ) -> Transcription:
        data = {"model": self.model, "response_format": "json"}
        if language is not None:
            data["language"] = language
        headers = {}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = self._client.post(
                f"{self.base_url}/audio/transcriptions",
                data=data,
                files={"file": ("audio.wav", _audio_to_pcm16_wav(audio), "audio/wav")},
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except TimeoutError as exc:
            raise AdapterError("OpenAI-compatible transcription request timed out") from exc
        except OSError as exc:
            raise AdapterError("OpenAI-compatible transcription request failed") from exc

        if not 200 <= response.status_code < 300:
            raise AdapterError(f"OpenAI-compatible endpoint returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise AdapterError("OpenAI-compatible endpoint returned invalid JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise AdapterError("OpenAI-compatible JSON response must contain string 'text'")
        return Transcription(payload["text"])
