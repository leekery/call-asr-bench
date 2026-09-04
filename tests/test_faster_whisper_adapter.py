from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from callasr.audio import AudioBuffer


def _api():
    try:
        from callasr.adapters.base import AdapterError, ASRAdapter, Transcription
        from callasr.adapters.faster_whisper import FasterWhisperAdapter
    except ModuleNotFoundError as exc:
        pytest.fail(f"adapter implementation is missing: {exc}")
    return ASRAdapter, AdapterError, Transcription, FasterWhisperAdapter


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, dict[str, object]]] = []

    def transcribe(self, samples: np.ndarray, **kwargs: object):
        self.calls.append((samples, kwargs))
        return iter([SimpleNamespace(text=" hello"), SimpleNamespace(text=" world ")]), object()


class FakeWhisperModule:
    def __init__(self) -> None:
        self.model = FakeModel()
        self.constructed_with: tuple[str, str, str] | None = None

    def WhisperModel(self, model: str, *, device: str, compute_type: str) -> FakeModel:
        self.constructed_with = (model, device, compute_type)
        return self.model


def test_transcription_is_immutable() -> None:
    _, _, Transcription, _ = _api()
    result = Transcription(text="hello")
    with pytest.raises(FrozenInstanceError):
        result.text = "changed"


def test_adapter_exposes_stable_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    ASRAdapter, _, _, FasterWhisperAdapter = _api()
    fake_module = FakeWhisperModule()
    monkeypatch.setattr("callasr.adapters.faster_whisper.import_module", lambda name: fake_module)

    adapter = FasterWhisperAdapter(
        model="large-v3",
        device="cuda",
        compute_type="float16",
    )

    assert isinstance(adapter, ASRAdapter)
    assert fake_module.constructed_with == ("large-v3", "cuda", "float16")
    assert adapter.name == "faster-whisper"
    assert adapter.model == "large-v3"
    assert adapter.device == "cuda"
    assert adapter.compute_type == "float16"
    assert adapter.decoding_options == {"beam_size": 5, "temperature": 0.0}


def test_transcribe_resamples_to_16khz_and_propagates_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, Transcription, FasterWhisperAdapter = _api()
    fake_module = FakeWhisperModule()
    monkeypatch.setattr("callasr.adapters.faster_whisper.import_module", lambda name: fake_module)
    adapter = FasterWhisperAdapter(model="large-v3")
    audio = AudioBuffer(np.linspace(-0.5, 0.5, 800, dtype=np.float32), 8_000)

    result = adapter.transcribe(audio, language="ru")

    samples, kwargs = fake_module.model.calls[0]
    assert samples.dtype == np.float32
    assert samples.shape == (1600,)
    assert kwargs == {"language": "ru", "beam_size": 5, "temperature": 0.0}
    assert result == Transcription(text="hello world")


def test_transcribe_passes_none_language_for_autodetection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, FasterWhisperAdapter = _api()
    fake_module = FakeWhisperModule()
    monkeypatch.setattr("callasr.adapters.faster_whisper.import_module", lambda name: fake_module)
    adapter = FasterWhisperAdapter(model="small")

    adapter.transcribe(AudioBuffer(np.zeros(160, dtype=np.float32), 16_000))

    _, kwargs = fake_module.model.calls[0]
    assert kwargs["language"] is None


def test_missing_optional_dependency_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, AdapterError, _, FasterWhisperAdapter = _api()

    def missing_module(name: str):
        raise ModuleNotFoundError("No module named 'faster_whisper'", name=name)

    monkeypatch.setattr("callasr.adapters.faster_whisper.import_module", missing_module)

    with pytest.raises(AdapterError, match=r"uv sync --extra faster-whisper"):
        FasterWhisperAdapter(model="large-v3")


def test_transitive_missing_dependency_is_not_misreported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, FasterWhisperAdapter = _api()

    def broken_dependency(name: str):
        raise ModuleNotFoundError("No module named 'ctranslate2'", name="ctranslate2")

    monkeypatch.setattr("callasr.adapters.faster_whisper.import_module", broken_dependency)

    with pytest.raises(ModuleNotFoundError, match="ctranslate2"):
        FasterWhisperAdapter(model="large-v3")
