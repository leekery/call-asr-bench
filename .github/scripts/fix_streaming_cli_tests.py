from pathlib import Path

path = Path("tests/test_streaming_cli.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "def _result(manifest: Path, *, frame_duration_ms: int, language_mode: str) -> StreamingDatasetResult:",
    "def _result(\n    manifest: Path,\n    *,\n    frame_duration_ms: int,\n    language_mode: str,\n) -> StreamingDatasetResult:",
)
old = '''    monkeypatch.setattr(
        cli,
        "VLLMRealtimeAdapter",
        lambda model, **kwargs: FakeVLLMAdapter(model, kwargs["base_url"], kwargs["timeout_seconds"]),
        raising=False,
    )'''
new = '''    def constructor(model: str, **kwargs):
        return FakeVLLMAdapter(model, kwargs["base_url"], kwargs["timeout_seconds"])

    monkeypatch.setattr(cli, "VLLMRealtimeAdapter", constructor, raising=False)'''
if text.count(old) != 2:
    raise SystemExit(f"expected two long constructor blocks, got {text.count(old)}")
text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
