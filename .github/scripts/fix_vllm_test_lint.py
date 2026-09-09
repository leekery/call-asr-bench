from pathlib import Path

path = Path("tests/test_vllm_realtime_adapter.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    '    connection = FakeConnection([_event("session.created"), _event("transcription.done", text="ok")])',
    '    connection = FakeConnection(\n        [_event("session.created"), _event("transcription.done", text="ok")]\n    )',
)
text = text.replace(
    '    connection = FakeConnection([_event("session.created"), _event("transcription.delta", delta="hi")])',
    '    connection = FakeConnection(\n        [_event("session.created"), _event("transcription.delta", delta="hi")]\n    )',
)
text = text.replace(
    'match="before transcription.done"',
    'match=r"before transcription\\.done"',
)
path.write_text(text, encoding="utf-8")
