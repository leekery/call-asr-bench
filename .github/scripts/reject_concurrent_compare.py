from pathlib import Path

path = Path("src/callasr/report.py")
text = path.read_text(encoding="utf-8")
old = '''def _row_from_payload(path: Path, payload: dict[str, object]) -> ComparisonRow:\n    schema = _schema_version(payload, path)'''
new = '''def _row_from_payload(path: Path, payload: dict[str, object]) -> ComparisonRow:\n    if payload.get("kind") == "concurrent":\n        raise _fail(path, "concurrent artifacts are not supported by batch comparison")\n    schema = _schema_version(payload, path)'''
if text.count(old) != 1:
    raise SystemExit("comparison row anchor not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
