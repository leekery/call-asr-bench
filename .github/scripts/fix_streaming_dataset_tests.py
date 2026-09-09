from pathlib import Path

path = Path("tests/test_streaming_dataset.py")
text = path.read_text(encoding="utf-8")
text = text.replace("from callasr.audio import AudioBuffer\n", "")
text = text.replace(
    "def test_streaming_dataset_runner_aggregates_exact_metrics_in_manifest_order(tmp_path: Path) -> None:",
    "def test_streaming_dataset_runner_aggregates_exact_metrics_in_manifest_order(\n    tmp_path: Path,\n) -> None:",
)
old = '''    assert [item.time_to_first_partial_seconds for item in result.items] == pytest.approx(
        [0.1, None, 0.3], nan_ok=True
    )'''
new = '''    ttft = [item.time_to_first_partial_seconds for item in result.items]
    assert ttft[0] == pytest.approx(0.1)
    assert ttft[1] is None
    assert ttft[2] == pytest.approx(0.3)'''
if text.count(old) != 1:
    raise SystemExit("TTFT assertion block not found")
text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
