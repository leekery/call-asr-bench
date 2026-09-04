import json
from pathlib import Path

import pytest


def _write_manifest(path: Path, *items: dict[str, object] | str) -> Path:
    lines = [
        item if isinstance(item, str) else json.dumps(item, ensure_ascii=False) for item in items
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_dataset_manifest_resolves_relative_audio_and_preserves_order(tmp_path: Path) -> None:
    from callasr.dataset import load_dataset_manifest

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    first_audio = audio_dir / "first.wav"
    second_audio = audio_dir / "second.wav"
    first_audio.touch()
    second_audio.touch()
    manifest = _write_manifest(
        tmp_path / "dataset.jsonl",
        "",
        {
            "id": "call-001",
            "audio": "audio/first.wav",
            "reference": "добрый день",
            "language": "ru",
        },
        {"id": "call-002", "audio": "audio/second.wav", "reference": ""},
    )

    items = load_dataset_manifest(manifest)

    assert isinstance(items, tuple)
    assert [item.id for item in items] == ["call-001", "call-002"]
    assert items[0].audio == first_audio.resolve()
    assert items[0].reference == "добрый день"
    assert items[0].language == "ru"
    assert items[0].line_number == 2
    assert items[1].audio == second_audio.resolve()
    assert items[1].reference == ""
    assert items[1].language is None
    assert items[1].line_number == 3


def test_dataset_item_is_immutable(tmp_path: Path) -> None:
    from dataclasses import FrozenInstanceError

    from callasr.dataset import load_dataset_manifest

    audio = tmp_path / "sample.wav"
    audio.touch()
    manifest = _write_manifest(
        tmp_path / "dataset.jsonl",
        {"id": "call-001", "audio": "sample.wav", "reference": "text"},
    )

    (item,) = load_dataset_manifest(manifest)

    with pytest.raises(FrozenInstanceError):
        item.reference = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["{not-json}"], "invalid JSON"),
        ([{"id": "call-001", "audio": "sample.wav"}], "missing required field 'reference'"),
        (
            [{"id": "call-001", "audio": "sample.wav", "reference": "x", "langauge": "ru"}],
            "unknown field 'langauge'",
        ),
        ([{"id": "", "audio": "sample.wav", "reference": "x"}], "id must be a non-empty string"),
        (
            [{"id": "call-001", "audio": "", "reference": "x"}],
            "audio must be a non-empty path string",
        ),
        (
            [{"id": "call-001", "audio": "sample.wav", "reference": 42}],
            "reference must be a string",
        ),
        (
            [{"id": "call-001", "audio": "sample.wav", "reference": "x", "language": "RU"}],
            "language must be a lowercase ISO 639-1 code",
        ),
        (
            [{"id": "call-001", "audio": "sample.wav", "reference": "x", "language": None}],
            "language must be a lowercase ISO 639-1 code",
        ),
    ],
)
def test_manifest_validation_errors_include_path_and_line(
    tmp_path: Path, lines: list[dict[str, object] | str], expected: str
) -> None:
    from callasr.dataset import DatasetError, load_dataset_manifest

    (tmp_path / "sample.wav").touch()
    manifest = _write_manifest(tmp_path / "dataset.jsonl", "", *lines)

    with pytest.raises(DatasetError) as exc_info:
        load_dataset_manifest(manifest)

    message = str(exc_info.value)
    assert str(manifest.resolve()) in message
    assert ":2:" in message
    assert expected in message


def test_manifest_rejects_duplicate_ids_with_second_line_context(tmp_path: Path) -> None:
    from callasr.dataset import DatasetError, load_dataset_manifest

    audio = tmp_path / "sample.wav"
    audio.touch()
    item = {"id": "call-001", "audio": "sample.wav", "reference": "text"}
    manifest = _write_manifest(tmp_path / "dataset.jsonl", item, item)

    with pytest.raises(DatasetError, match="duplicate id 'call-001'") as exc_info:
        load_dataset_manifest(manifest)

    assert f"{manifest.resolve()}:2:" in str(exc_info.value)


def test_manifest_rejects_missing_audio_file_with_line_context(tmp_path: Path) -> None:
    from callasr.dataset import DatasetError, load_dataset_manifest

    manifest = _write_manifest(
        tmp_path / "dataset.jsonl",
        {"id": "call-001", "audio": "missing.wav", "reference": "text"},
    )

    with pytest.raises(DatasetError, match="audio file does not exist") as exc_info:
        load_dataset_manifest(manifest)

    assert f"{manifest.resolve()}:1:" in str(exc_info.value)
