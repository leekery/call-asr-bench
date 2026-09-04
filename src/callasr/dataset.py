"""Dataset manifest parsing and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_REQUIRED_FIELDS = ("id", "audio", "reference")
_ALLOWED_FIELDS = frozenset((*_REQUIRED_FIELDS, "language"))


class DatasetError(ValueError):
    """A dataset manifest validation error with source context."""

    def __init__(
        self,
        manifest_path: Path,
        message: str,
        *,
        line_number: int | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.line_number = line_number
        prefix = str(manifest_path)
        if line_number is not None:
            prefix = f"{prefix}:{line_number}"
        super().__init__(f"{prefix}: {message}")


@dataclass(frozen=True, slots=True)
class DatasetItem:
    """One validated benchmark utterance from a JSONL manifest."""

    id: str
    audio: Path
    reference: str
    language: str | None
    line_number: int


def _raise_line_error(manifest_path: Path, line_number: int, message: str) -> None:
    raise DatasetError(manifest_path, message, line_number=line_number)


def _validate_language(manifest_path: Path, line_number: int, language: object) -> str:
    if (
        not isinstance(language, str)
        or len(language) != 2
        or not language.isascii()
        or not language.isalpha()
        or not language.islower()
    ):
        _raise_line_error(
            manifest_path,
            line_number,
            "language must be a lowercase ISO 639-1 code",
        )
    return language


def _parse_item(
    manifest_path: Path,
    line_number: int,
    value: object,
    seen_ids: set[str],
) -> DatasetItem:
    if not isinstance(value, dict):
        _raise_line_error(manifest_path, line_number, "manifest item must be a JSON object")

    unknown_fields = sorted(set(value) - _ALLOWED_FIELDS)
    if unknown_fields:
        field = unknown_fields[0]
        _raise_line_error(manifest_path, line_number, f"unknown field '{field}'")

    for field in _REQUIRED_FIELDS:
        if field not in value:
            _raise_line_error(manifest_path, line_number, f"missing required field '{field}'")

    item_id = value["id"]
    if not isinstance(item_id, str) or not item_id.strip():
        _raise_line_error(manifest_path, line_number, "id must be a non-empty string")
    if item_id in seen_ids:
        _raise_line_error(manifest_path, line_number, f"duplicate id '{item_id}'")

    audio_value = value["audio"]
    if not isinstance(audio_value, str) or not audio_value.strip():
        _raise_line_error(manifest_path, line_number, "audio must be a non-empty path string")
    audio_path = (manifest_path.parent / audio_value).resolve()
    if not audio_path.is_file():
        _raise_line_error(
            manifest_path,
            line_number,
            f"audio file does not exist: {audio_value}",
        )

    reference = value["reference"]
    if not isinstance(reference, str):
        _raise_line_error(manifest_path, line_number, "reference must be a string")

    language = None
    if "language" in value:
        language = _validate_language(manifest_path, line_number, value["language"])
    seen_ids.add(item_id)
    return DatasetItem(
        id=item_id,
        audio=audio_path,
        reference=reference,
        language=language,
        line_number=line_number,
    )


def load_dataset_manifest(path: str | Path) -> tuple[DatasetItem, ...]:
    """Load and strictly validate a UTF-8 JSONL benchmark manifest."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DatasetError(manifest_path, f"cannot read manifest: {exc}") from exc

    items: list[DatasetItem] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(
                manifest_path,
                f"invalid JSON: {exc.msg}",
                line_number=line_number,
            ) from exc
        items.append(_parse_item(manifest_path, line_number, value, seen_ids))

    return tuple(items)
