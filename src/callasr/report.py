"""Deterministic comparison reports for saved benchmark artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path

_KNOWN_SCHEMAS = {1, 2, 3, 4, 5, 6}
_FINGERPRINT_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MISSING = object()


class ComparisonError(ValueError):
    """A saved benchmark artifact cannot be compared safely."""


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    artifact: str
    schema_version: int
    adapter: str
    model: str
    codec: str
    packet_loss_rate: float
    gain_db: float | None
    clip_threshold: float | None
    snr_db: float | None
    jitter_std_ms: float | None
    playout_buffer_ms: float | None
    wer: float
    cer: float
    rtf: float
    speed_factor: float | None
    numeric_entity_accuracy: float | None
    item_count: int
    dataset_fingerprint: str | None


def _fail(path: Path, message: str) -> ComparisonError:
    return ComparisonError(f"{path}: {message}")


def _mapping(parent: dict[str, object], key: str, path: Path) -> dict[str, object]:
    value = parent.get(key, _MISSING)
    if not isinstance(value, dict):
        raise _fail(path, f"{key} must be an object")
    return value


def _string(parent: dict[str, object], key: str, path: Path) -> str:
    value = parent.get(key, _MISSING)
    if not isinstance(value, str) or not value:
        raise _fail(path, f"{key} must be a non-empty string")
    return value


def _number(
    parent: dict[str, object],
    key: str,
    path: Path,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    optional: bool = False,
) -> float | None:
    value = parent.get(key, _MISSING)
    if optional and value is None:
        return None
    if value is _MISSING:
        raise _fail(path, f"{key} is required")
    if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value):
        raise _fail(path, f"{key} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise _fail(path, f"{key} must be at least {minimum:g}")
    if maximum is not None and result > maximum:
        raise _fail(path, f"{key} must be at most {maximum:g}")
    return result


def _item_count(dataset: dict[str, object], path: Path) -> int:
    value = dataset.get("item_count", _MISSING)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _fail(path, "item_count must be a non-negative integer")
    return value


def _schema_version(payload: dict[str, object], path: Path) -> int:
    value = payload.get("schema_version", _MISSING)
    if not isinstance(value, int) or isinstance(value, bool) or value not in _KNOWN_SCHEMAS:
        raise _fail(path, "unsupported schema_version; expected one of 1, 2, 3, 4, 5, 6")
    return value


def _dataset_fingerprint(
    dataset: dict[str, object],
    schema: int,
    path: Path,
) -> str | None:
    if schema < 5:
        return None
    value = dataset.get("fingerprint", _MISSING)
    if not isinstance(value, str) or _FINGERPRINT_PATTERN.fullmatch(value) is None:
        raise _fail(path, "dataset fingerprint must be lowercase sha256:<64 hex digits>")
    return value


def _codec(channel: dict[str, object], path: Path) -> str:
    value = channel.get("codec", _MISSING)
    if value not in {"none", "pcmu", "pcma"}:
        raise _fail(path, "codec must be one of none, pcmu, pcma")
    return str(value)


def _optional_channel_number(
    channel: dict[str, object],
    key: str,
    path: Path,
    *,
    minimum: float | None = None,
) -> float | None:
    return _number(channel, key, path, minimum=minimum, optional=True)


def _load_payload(path: Path) -> dict[str, object]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _fail(path, "cannot read artifact") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _fail(path, "invalid JSON") from exc
    if not isinstance(payload, dict):
        raise _fail(path, "artifact root must be a JSON object")
    return payload


def _row_from_payload(path: Path, payload: dict[str, object]) -> ComparisonRow:
    artifact_kind = payload.get("kind")
    if artifact_kind == "concurrent":
        raise _fail(path, "concurrent artifacts are not supported by batch comparison")
    if artifact_kind == "streaming":
        raise _fail(path, "streaming artifacts are not supported by batch comparison")
    schema = _schema_version(payload, path)
    dataset = _mapping(payload, "dataset", path)
    adapter = _mapping(payload, "adapter", path)
    channel = _mapping(payload, "channel", path)
    summary = _mapping(payload, "summary", path)

    packet_loss = _number(
        channel,
        "packet_loss_rate",
        path,
        minimum=0.0,
        maximum=1.0,
    )
    assert packet_loss is not None

    gain_db = None
    clip_threshold = None
    if schema >= 6:
        gain_db = _number(channel, "gain_db", path)
        assert gain_db is not None
        clip_threshold = _number(channel, "clip_threshold", path, optional=True)
        if clip_threshold is not None and clip_threshold <= 0.0:
            raise _fail(path, "clip_threshold must be a positive number")

    snr_db = None
    if schema >= 2:
        snr_db = _optional_channel_number(channel, "additive_noise_snr_db", path)

    jitter_std_ms = None
    playout_buffer_ms = None
    if schema >= 3:
        if "jitter_std_ms" not in channel or "playout_buffer_ms" not in channel:
            raise _fail(path, "jitter metadata requires both jitter fields")
        jitter_std_ms = _optional_channel_number(
            channel,
            "jitter_std_ms",
            path,
            minimum=0.0,
        )
        playout_buffer_ms = _optional_channel_number(
            channel,
            "playout_buffer_ms",
            path,
            minimum=0.0,
        )
        if (jitter_std_ms is None) != (playout_buffer_ms is None):
            raise _fail(path, "jitter metadata must be both set or both null")

    wer = _number(summary, "wer", path, minimum=0.0)
    cer = _number(summary, "cer", path, minimum=0.0)
    rtf = _number(summary, "rtf", path, minimum=0.0)
    speed_factor = _number(summary, "speed_factor", path, minimum=0.0, optional=True)
    assert wer is not None and cer is not None and rtf is not None

    numeric_entity_accuracy = None
    if schema >= 4:
        numeric_entity_accuracy = _number(
            summary,
            "numeric_entity_accuracy",
            path,
            minimum=0.0,
            maximum=1.0,
            optional=True,
        )

    return ComparisonRow(
        artifact=path.name,
        schema_version=schema,
        adapter=_string(adapter, "name", path),
        model=_string(adapter, "model", path),
        codec=_codec(channel, path),
        packet_loss_rate=packet_loss,
        gain_db=gain_db,
        clip_threshold=clip_threshold,
        snr_db=snr_db,
        jitter_std_ms=jitter_std_ms,
        playout_buffer_ms=playout_buffer_ms,
        wer=wer,
        cer=cer,
        rtf=rtf,
        speed_factor=speed_factor,
        numeric_entity_accuracy=numeric_entity_accuracy,
        item_count=_item_count(dataset, path),
        dataset_fingerprint=_dataset_fingerprint(dataset, schema, path),
    )


def _validate_dataset_identity(rows: tuple[ComparisonRow, ...]) -> None:
    known = [row for row in rows if row.dataset_fingerprint is not None]
    if len(known) < 2:
        return
    first = known[0]
    for row in known[1:]:
        if row.dataset_fingerprint != first.dataset_fingerprint:
            raise ComparisonError(
                f"dataset fingerprint mismatch between {first.artifact} and {row.artifact}"
            )


def load_comparison_rows(paths: Iterable[str | Path]) -> tuple[ComparisonRow, ...]:
    """Load and validate saved benchmark artifacts in caller-supplied order."""

    rows: list[ComparisonRow] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        rows.append(_row_from_payload(path, _load_payload(path)))
    result = tuple(rows)
    _validate_dataset_identity(result)
    return result


def _number_text(value: float | None) -> str:
    if value is None:
        return "—"
    return format(value, ".6g")


def _cell(value: object) -> str:
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "<br>")
    return text.replace("|", r"\|")


def render_comparison_markdown(rows: Iterable[ComparisonRow]) -> str:
    """Render comparison rows as deterministic Markdown."""

    lines = [
        "| Artifact | Schema | Adapter | Model | Codec | Loss | Gain dB | Clip | SNR dB | "
        "Jitter ms | WER | CER | RTF | Speed | Numeric entity | Items |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        jitter = (
            "—"
            if row.jitter_std_ms is None
            else f"{_number_text(row.jitter_std_ms)}/{_number_text(row.playout_buffer_ms)}"
        )
        cells = [
            _cell(row.artifact),
            str(row.schema_version),
            _cell(row.adapter),
            _cell(row.model),
            _cell(row.codec),
            _number_text(row.packet_loss_rate),
            _number_text(row.gain_db),
            _number_text(row.clip_threshold),
            _number_text(row.snr_db),
            jitter,
            _number_text(row.wer),
            _number_text(row.cer),
            _number_text(row.rtf),
            _number_text(row.speed_factor),
            _number_text(row.numeric_entity_accuracy),
            str(row.item_count),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compare_result_artifacts(paths: Iterable[str | Path]) -> str:
    """Load saved artifacts and render a deterministic Markdown comparison."""

    return render_comparison_markdown(load_comparison_rows(paths))
