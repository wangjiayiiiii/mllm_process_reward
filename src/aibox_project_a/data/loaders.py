from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aibox_project_a.data.schema import EvalSample
from aibox_project_a.utils.io import read_jsonl, resolve_path


QUESTION_KEYS = ("question", "problem", "query", "prompt", "text")
ANSWER_KEYS = ("answer", "gt", "ground_truth", "label", "final_answer")
IMAGE_KEYS = ("image_path", "image", "image_file", "img", "picture")
CHOICES_KEYS = ("choices", "options", "candidates")
ID_KEYS = ("id", "sample_id", "question_id", "uid", "index")


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _normalize_choices(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, dict):
        return [f"{k}. {v}" for k, v in value.items()]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return _normalize_choices(decoded)
    return [str(value)]


def _normalize_image_path(value: Any, image_root: str | Path | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        path = value.get("path") or value.get("filename")
    else:
        path = value
    if path is None:
        return None
    path_str = str(path)
    if path_str.startswith(("http://", "https://", "data:")):
        return path_str
    path_obj = Path(path_str)
    if path_obj.is_absolute():
        return str(path_obj)
    if image_root:
        return str(resolve_path(image_root) / path_obj)
    return str(path_obj)


def row_to_sample(
    row: dict[str, Any],
    *,
    dataset: str,
    split: str = "",
    image_root: str | Path | None = None,
    row_index: int = 0,
) -> EvalSample:
    sample_id = _first_present(row, ID_KEYS)
    question = _first_present(row, QUESTION_KEYS)
    answer = _first_present(row, ANSWER_KEYS)
    image = _first_present(row, IMAGE_KEYS)
    choices = _first_present(row, CHOICES_KEYS)

    if question is None:
        raise ValueError(f"Missing question field in row {row_index}: keys={list(row.keys())}")
    if answer is None:
        raise ValueError(f"Missing answer field in row {row_index}: keys={list(row.keys())}")

    return EvalSample(
        id=str(sample_id if sample_id is not None else row_index),
        question=str(question),
        answer=str(answer),
        image_path=_normalize_image_path(image, image_root),
        choices=_normalize_choices(choices),
        dataset=dataset,
        split=split,
        metadata={k: v for k, v in row.items() if k not in set(QUESTION_KEYS + ANSWER_KEYS + IMAGE_KEYS + CHOICES_KEYS)},
    )


def load_local_dataset(
    input_path: str | Path,
    *,
    dataset: str,
    split: str = "",
    image_root: str | Path | None = None,
    limit: int | None = None,
) -> list[EvalSample]:
    path = resolve_path(input_path)
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("data", [])
    else:
        raise ValueError(f"Unsupported local dataset format: {path}")

    if limit is not None:
        rows = rows[:limit]
    return [
        row_to_sample(row, dataset=dataset, split=split, image_root=image_root, row_index=i)
        for i, row in enumerate(rows)
    ]


def load_hf_dataset(
    hf_name: str,
    *,
    dataset: str,
    split: str,
    image_root: str | Path | None = None,
    subset: str | None = None,
    limit: int | None = None,
) -> list[EvalSample]:
    from datasets import load_dataset

    ds = load_dataset(hf_name, subset, split=split) if subset else load_dataset(hf_name, split=split)
    rows = [dict(row) for row in ds]
    if limit is not None:
        rows = rows[:limit]
    return [
        row_to_sample(row, dataset=dataset, split=split, image_root=image_root, row_index=i)
        for i, row in enumerate(rows)
    ]
