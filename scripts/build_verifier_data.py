from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from aibox_project_a.eval.matcher import answers_match
from aibox_project_a.utils.io import read_jsonl, write_jsonl
from aibox_project_a.verifier.data import prediction_to_verifier_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build answer-level verifier data from predictions or candidates.")
    parser.add_argument("--predictions", default=None, help="Evaluated predictions JSONL.")
    parser.add_argument("--candidates", default=None, help="Best-of-N candidate JSONL.")
    parser.add_argument("--train-output", default="outputs/verifier/verifier_train.jsonl")
    parser.add_argument("--valid-output", default="outputs/verifier/verifier_valid.jsonl")
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--numeric-tolerance", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _label_row(row: dict[str, Any], numeric_tolerance: float) -> dict[str, Any]:
    row = dict(row)
    if row.get("is_correct") is True or row.get("is_correct") is False:
        return row
    correct, pred_norm, gt_norm = answers_match(
        row.get("parsed_answer"),
        row.get("ground_truth"),
        choices=row.get("choices") or [],
        numeric_tolerance=numeric_tolerance,
    )
    row["is_correct"] = correct
    row["normalized_prediction"] = pred_norm
    row["normalized_ground_truth"] = gt_norm
    return row


def _split_by_question_id(
    rows: list[dict[str, Any]],
    valid_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("id"))].append(row)

    ids = list(grouped)
    random.Random(seed).shuffle(ids)
    valid_size = max(1, int(len(ids) * valid_ratio)) if len(ids) > 1 else 0
    valid_ids = set(ids[:valid_size])

    train_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    for sample_id, group in grouped.items():
        if sample_id in valid_ids:
            valid_rows.extend(group)
        else:
            train_rows.extend(group)
    return train_rows, valid_rows


def main() -> None:
    args = parse_args()
    if bool(args.predictions) == bool(args.candidates):
        raise ValueError("Provide exactly one of --predictions or --candidates")

    source_path = args.candidates or args.predictions
    raw_rows = read_jsonl(source_path)
    labeled_rows = [
        _label_row(row, args.numeric_tolerance)
        for row in tqdm(raw_rows, desc="Labeling verifier rows", unit="sample")
    ]
    rows = [
        prediction_to_verifier_row(row)
        for row in tqdm(labeled_rows, desc="Building verifier rows", unit="sample")
    ]
    train_rows, valid_rows = _split_by_question_id(rows, args.valid_ratio, args.seed)
    random.Random(args.seed).shuffle(train_rows)
    random.Random(args.seed + 1).shuffle(valid_rows)

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.valid_output, valid_rows)
    print(
        {
            "train": len(train_rows),
            "valid": len(valid_rows),
            "train_positive": sum(row["label"] for row in train_rows),
            "valid_positive": sum(row["label"] for row in valid_rows),
        }
    )


if __name__ == "__main__":
    main()
