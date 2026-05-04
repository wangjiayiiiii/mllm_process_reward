from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from aibox_project_a.eval.matcher import answers_match
from aibox_project_a.eval.parser import parse_final_answer
from aibox_project_a.utils.io import read_jsonl, write_json, write_jsonl


COUNTING_KEYWORDS = ("count", "how many", "number of", "missing", "total")
GEOMETRY_KEYWORDS = ("angle", "triangle", "circle", "rectangle", "square", "length", "area", "perimeter", "parallel")
CHART_TABLE_KEYWORDS = ("chart", "graph", "table", "bar", "line plot", "axis", "row", "column")
OCR_KEYWORDS = ("text", "word", "label", "letter", "read", "written", "shown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and sample bad cases for Project A reports.")
    parser.add_argument("--predictions", required=True, help="Evaluated predictions JSONL.")
    parser.add_argument("--candidates", default=None, help="Optional best-of-N candidates JSONL.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--samples-output", required=True)
    parser.add_argument("--max-samples-per-category", type=int, default=5)
    parser.add_argument("--numeric-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def _group_candidates(path: str | None) -> dict[str, list[dict[str, Any]]]:
    if not path:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        grouped[str(row.get("id"))].append(row)
    return grouped


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def infer_problem_type(row: dict[str, Any]) -> str:
    question = str(row.get("question") or "").lower()
    if _contains_any(question, COUNTING_KEYWORDS):
        return "counting"
    if _contains_any(question, GEOMETRY_KEYWORDS):
        return "geometry_or_measurement"
    if _contains_any(question, CHART_TABLE_KEYWORDS):
        return "chart_or_table"
    if _contains_any(question, OCR_KEYWORDS):
        return "ocr_or_text_reading"
    if row.get("choices"):
        return "multiple_choice_visual_reasoning"
    return "open_visual_reasoning"


def candidate_oracle_info(
    candidates: list[dict[str, Any]],
    numeric_tolerance: float,
) -> tuple[bool | None, list[dict[str, Any]]]:
    if not candidates:
        return None, []

    results: list[dict[str, Any]] = []
    oracle_correct = False
    for row in candidates:
        parsed = row.get("parsed_answer") or parse_final_answer(row.get("raw_response", ""))
        correct, pred_norm, gt_norm = answers_match(
            parsed,
            row.get("ground_truth"),
            choices=row.get("choices") or [],
            numeric_tolerance=numeric_tolerance,
        )
        oracle_correct = oracle_correct or correct
        results.append(
            {
                "candidate_id": row.get("candidate_id"),
                "parsed_answer": parsed,
                "normalized_prediction": pred_norm,
                "normalized_ground_truth": gt_norm,
                "is_correct": correct,
            }
        )
    return oracle_correct, results


def infer_error_category(row: dict[str, Any], oracle_correct: bool | None) -> str:
    if not row.get("parsed_answer"):
        return "parse_failure"
    if oracle_correct is True:
        return "reranker_or_voting_failure"
    if oracle_correct is False:
        return "candidate_generation_failure"
    return infer_problem_type(row)


def build_case_summary(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    numeric_tolerance: float,
) -> dict[str, Any]:
    oracle_correct, candidate_results = candidate_oracle_info(candidates, numeric_tolerance)
    category = infer_error_category(row, oracle_correct)
    problem_type = infer_problem_type(row)
    return {
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "category": category,
        "problem_type": problem_type,
        "question": row.get("question"),
        "ground_truth": row.get("ground_truth"),
        "parsed_answer": row.get("parsed_answer"),
        "normalized_prediction": row.get("normalized_prediction"),
        "normalized_ground_truth": row.get("normalized_ground_truth"),
        "image_path": row.get("image_path"),
        "choices": row.get("choices") or [],
        "oracle_correct_among_candidates": oracle_correct,
        "candidate_results": candidate_results,
        "raw_response_tail": str(row.get("raw_response") or "")[-800:],
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions)
    bad_rows = [row for row in rows if row.get("is_correct") is not True]
    candidates_by_id = _group_candidates(args.candidates)

    summaries = [
        build_case_summary(row, candidates_by_id.get(str(row.get("id")), []), args.numeric_tolerance)
        for row in tqdm(bad_rows, desc="Summarizing bad cases", unit="case")
    ]

    category_counts = Counter(row["category"] for row in summaries)
    problem_type_counts = Counter(row["problem_type"] for row in summaries)
    sampled: list[dict[str, Any]] = []
    per_category: Counter = Counter()
    for row in summaries:
        category = row["category"]
        if per_category[category] >= args.max_samples_per_category:
            continue
        sampled.append(row)
        per_category[category] += 1

    report = {
        "predictions_path": args.predictions,
        "candidates_path": args.candidates,
        "total_predictions": len(rows),
        "bad_cases": len(bad_rows),
        "bad_case_rate": len(bad_rows) / len(rows) if rows else 0.0,
        "category_counts": dict(category_counts),
        "problem_type_counts": dict(problem_type_counts),
        "sampled_cases": len(sampled),
    }
    write_json(args.report, report)
    write_jsonl(args.samples_output, sampled)
    print(report)


if __name__ == "__main__":
    main()
