from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from aibox_project_a.data.schema import Prediction
from aibox_project_a.eval.normalizer import option_label_from_text, parse_number
from aibox_project_a.utils.io import read_jsonl, write_jsonl


UNCERTAIN_PATTERNS = (
    "cannot determine",
    "can't determine",
    "not enough information",
    "unclear",
    "unable to",
    "i cannot",
    "i can't",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank best-of-N candidates with a rule verifier.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scores-output", default=None)
    return parser.parse_args()


def _group_candidates(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("id"))].append(row)
    return grouped


def _has_final_answer(raw_response: str) -> bool:
    return bool(re.search(r"final\s+answer\s*[:：]", raw_response, flags=re.IGNORECASE))


def _reasoning_length_score(raw_response: str) -> float:
    words = re.findall(r"\w+", raw_response)
    if len(words) >= 80:
        return 1.0
    if len(words) >= 35:
        return 0.6
    if len(words) >= 12:
        return 0.2
    return -0.5


def _uncertainty_penalty(raw_response: str) -> float:
    text = raw_response.lower()
    return -1.0 if any(pattern in text for pattern in UNCERTAIN_PATTERNS) else 0.0


def _format_score(row: dict[str, Any]) -> float:
    parsed = row.get("parsed_answer")
    if not parsed:
        return -2.0
    if row.get("choices"):
        return 1.0 if option_label_from_text(parsed, row.get("choices") or []) is not None else -1.0
    return 0.8


def _numeric_consistency_score(row: dict[str, Any]) -> float:
    parsed = row.get("parsed_answer")
    parsed_number = parse_number(parsed)
    if parsed_number is None:
        return 0.0

    raw = str(row.get("raw_response") or "")
    tail = raw[-500:]
    numbers = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?", tail)
    if not numbers:
        return 0.0
    try:
        tail_numbers = [float(num) for num in numbers[-5:]]
    except ValueError:
        return 0.0
    if any(abs(num - parsed_number) <= 1e-6 for num in tail_numbers):
        return 0.6
    return -0.2


def score_candidate(row: dict[str, Any], vote_counts: Counter) -> dict[str, Any]:
    vote_key = row.get("vote_key") or row.get("normalized_prediction") or ""
    support = vote_counts.get(vote_key, 0)
    components = {
        "parseable": 1.0 if row.get("parsed_answer") else -2.0,
        "has_final_answer": 0.5 if _has_final_answer(str(row.get("raw_response") or "")) else -0.2,
        "format": _format_score(row),
        "vote_support": 0.4 * support,
        "reasoning_length": _reasoning_length_score(str(row.get("raw_response") or "")),
        "uncertainty": _uncertainty_penalty(str(row.get("raw_response") or "")),
        "numeric_consistency": _numeric_consistency_score(row),
    }
    return {
        "candidate_id": row.get("candidate_id"),
        "vote_key": vote_key,
        "parsed_answer": row.get("parsed_answer"),
        "score": sum(components.values()),
        "components": components,
    }


def select_candidate(sample_id: str, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    vote_counts = Counter(row.get("vote_key") or row.get("normalized_prediction") or "" for row in candidates)
    scored = [score_candidate(row, vote_counts) for row in candidates]
    score_by_candidate_id = {item["candidate_id"]: item["score"] for item in scored}
    selected = max(
        candidates,
        key=lambda row: (
            score_by_candidate_id.get(row.get("candidate_id"), float("-inf")),
            vote_counts.get(row.get("vote_key") or row.get("normalized_prediction") or "", 0),
            -int(row.get("candidate_id") or 0),
        ),
    )
    selected_score = next(item for item in scored if item["candidate_id"] == selected.get("candidate_id"))

    final = Prediction(
        id=sample_id,
        dataset=selected.get("dataset", ""),
        question=selected.get("question", ""),
        ground_truth=selected.get("ground_truth", ""),
        raw_response=selected.get("raw_response", ""),
        image_path=selected.get("image_path"),
        choices=selected.get("choices") or [],
        parsed_answer=selected.get("parsed_answer"),
        normalized_prediction=selected.get("vote_key") or selected.get("normalized_prediction"),
        metadata={
            **(selected.get("metadata") or {}),
            "rule_verifier": {
                "selected_candidate_id": selected.get("candidate_id"),
                "selected_score": selected_score["score"],
                "selected_components": selected_score["components"],
                "vote_counts": dict(vote_counts),
            },
        },
    ).to_dict()
    return final, scored


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.candidates)
    grouped = _group_candidates(rows)

    predictions: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for sample_id, candidates in tqdm(grouped.items(), desc="Rule verifier rerank", unit="sample"):
        prediction, scored = select_candidate(sample_id, candidates)
        predictions.append(prediction)
        for item in scored:
            score_rows.append({"id": sample_id, **item})

    write_jsonl(args.output, predictions)
    if args.scores_output:
        write_jsonl(args.scores_output, score_rows)
    print(f"Saved rule-verifier predictions -> {args.output}")
    if args.scores_output:
        print(f"Saved rule-verifier scores -> {args.scores_output}")


if __name__ == "__main__":
    main()
