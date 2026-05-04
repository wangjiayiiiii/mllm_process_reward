from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from aibox_project_a.eval.matcher import answers_match
from aibox_project_a.utils.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze best-of-N candidate predictions.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--details-output", default=None)
    parser.add_argument("--numeric-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def _group_candidates(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("id"))].append(row)
    return grouped


def _majority_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row.get("vote_key") or row.get("normalized_prediction") or "" for row in candidates)
    if not counts:
        return candidates[0]
    best_key, _ = counts.most_common(1)[0]
    return next(
        row for row in candidates
        if (row.get("vote_key") or row.get("normalized_prediction") or "") == best_key
    )


def analyze_group(sample_id: str, candidates: list[dict[str, Any]], numeric_tolerance: float) -> dict[str, Any]:
    candidate_results = []
    for row in candidates:
        correct, pred_norm, gt_norm = answers_match(
            row.get("parsed_answer"),
            row.get("ground_truth"),
            choices=row.get("choices") or [],
            numeric_tolerance=numeric_tolerance,
        )
        vote_key = row.get("vote_key") or row.get("normalized_prediction") or pred_norm
        candidate_results.append(
            {
                "candidate_id": row.get("candidate_id"),
                "parsed_answer": row.get("parsed_answer"),
                "vote_key": vote_key,
                "is_correct": correct,
                "normalized_prediction": pred_norm,
                "normalized_ground_truth": gt_norm,
            }
        )

    vote_counts = Counter(item["vote_key"] for item in candidate_results if item["vote_key"])
    majority = _majority_candidate(candidates)
    majority_correct, majority_pred_norm, majority_gt_norm = answers_match(
        majority.get("parsed_answer"),
        majority.get("ground_truth"),
        choices=majority.get("choices") or [],
        numeric_tolerance=numeric_tolerance,
    )
    oracle_correct = any(item["is_correct"] for item in candidate_results)
    unique_answers = len(vote_counts)
    num_candidates = len(candidates)
    consensus_rate = max(vote_counts.values()) / num_candidates if vote_counts and num_candidates else 0.0

    return {
        "id": sample_id,
        "dataset": candidates[0].get("dataset"),
        "question": candidates[0].get("question"),
        "ground_truth": candidates[0].get("ground_truth"),
        "num_candidates": num_candidates,
        "unique_answers": unique_answers,
        "consensus_rate": consensus_rate,
        "vote_counts": dict(vote_counts),
        "oracle_correct": oracle_correct,
        "majority_correct": majority_correct,
        "majority_candidate_id": majority.get("candidate_id"),
        "majority_parsed_answer": majority.get("parsed_answer"),
        "majority_normalized_prediction": majority_pred_norm,
        "majority_normalized_ground_truth": majority_gt_norm,
        "candidate_results": candidate_results,
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.candidates)
    grouped = _group_candidates(rows)
    details = [
        analyze_group(sample_id, candidates, args.numeric_tolerance)
        for sample_id, candidates in tqdm(grouped.items(), desc="Analyzing candidates", unit="sample")
    ]

    total = len(details)
    oracle_correct = sum(1 for item in details if item["oracle_correct"])
    majority_correct = sum(1 for item in details if item["majority_correct"])
    diverse = sum(1 for item in details if item["unique_answers"] > 1)
    oracle_not_majority = sum(
        1 for item in details
        if item["oracle_correct"] and not item["majority_correct"]
    )
    report = {
        "total": total,
        "num_candidates_per_sample": sorted({item["num_candidates"] for item in details}),
        "oracle_correct": oracle_correct,
        "oracle_at_n": oracle_correct / total if total else 0.0,
        "majority_correct": majority_correct,
        "majority_accuracy": majority_correct / total if total else 0.0,
        "oracle_majority_gap": (oracle_correct - majority_correct) / total if total else 0.0,
        "oracle_not_majority": oracle_not_majority,
        "diverse_answer_samples": diverse,
        "diverse_answer_rate": diverse / total if total else 0.0,
        "avg_unique_answers": mean(item["unique_answers"] for item in details) if details else 0.0,
        "avg_consensus_rate": mean(item["consensus_rate"] for item in details) if details else 0.0,
    }

    write_json(args.report, report)
    if args.details_output:
        write_jsonl(args.details_output, details)
    print(report)


if __name__ == "__main__":
    main()
