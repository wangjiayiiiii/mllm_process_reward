from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def compute_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("is_correct") is True)
    parsed = sum(1 for row in rows if row.get("parsed_answer"))
    report: dict[str, Any] = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "parsed": parsed,
        "parse_success_rate": parsed / total if total else 0.0,
    }

    by_dataset: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        dataset = row.get("dataset") or "unknown"
        by_dataset[dataset]["total"] += 1
        if row.get("is_correct") is True:
            by_dataset[dataset]["correct"] += 1

    report["by_dataset"] = {
        dataset: {
            "total": counts["total"],
            "correct": counts["correct"],
            "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
        }
        for dataset, counts in by_dataset.items()
    }
    return report
