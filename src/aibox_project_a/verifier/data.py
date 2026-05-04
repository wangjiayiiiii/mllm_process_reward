from __future__ import annotations

from typing import Any


def format_verifier_input(row: dict[str, Any]) -> str:
    choices = row.get("choices") or []
    choices_text = "\n".join(str(choice) for choice in choices)
    parts = [
        "Question:",
        str(row.get("question", "")),
    ]
    if choices_text:
        parts.extend(["Choices:", choices_text])
    parts.extend(
        [
            "Candidate solution:",
            str(row.get("raw_response", "")),
            "Parsed final answer:",
            str(row.get("parsed_answer", "")),
        ]
    )
    return "\n".join(parts)


def prediction_to_verifier_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "text": format_verifier_input(row),
        "label": 1 if row.get("is_correct") is True else 0,
        "metadata": {
            "dataset": row.get("dataset"),
            "ground_truth": row.get("ground_truth"),
            "parsed_answer": row.get("parsed_answer"),
            "normalized_prediction": row.get("normalized_prediction"),
            "normalized_ground_truth": row.get("normalized_ground_truth"),
        },
    }
