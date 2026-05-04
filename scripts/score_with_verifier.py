from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from aibox_project_a.eval.matcher import answers_match
from aibox_project_a.eval.metrics import compute_report
from aibox_project_a.eval.parser import parse_final_answer
from aibox_project_a.utils.io import read_jsonl, write_json, write_jsonl
from aibox_project_a.verifier.data import format_verifier_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select best-of-N candidates with a trained verifier.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--verifier-dir", required=True)
    parser.add_argument("--output", default="outputs/verifier/verifier_selected_predictions.jsonl")
    parser.add_argument("--report", default="outputs/verifier/verifier_selected_report.json")
    parser.add_argument("--scores-output", default=None)
    parser.add_argument("--report-subset-ids-from", default=None, help="JSONL file whose ids define the report subset.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--numeric-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def score_texts(texts: list[str], verifier_dir: str, batch_size: int, max_length: int) -> list[float]:
    tokenizer = AutoTokenizer.from_pretrained(verifier_dir)
    model = AutoModelForSequenceClassification.from_pretrained(verifier_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Scoring candidates", unit="batch"):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
            scores.extend(float(prob) for prob in probs)
    return scores


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.candidates)
    report_ids = None
    if args.report_subset_ids_from:
        report_ids = {str(row.get("id")) for row in read_jsonl(args.report_subset_ids_from)}

    for row in tqdm(rows, desc="Parsing candidate answers", unit="candidate"):
        if not row.get("parsed_answer"):
            row["parsed_answer"] = parse_final_answer(row.get("raw_response", ""))

    texts = [format_verifier_input(row) for row in rows]
    scores = score_texts(texts, args.verifier_dir, args.batch_size, args.max_length)
    for row, score in zip(rows, scores):
        row["verifier_score"] = score

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in tqdm(rows, desc="Grouping candidates", unit="candidate"):
        grouped[str(row.get("id"))].append(row)

    selected: list[dict] = []
    for candidates in tqdm(grouped.values(), desc="Selecting best candidates", unit="question"):
        best = max(candidates, key=lambda item: item.get("verifier_score", 0.0))
        correct, pred_norm, gt_norm = answers_match(
            best.get("parsed_answer"),
            best.get("ground_truth"),
            choices=best.get("choices") or [],
            numeric_tolerance=args.numeric_tolerance,
        )
        best["normalized_prediction"] = pred_norm
        best["normalized_ground_truth"] = gt_norm
        best["is_correct"] = correct
        selected.append(best)

    report_rows = [row for row in selected if report_ids is None or str(row.get("id")) in report_ids]
    report = compute_report(report_rows)
    if report_ids is not None:
        report["report_subset_ids_from"] = args.report_subset_ids_from
        report["selected_total_before_subset"] = len(selected)
    write_jsonl(args.output, selected)
    write_json(args.report, report)
    if args.scores_output:
        write_jsonl(args.scores_output, rows)
    print(report)


if __name__ == "__main__":
    main()
