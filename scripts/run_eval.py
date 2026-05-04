from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from aibox_project_a.eval.matcher import answers_match
from aibox_project_a.eval.metrics import compute_report
from aibox_project_a.eval.parser import parse_final_answer
from aibox_project_a.utils.io import load_yaml, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Project A predictions.")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--bad-cases", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    eval_cfg = config.get("evaluation", {})
    outputs_cfg = config.get("outputs", {})

    predictions_path = args.predictions or outputs_cfg["predictions_path"]
    report_path = args.report or outputs_cfg["report_path"]
    bad_cases_path = args.bad_cases or outputs_cfg["bad_cases_path"]
    tolerance = float(eval_cfg.get("numeric_tolerance", 1e-6))

    rows = read_jsonl(predictions_path)
    evaluated: list[dict] = []
    bad_cases: list[dict] = []

    for row in tqdm(rows, desc="Evaluating predictions", unit="sample"):
        parsed = row.get("parsed_answer") or parse_final_answer(row.get("raw_response", ""))
        correct, pred_norm, gt_norm = answers_match(
            parsed,
            row.get("ground_truth"),
            choices=row.get("choices") or [],
            numeric_tolerance=tolerance,
        )
        row["parsed_answer"] = parsed
        row["normalized_prediction"] = pred_norm
        row["normalized_ground_truth"] = gt_norm
        row["is_correct"] = correct
        evaluated.append(row)
        if not correct:
            bad_cases.append(row)

    report = compute_report(evaluated)
    report["predictions_path"] = predictions_path
    report["bad_cases_path"] = bad_cases_path

    write_json(report_path, report)
    write_jsonl(predictions_path, evaluated)
    write_jsonl(bad_cases_path, bad_cases)
    print(report)


if __name__ == "__main__":
    main()
