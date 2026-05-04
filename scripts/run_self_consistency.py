from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from aibox_project_a.data.loaders import row_to_sample
from aibox_project_a.data.schema import Prediction
from aibox_project_a.eval.normalizer import normalize_for_match, option_label_from_text
from aibox_project_a.eval.parser import parse_final_answer
from aibox_project_a.inference.factory import build_runner
from aibox_project_a.utils.io import load_yaml, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-consistency best-of-N inference.")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="Final voted predictions JSONL.")
    parser.add_argument("--candidates-output", required=True, help="All candidate responses JSONL.")
    parser.add_argument("--n", type=int, default=4, help="Number of candidates per sample.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser.parse_args()


def _vote_key(parsed_answer: str | None, choices: list[str]) -> str:
    option = option_label_from_text(parsed_answer, choices)
    if option is not None:
        return option
    return normalize_for_match(parsed_answer)


def _build_samples(rows: list[dict[str, Any]], eval_config: dict[str, Any]):
    return [
        row_to_sample(
            row,
            dataset=row.get("dataset") or eval_config["dataset"].get("name", "dataset"),
            split=row.get("split", ""),
            image_root=eval_config["dataset"].get("image_root"),
            row_index=i,
        )
        for i, row in enumerate(rows)
    ]


def _build_runner_config(base_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    generation = config.setdefault("generation", {})
    generation["temperature"] = args.temperature
    generation["top_p"] = args.top_p
    generation["n"] = 1
    if args.max_new_tokens is not None:
        generation["max_new_tokens"] = args.max_new_tokens
    return config


def _select_winner(sample_id: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(candidate["vote_key"] for candidate in candidates if candidate["vote_key"])
    if not counts:
        winner = candidates[0]
        vote_counts: dict[str, int] = {}
    else:
        best_key, _ = counts.most_common(1)[0]
        winner = next(candidate for candidate in candidates if candidate["vote_key"] == best_key)
        vote_counts = dict(counts)

    final = Prediction(
        id=sample_id,
        dataset=winner["dataset"],
        question=winner["question"],
        ground_truth=winner["ground_truth"],
        raw_response=winner["raw_response"],
        image_path=winner.get("image_path"),
        choices=winner.get("choices") or [],
        parsed_answer=winner.get("parsed_answer"),
        normalized_prediction=winner.get("vote_key"),
        metadata={
            **(winner.get("metadata") or {}),
            "self_consistency": {
                "num_candidates": len(candidates),
                "vote_counts": vote_counts,
                "selected_candidate_id": winner["candidate_id"],
            },
        },
    ).to_dict()
    return final


def main() -> None:
    args = parse_args()
    if args.n < 1:
        raise ValueError("--n must be >= 1")

    model_config = _build_runner_config(load_yaml(args.model_config), args)
    eval_config = load_yaml(args.eval_config)

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    samples = _build_samples(rows, eval_config)
    runner = build_runner(model_config)

    candidates: list[dict[str, Any]] = []
    total_batches = ((len(samples) + args.batch_size - 1) // args.batch_size) * args.n
    progress = tqdm(total=total_batches, desc=f"Self-consistency N={args.n}", unit="batch")

    for candidate_id in range(args.n):
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            outputs = runner.generate(batch)
            for sample, raw_response in zip(batch, outputs):
                parsed = parse_final_answer(raw_response)
                vote_key = _vote_key(parsed, sample.choices)
                candidates.append(
                    Prediction(
                        id=sample.id,
                        dataset=sample.dataset,
                        question=sample.question,
                        ground_truth=sample.answer,
                        raw_response=raw_response,
                        image_path=sample.image_path,
                        choices=sample.choices,
                        parsed_answer=parsed,
                        normalized_prediction=vote_key,
                        metadata={
                            "split": sample.split,
                            "source_metadata": sample.metadata,
                            "candidate_id": candidate_id,
                            "model": model_config.get("model", {}).get("name"),
                            "backend": model_config.get("model", {}).get("backend"),
                            "generation": model_config.get("generation", {}),
                        },
                    ).to_dict()
                    | {"candidate_id": candidate_id, "vote_key": vote_key}
                )
            progress.update(1)
            progress.set_postfix(candidate=candidate_id + 1, samples=min(start + args.batch_size, len(samples)))
    progress.close()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["id"])].append(candidate)

    final_predictions = [
        _select_winner(sample_id, group)
        for sample_id, group in tqdm(grouped.items(), desc="Voting candidates", unit="sample")
    ]

    write_jsonl(args.candidates_output, candidates)
    write_jsonl(args.output, final_predictions)
    print(f"Saved candidates -> {args.candidates_output}")
    print(f"Saved voted predictions -> {args.output}")


if __name__ == "__main__":
    main()
