from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from aibox_project_a.data.loaders import row_to_sample
from aibox_project_a.data.schema import Prediction
from aibox_project_a.inference.factory import build_runner
from aibox_project_a.utils.io import load_yaml, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLLM inference for Project A.")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = load_yaml(args.model_config)
    eval_config = load_yaml(args.eval_config)

    input_path = args.input or eval_config["dataset"]["input_path"]
    output_path = args.output or eval_config["outputs"]["predictions_path"]
    rows = read_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    samples = [
        row_to_sample(
            row,
            dataset=row.get("dataset") or eval_config["dataset"].get("name", "dataset"),
            split=row.get("split", ""),
            image_root=eval_config["dataset"].get("image_root"),
            row_index=i,
        )
        for i, row in enumerate(rows)
    ]

    runner = build_runner(model_config)
    predictions: list[dict] = []
    total_batches = (len(samples) + args.batch_size - 1) // args.batch_size
    progress = tqdm(
        range(0, len(samples), args.batch_size),
        total=total_batches,
        desc="Running inference",
        unit="batch",
    )
    for start in progress:
        batch = samples[start : start + args.batch_size]
        outputs = runner.generate(batch)
        for sample, raw_response in zip(batch, outputs):
            predictions.append(
                Prediction(
                    id=sample.id,
                    dataset=sample.dataset,
                    question=sample.question,
                    ground_truth=sample.answer,
                    raw_response=raw_response,
                    image_path=sample.image_path,
                    choices=sample.choices,
                    metadata={
                        "split": sample.split,
                        "source_metadata": sample.metadata,
                        "model": model_config.get("model", {}).get("name"),
                        "backend": model_config.get("model", {}).get("backend"),
                        "generation": model_config.get("generation", {}),
                    },
                ).to_dict()
            )
        progress.set_postfix(samples=min(start + args.batch_size, len(samples)))

    write_jsonl(output_path, predictions)
    print(f"Saved predictions -> {output_path}")


if __name__ == "__main__":
    main()
