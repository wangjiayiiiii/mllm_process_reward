from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from aibox_project_a.utils.io import load_yaml, read_jsonl


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight answer-level verifier.")
    parser.add_argument("--config", default="configs/verifier.yaml")
    return parser.parse_args()


def _load_rows(path: str) -> list[dict]:
    return read_jsonl(path)


def _rows_to_dataset(rows: list[dict]) -> Dataset:
    if not rows:
        raise ValueError("Verifier dataset is empty")
    return Dataset.from_list([{"text": row["text"], "label": int(row["label"])} for row in rows])


def _class_weights(rows: list[dict]) -> torch.Tensor:
    labels = [int(row["label"]) for row in rows]
    counts = np.bincount(labels, minlength=2)
    total = max(1, int(counts.sum()))
    weights = total / (2.0 * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def _training_args(model_cfg: dict, train_cfg: dict) -> TrainingArguments:
    strategy_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    kwargs = {
        "output_dir": model_cfg["output_dir"],
        "learning_rate": float(train_cfg.get("learning_rate", 2e-5)),
        "per_device_train_batch_size": int(train_cfg.get("batch_size", 8)),
        "per_device_eval_batch_size": int(train_cfg.get("batch_size", 8)),
        "num_train_epochs": float(train_cfg.get("epochs", 3)),
        strategy_key: "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "report_to": [],
        "seed": int(train_cfg.get("seed", 42)),
        "logging_steps": int(train_cfg.get("logging_steps", 20)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 2)),
    }
    if bool(train_cfg.get("fp16", False)):
        kwargs["fp16"] = True
    return TrainingArguments(**kwargs)


def _load_dataset(path: str) -> Dataset:
    rows = read_jsonl(path)
    return _rows_to_dataset(rows)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    train_rows = _load_rows(data_cfg["train_path"])
    valid_rows = _load_rows(data_cfg["valid_path"])
    weights = _class_weights(train_rows)
    print(
        {
            "train": len(train_rows),
            "valid": len(valid_rows),
            "train_positive": sum(int(row["label"]) for row in train_rows),
            "valid_positive": sum(int(row["label"]) for row in valid_rows),
            "class_weights": weights.tolist(),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(model_cfg["base_model"], num_labels=2)

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=int(train_cfg.get("max_length", 512)),
        )

    train_ds = _rows_to_dataset(train_rows).map(tokenize, batched=True)
    valid_ds = _rows_to_dataset(valid_rows).map(tokenize, batched=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    training_args = _training_args(model_cfg, train_cfg)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=weights,
    )
    trainer.train()
    trainer.save_model(model_cfg["output_dir"])
    tokenizer.save_pretrained(model_cfg["output_dir"])


if __name__ == "__main__":
    main()
