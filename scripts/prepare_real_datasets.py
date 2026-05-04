from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from aibox_project_a.data.schema import EvalSample
from aibox_project_a.utils.io import resolve_path, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real MathVision-testmini or V*Bench data.")
    parser.add_argument(
        "--dataset",
        choices=["mathvision_testmini", "mathvision_test", "vstar_bench"],
        required=True,
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _clean_question(text: str) -> str:
    text = text.replace("<image1>", "").replace("<image>", "")
    return re.sub(r"\s+", " ", text).strip()


def _save_pil_image(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGB")
    image.save(path)


def prepare_mathvision(split: str, output: str, image_root: str, limit: int | None) -> None:
    ds = load_dataset("MathLLMs/MathVision", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    root = resolve_path(image_root)
    samples: list[dict[str, Any]] = []
    for idx, row in enumerate(tqdm(ds, desc=f"Preparing MathVision {split}", unit="sample")):
        image_rel = str(row.get("image") or f"images/{row['id']}.jpg")
        image_path = root / image_rel
        decoded_image = row.get("decoded_image")
        if decoded_image is not None:
            _save_pil_image(decoded_image, image_path)

        sample = EvalSample(
            id=str(row.get("id", idx)),
            question=_clean_question(str(row["question"])),
            answer=str(row["answer"]),
            image_path=str(image_path),
            choices=[str(choice) for choice in (row.get("options") or [])],
            dataset="mathvision",
            split=split,
            metadata={
                "level": row.get("level"),
                "subject": row.get("subject"),
                "solution": row.get("solution"),
                "source_image": row.get("image"),
            },
        )
        samples.append(sample.to_dict())

    write_jsonl(output, samples)
    print(f"Prepared MathVision {split}: {len(samples)} samples -> {output}")
    print(f"Images saved under: {root}")


def _parse_vstar_text(text: str) -> tuple[str, list[str]]:
    question_lines: list[str] = []
    options: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        option_match = re.match(r"^\(([A-D])\)\s*(.+)$", line, flags=re.IGNORECASE)
        if option_match:
            options[option_match.group(1).upper()] = option_match.group(2).strip()
            continue
        if line.lower().startswith("answer with"):
            continue
        question_lines.append(line)
    choices = [options[key] for key in sorted(options)]
    return " ".join(question_lines).strip(), choices


def prepare_vstar(output: str, image_root: str, limit: int | None) -> None:
    ds = load_dataset("craigwu/vstar_bench", split="test")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    root = resolve_path(image_root)
    samples: list[dict[str, Any]] = []
    for idx, row in enumerate(tqdm(ds, desc="Preparing V*Bench", unit="sample")):
        image_rel = str(row["image"])
        local_cached = Path(
            hf_hub_download(
                repo_id="craigwu/vstar_bench",
                repo_type="dataset",
                filename=image_rel,
            )
        )
        image_path = root / image_rel
        image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_cached, image_path)

        question, choices = _parse_vstar_text(str(row["text"]))
        sample = EvalSample(
            id=f"vstar_{row.get('question_id', idx)}",
            question=question,
            answer=str(row["label"]).strip().upper(),
            image_path=str(image_path),
            choices=choices,
            dataset="vstar_bench",
            split="test",
            metadata={
                "category": row.get("category"),
                "question_id": row.get("question_id"),
                "source_image": image_rel,
                "original_text": row.get("text"),
            },
        )
        samples.append(sample.to_dict())

    write_jsonl(output, samples)
    print(f"Prepared V*Bench: {len(samples)} samples -> {output}")
    print(f"Images saved under: {root}")


def main() -> None:
    args = parse_args()
    if args.dataset == "mathvision_testmini":
        prepare_mathvision(
            split="testmini",
            output=args.output or "data/processed/mathvision_testmini.jsonl",
            image_root=args.image_root or "data/raw/mathvision",
            limit=args.limit,
        )
    elif args.dataset == "mathvision_test":
        prepare_mathvision(
            split="test",
            output=args.output or "data/processed/mathvision_test.jsonl",
            image_root=args.image_root or "data/raw/mathvision",
            limit=args.limit,
        )
    elif args.dataset == "vstar_bench":
        prepare_vstar(
            output=args.output or "data/processed/vstar_bench.jsonl",
            image_root=args.image_root or "data/raw/vstar_bench",
            limit=args.limit,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()
