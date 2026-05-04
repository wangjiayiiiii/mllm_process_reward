from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aibox_project_a.data.loaders import load_hf_dataset, load_local_dataset
from aibox_project_a.utils.io import load_yaml, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Project A datasets into unified JSONL.")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", choices=["local", "hf"], default=None)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--hf-name", default=None)
    parser.add_argument("--hf-subset", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    dataset_cfg = config.get("dataset", {})

    source = args.source or dataset_cfg.get("source", "local")
    dataset_name = args.dataset_name or dataset_cfg.get("name", "dataset")
    image_root = args.image_root or dataset_cfg.get("image_root")
    split = args.split or dataset_cfg.get("split", "test")

    if source == "local":
        input_path = args.input_path or dataset_cfg.get("input_path")
        if not input_path:
            raise ValueError("Local source requires --input-path or dataset.input_path in config.")
        samples = load_local_dataset(
            input_path,
            dataset=dataset_name,
            split=split,
            image_root=image_root,
            limit=args.limit,
        )
    else:
        hf_name = args.hf_name or dataset_cfg.get("hf_name")
        if not hf_name:
            raise ValueError("HF source requires --hf-name or dataset.hf_name in config.")
        samples = load_hf_dataset(
            hf_name,
            dataset=dataset_name,
            split=split,
            subset=args.hf_subset or dataset_cfg.get("hf_subset"),
            image_root=image_root,
            limit=args.limit,
        )

    write_jsonl(args.output, [sample.to_dict() for sample in samples])
    print(f"Prepared {len(samples)} samples -> {args.output}")


if __name__ == "__main__":
    main()
