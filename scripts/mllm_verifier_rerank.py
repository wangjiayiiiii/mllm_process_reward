from __future__ import annotations

import argparse
import base64
import mimetypes
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai import OpenAI
from tqdm.auto import tqdm

from aibox_project_a.eval.matcher import answers_match
from aibox_project_a.eval.metrics import compute_report
from aibox_project_a.utils.io import load_yaml, read_jsonl, write_json, write_jsonl


CANDIDATE_LABELS = ("A", "B", "C", "D", "E", "F", "G", "H")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use an MLLM judge to rerank best-of-N candidates.")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--judge-output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-candidate-chars", type=int, default=1200)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--numeric-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def _as_image_url(path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://", "data:")):
        return path_or_url
    path = Path(path_or_url)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _group_candidates(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("id"))].append(row)
    return grouped


def _trim_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def _format_choices(choices: list[str]) -> str:
    if not choices:
        return ""
    lines = ["Original answer choices:"]
    for idx, choice in enumerate(choices):
        label = chr(ord("A") + idx)
        text = str(choice).strip()
        if text[:2].upper() in {f"{label}.", f"{label})"}:
            lines.append(text)
        else:
            lines.append(f"{label}. {text}")
    return "\n".join(lines)


def build_judge_prompt(candidates: list[dict[str, Any]], max_candidate_chars: int) -> str:
    first = candidates[0]
    chunks = [
        "You are a strict multimodal verifier.",
        "Your task is NOT to generate a new answer.",
        "Inspect the image and question, then choose which candidate solution has the most reliable final answer.",
        "",
        "Question:",
        str(first.get("question", "")).strip(),
    ]
    choices_text = _format_choices(first.get("choices") or [])
    if choices_text:
        chunks.extend(["", choices_text])

    chunks.extend(["", "Candidate solutions:"])
    for idx, row in enumerate(candidates):
        label = CANDIDATE_LABELS[idx]
        chunks.extend(
            [
                "",
                f"Candidate {label}:",
                f"Parsed final answer: {row.get('parsed_answer')}",
                "Solution:",
                _trim_text(str(row.get("raw_response") or ""), max_candidate_chars),
            ]
        )

    valid_labels = ", ".join(CANDIDATE_LABELS[: len(candidates)])
    chunks.extend(
        [
            "",
            "Selection rules:",
            "1. Prefer candidates whose final answer is visually and mathematically supported by the image.",
            "2. Penalize candidates that invent visual details, make arithmetic mistakes, or contradict the image.",
            "3. If several candidates are plausible, choose the one with the clearest verified reasoning.",
            "4. Do not output a new answer. Choose one candidate label only.",
            "",
            f"Valid candidate labels: {valid_labels}",
            "Final format:",
            "Selected Candidate: <label>",
        ]
    )
    return "\n".join(chunks)


def _build_messages(candidates: list[dict[str, Any]], max_candidate_chars: int) -> list[dict[str, Any]]:
    image_path = candidates[0].get("image_path")
    content: list[dict[str, Any]] = []
    if image_path:
        content.append({"type": "image_url", "image_url": {"url": _as_image_url(str(image_path))}})
    content.append({"type": "text", "text": build_judge_prompt(candidates, max_candidate_chars)})
    return [
        {"role": "system", "content": "You are a careful verifier for multimodal math reasoning."},
        {"role": "user", "content": content},
    ]


def parse_selected_label(text: str, num_candidates: int) -> str | None:
    if not text:
        return None
    valid = set(CANDIDATE_LABELS[:num_candidates])
    patterns = [
        r"selected\s+candidate\s*[:：]\s*([A-H])",
        r"candidate\s+([A-H])",
        r"\b([A-H])\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            label = match.group(1).upper()
            if label in valid:
                return label
    return None


def _majority_fallback(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    counts = Counter(row.get("vote_key") or row.get("normalized_prediction") or "" for row in candidates)
    if not counts:
        return candidates[0], "first_candidate"
    best_key, _ = counts.most_common(1)[0]
    for row in candidates:
        if (row.get("vote_key") or row.get("normalized_prediction") or "") == best_key:
            return row, "majority_vote"
    return candidates[0], "first_candidate"


def _select_prediction(
    candidates: list[dict[str, Any]],
    judge_response: str,
    numeric_tolerance: float,
) -> dict[str, Any]:
    selected_label = parse_selected_label(judge_response, len(candidates))
    fallback_reason = None
    if selected_label is None:
        selected, fallback_reason = _majority_fallback(candidates)
    else:
        selected = candidates[CANDIDATE_LABELS.index(selected_label)]

    correct, pred_norm, gt_norm = answers_match(
        selected.get("parsed_answer"),
        selected.get("ground_truth"),
        choices=selected.get("choices") or [],
        numeric_tolerance=numeric_tolerance,
    )
    selected = dict(selected)
    selected["normalized_prediction"] = pred_norm
    selected["normalized_ground_truth"] = gt_norm
    selected["is_correct"] = correct
    selected["metadata"] = {
        **(selected.get("metadata") or {}),
        "mllm_verifier": {
            "selected_label": selected_label,
            "selected_candidate_id": selected.get("candidate_id"),
            "fallback_reason": fallback_reason,
            "judge_response": judge_response,
        },
    }
    return selected


def main() -> None:
    args = parse_args()
    config = load_yaml(args.model_config)
    model_cfg = config.get("model", {})
    gen_cfg = config.get("generation", {})
    if model_cfg.get("backend", "mock").lower() not in {"vllm", "openai"}:
        raise ValueError("mllm_verifier_rerank.py requires model.backend to be 'vllm' or 'openai'.")

    client = OpenAI(
        api_key=model_cfg.get("api_key", "EMPTY"),
        base_url=model_cfg.get("api_base", "http://localhost:8000/v1"),
    )
    rows = read_jsonl(args.candidates)
    grouped_items = list(_group_candidates(rows).items())
    if args.limit is not None:
        grouped_items = grouped_items[: args.limit]

    selected_rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    for sample_id, candidates in tqdm(grouped_items, desc="MLLM verifier rerank", unit="sample"):
        response = client.chat.completions.create(
            model=model_cfg.get("name", "Qwen/Qwen2.5-VL-3B-Instruct"),
            messages=_build_messages(candidates, args.max_candidate_chars),
            temperature=0.0,
            top_p=float(gen_cfg.get("top_p", 1.0)),
            max_tokens=args.max_new_tokens,
        )
        judge_response = response.choices[0].message.content or ""
        selected = _select_prediction(candidates, judge_response, args.numeric_tolerance)
        selected_rows.append(selected)
        judge_rows.append(
            {
                "id": sample_id,
                "judge_response": judge_response,
                "selected_label": selected.get("metadata", {}).get("mllm_verifier", {}).get("selected_label"),
                "selected_candidate_id": selected.get("candidate_id"),
                "selected_parsed_answer": selected.get("parsed_answer"),
                "is_correct": selected.get("is_correct"),
            }
        )

    report = compute_report(selected_rows)
    report["candidates_path"] = args.candidates
    report["model"] = model_cfg.get("name")
    write_jsonl(args.output, selected_rows)
    write_json(args.report, report)
    if args.judge_output:
        write_jsonl(args.judge_output, judge_rows)
    print(report)


if __name__ == "__main__":
    main()
