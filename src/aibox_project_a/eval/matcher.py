from __future__ import annotations

from aibox_project_a.eval.normalizer import normalize_for_match, option_label_from_text, parse_number


def answers_match(
    prediction: str | None,
    ground_truth: str | None,
    *,
    choices: list[str] | None = None,
    numeric_tolerance: float = 1e-6,
) -> tuple[bool, str, str]:
    pred_norm = normalize_for_match(prediction)
    gt_norm = normalize_for_match(ground_truth)

    if not pred_norm or not gt_norm:
        return False, pred_norm, gt_norm

    pred_option = option_label_from_text(prediction, choices)
    gt_option = option_label_from_text(ground_truth, choices)
    if pred_option is not None or gt_option is not None:
        return pred_option == gt_option, pred_option or pred_norm, gt_option or gt_norm

    pred_num = parse_number(prediction)
    gt_num = parse_number(ground_truth)
    if pred_num is not None and gt_num is not None:
        return abs(pred_num - gt_num) <= numeric_tolerance, pred_norm, gt_norm

    return pred_norm == gt_norm, pred_norm, gt_norm
