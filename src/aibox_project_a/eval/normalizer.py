from __future__ import annotations

import re
from fractions import Fraction


LATEX_COMMANDS = {
    r"\left": "",
    r"\right": "",
    r"\text": "",
    r"\mathrm": "",
    r"\mbox": "",
}


def normalize_text_answer(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    for old, new in LATEX_COMMANDS.items():
        text = text.replace(old, new)
    text = text.replace("$", "")
    text = text.replace("\\%", "%")
    text = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\n\r。；;,.")
    return text.lower()


def normalize_option(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    match = re.match(r"^\(?\s*([A-Da-d])\s*\)?[\.。:]?$", text)
    if match:
        return match.group(1).upper()
    match = re.match(r"^\(?\s*([A-Da-d])\s*\)?[\.。:]\s+.+$", text)
    if match:
        return match.group(1).upper()
    match = re.match(r"^(?:option|choice)\s+([A-Da-d])\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def option_label_from_text(value: str | None, choices: list[str] | None = None) -> str | None:
    direct = normalize_option(value)
    if direct is not None:
        return direct
    if value is None or not choices:
        return None

    pred = normalize_text_answer(value)
    if not pred:
        return None

    for idx, choice in enumerate(choices):
        label = chr(ord("A") + idx)
        choice_norm = normalize_text_answer(choice)
        if not choice_norm:
            continue
        if pred == choice_norm:
            return label
        if pred in {f"{label.lower()} {choice_norm}", f"{label.lower()}. {choice_norm}", f"({label.lower()}) {choice_norm}"}:
            return label
        if choice_norm in pred and len(choice_norm) >= 3:
            return label
    return None


def parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    text = normalize_text_answer(value)
    text = text.replace(",", "")
    percent = text.endswith("%")
    if percent:
        text = text[:-1].strip()

    frac_match = re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", text)
    try:
        if frac_match:
            number = float(Fraction(text.replace(" ", "")))
        else:
            numeric_match = re.search(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?", text)
            if not numeric_match:
                return None
            number = float(numeric_match.group(0))
    except (ValueError, ZeroDivisionError):
        return None

    return number / 100.0 if percent else number


def normalize_for_match(value: str | None) -> str:
    option = normalize_option(value)
    if option is not None:
        return option
    number = parse_number(value)
    if number is not None:
        return f"{number:.12g}"
    return normalize_text_answer(value)
