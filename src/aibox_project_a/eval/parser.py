from __future__ import annotations

import re


FINAL_PATTERNS = [
    re.compile(r"final\s+answer\s*[:：]\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"the\s+final\s+answer\s+is\s*[:：]?\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"answer\s*[:：]\s*(.+)$", re.IGNORECASE | re.MULTILINE),
]


def _extract_boxed(text: str) -> list[str]:
    results: list[str] = []
    start = 0
    marker = r"\boxed{"
    while True:
        idx = text.find(marker, start)
        if idx == -1:
            break
        pos = idx + len(marker)
        depth = 1
        chars: list[str] = []
        while pos < len(text) and depth > 0:
            ch = text[pos]
            if ch == "{":
                depth += 1
                chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth > 0:
                    chars.append(ch)
            else:
                chars.append(ch)
            pos += 1
        if chars:
            results.append("".join(chars).strip())
        start = pos
    return results


def _clean_candidate(candidate: str) -> str:
    candidate = candidate.strip()
    candidate = candidate.splitlines()[0].strip()
    candidate = re.sub(r"^[\"'`]+|[\"'`]+$", "", candidate)
    candidate = re.sub(r"[。；;,.]\s*$", "", candidate)
    return candidate.strip()


def parse_final_answer(text: str) -> str | None:
    if not text:
        return None

    boxed = _extract_boxed(text)
    if boxed:
        return _clean_candidate(boxed[-1])

    matches: list[str] = []
    for pattern in FINAL_PATTERNS:
        matches.extend(match.group(1) for match in pattern.finditer(text))
    if matches:
        return _clean_candidate(matches[-1])

    # Fallback for multiple-choice answers: prefer the last standalone option.
    option_matches = re.findall(r"(?:^|[^A-Za-z])([A-D])(?:[\).。]|$)", text, flags=re.MULTILINE)
    if option_matches:
        return option_matches[-1].upper()

    return _clean_candidate(text[-200:])
