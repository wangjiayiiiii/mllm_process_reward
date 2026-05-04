from __future__ import annotations

from aibox_project_a.data.schema import EvalSample


SYSTEM_PROMPT = (
    "You are a careful multimodal reasoning assistant. "
    "Solve the problem using the image and text evidence. "
    "Do not guess details that are not visible or stated. "
    "Reason step by step, verify your calculation, then give the final answer in a strict format."
)


def format_choices(choices: list[str]) -> str:
    if not choices:
        return ""
    lines = ["Choices:"]
    for idx, choice in enumerate(choices):
        label = chr(ord("A") + idx)
        text = str(choice).strip()
        if text.upper() == label:
            lines.append(label)
        elif text[:2].upper() in {f"{label}.", f"{label})"}:
            lines.append(text)
        else:
            lines.append(f"{label}. {text}")
    return "\n".join(lines)


def build_task_instructions(sample: EvalSample) -> list[str]:
    has_choices = bool(sample.choices)
    instructions = [
        "Instructions:",
        "1. Inspect the image carefully before solving. Use visible text, labels, shapes, counts, and spatial relations.",
        "2. Reason step by step, but avoid inventing objects, rows, labels, or measurements that are not clearly visible.",
        "3. For counting problems, count only the relevant visible items and double-check the count.",
        "4. For arithmetic, geometry, chart, or table problems, write the key calculation explicitly and verify it once.",
    ]
    if has_choices:
        instructions.extend(
            [
                "5. Compare your result with every option.",
                "6. The final answer must be exactly one option letter from the choices.",
                "Final Answer: <option letter>",
            ]
        )
    else:
        instructions.extend(
            [
                "5. The final answer must be short and directly comparable to the ground truth.",
                "6. If the answer is a number, output only the number or a simple exact expression.",
                "Final Answer: <answer>",
            ]
        )
    return instructions


def build_user_prompt(sample: EvalSample) -> str:
    chunks = [
        "Question:",
        sample.question.strip(),
    ]
    choices = format_choices(sample.choices)
    if choices:
        chunks.extend(["", choices])
    chunks.extend(["", *build_task_instructions(sample)])
    return "\n".join(chunks)


def build_messages(sample: EvalSample) -> list[dict]:
    content: list[dict] = []
    if sample.image_path:
        content.append({"type": "image_url", "image_url": {"url": sample.image_path}})
    content.append({"type": "text", "text": build_user_prompt(sample)})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
