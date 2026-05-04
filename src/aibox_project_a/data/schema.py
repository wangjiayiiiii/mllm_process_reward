from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvalSample:
    id: str
    question: str
    answer: str
    image_path: str | None = None
    choices: list[str] = field(default_factory=list)
    dataset: str = ""
    split: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Prediction:
    id: str
    dataset: str
    question: str
    ground_truth: str
    raw_response: str
    image_path: str | None = None
    choices: list[str] = field(default_factory=list)
    parsed_answer: str | None = None
    normalized_prediction: str | None = None
    normalized_ground_truth: str | None = None
    is_correct: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
