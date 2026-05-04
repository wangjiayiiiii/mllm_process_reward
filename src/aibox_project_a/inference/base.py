from __future__ import annotations

from abc import ABC, abstractmethod

from aibox_project_a.data.schema import EvalSample


class InferenceRunner(ABC):
    @abstractmethod
    def generate(self, samples: list[EvalSample]) -> list[str]:
        raise NotImplementedError
