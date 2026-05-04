from __future__ import annotations

from aibox_project_a.data.schema import EvalSample
from aibox_project_a.inference.base import InferenceRunner


class MockRunner(InferenceRunner):
    """Deterministic runner for pipeline debugging without loading an MLLM."""

    def generate(self, samples: list[EvalSample]) -> list[str]:
        outputs: list[str] = []
        for sample in samples:
            outputs.append(
                "I will solve the problem step by step.\n"
                "This is a mock response used to verify the evaluation pipeline.\n"
                f"Final Answer: {sample.answer}"
            )
        return outputs
