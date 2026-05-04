from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from openai import OpenAI

from aibox_project_a.data.schema import EvalSample
from aibox_project_a.inference.base import InferenceRunner
from aibox_project_a.prompts.builder import build_messages


def _as_image_url(path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://", "data:")):
        return path_or_url
    path = Path(path_or_url)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


class VllmOpenAIRunner(InferenceRunner):
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 1024,
        n: int = 1,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.n = n
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def generate(self, samples: list[EvalSample]) -> list[str]:
        outputs: list[str] = []
        for sample in samples:
            messages = build_messages(sample)
            if sample.image_path:
                for message in messages:
                    if message["role"] != "user":
                        continue
                    for item in message["content"]:
                        if item.get("type") == "image_url":
                            item["image_url"]["url"] = _as_image_url(sample.image_path)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
                n=self.n,
            )
            outputs.append(response.choices[0].message.content or "")
        return outputs
