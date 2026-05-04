from __future__ import annotations

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from aibox_project_a.data.schema import EvalSample
from aibox_project_a.inference.base import InferenceRunner
from aibox_project_a.prompts.builder import build_messages


class HuggingFaceQwenRunner(InferenceRunner):
    """Debug runner for small batches. Use vLLM/sglang for official accelerated runs."""

    def __init__(
        self,
        *,
        model: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model,
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model)

    def generate(self, samples: list[EvalSample]) -> list[str]:
        outputs: list[str] = []
        for sample in samples:
            messages = build_messages(sample)
            if sample.image_path and not sample.image_path.startswith(("http://", "https://")):
                # Ensure local image paths are readable before qwen_vl_utils processes them.
                Image.open(sample.image_path).close()
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1.0e-6),
                top_p=self.top_p,
            )
            trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            decoded = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outputs.append(decoded[0])
        return outputs
