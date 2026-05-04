from __future__ import annotations

from aibox_project_a.inference.base import InferenceRunner
from aibox_project_a.inference.mock_runner import MockRunner


def build_runner(config: dict) -> InferenceRunner:
    model_cfg = config.get("model", {})
    gen_cfg = config.get("generation", {})
    backend = model_cfg.get("backend", "mock").lower()
    model_name = model_cfg.get("name", "Qwen/Qwen2.5-VL-3B-Instruct")

    if backend == "mock":
        return MockRunner()
    if backend in {"vllm", "openai"}:
        from aibox_project_a.inference.vllm_openai_runner import VllmOpenAIRunner

        return VllmOpenAIRunner(
            model=model_name,
            api_base=model_cfg.get("api_base", "http://localhost:8000/v1"),
            api_key=model_cfg.get("api_key", "EMPTY"),
            temperature=float(gen_cfg.get("temperature", 0.0)),
            top_p=float(gen_cfg.get("top_p", 1.0)),
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 1024)),
            n=int(gen_cfg.get("n", 1)),
        )
    if backend in {"hf", "transformers"}:
        from aibox_project_a.inference.hf_qwen_runner import HuggingFaceQwenRunner

        return HuggingFaceQwenRunner(
            model=model_name,
            temperature=float(gen_cfg.get("temperature", 0.0)),
            top_p=float(gen_cfg.get("top_p", 1.0)),
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 1024)),
        )
    raise ValueError(f"Unsupported inference backend: {backend}")
