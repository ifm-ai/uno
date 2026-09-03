__version__ = "0.1.0"

__all__ = ["AsyncLLMEngine", "LLM", "SamplingParams", "__version__"]


def __getattr__(name: str):
    if name == "AsyncLLMEngine":
        from nano_vllm_uno.engine.async_engine import AsyncLLMEngine

        return AsyncLLMEngine
    if name == "LLM":
        from nano_vllm_uno.llm import LLM

        return LLM
    if name == "SamplingParams":
        from nano_vllm_uno.sampling_params import SamplingParams

        return SamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
