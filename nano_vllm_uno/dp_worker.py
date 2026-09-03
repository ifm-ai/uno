from __future__ import annotations

import sys
import traceback
from multiprocessing.connection import Connection

from nano_vllm_uno.engine.llm_engine import LLMEngine
from nano_vllm_uno.engine.sequence import DECODE_STAT_KEYS


def aggregate_output_stats(outputs: list[dict[str, object]]) -> dict[str, int]:
    return {
        key: sum(int(output["stats"][key]) for output in outputs)
        for key in DECODE_STAT_KEYS
    }


def main(fd: int) -> None:
    conn = Connection(fd)
    engine = None
    try:
        model, tokenizer_path, kwargs = conn.recv()
        engine = LLMEngine(model, tokenizer_path, **kwargs)
        conn.send((True, None))
        while True:
            command = conn.recv()
            if command[0] == "exit":
                break
            _, prompts, sampling_params, generate_kwargs, report_progress = command
            try:
                progress_callback = (
                    lambda completed_stats: conn.send(
                        ("progress", completed_stats)
                    )
                    if report_progress
                    else None
                )
                outputs = engine.generate(
                    prompts,
                    sampling_params,
                    progress_callback=progress_callback,
                    **generate_kwargs,
                )
                conn.send((True, (outputs, aggregate_output_stats(outputs))))
            except BaseException:
                conn.send((False, traceback.format_exc()))
    except BaseException:
        try:
            conn.send((False, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if engine is not None:
            engine.exit()
        conn.close()


if __name__ == "__main__":
    main(int(sys.argv[1]))
