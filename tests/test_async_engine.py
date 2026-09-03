import threading
import unittest

from nano_vllm_uno.engine.async_engine import AsyncLLMEngine
from nano_vllm_uno.sampling_params import SamplingParams


class FakeTokenizer:
    def decode(self, token_ids):
        return ",".join(str(token_id) for token_id in token_ids)


class FakeEngine:
    def __init__(self, *, fail_step=False, block_first_step=False):
        self.tokenizer = FakeTokenizer()
        self.fail_step = fail_step
        self.block_first_step = block_first_step
        self.first_step_started = threading.Event()
        self.release_first_step = threading.Event()
        self.active = []
        self.added_prompts = []
        self.added_max_tokens = []
        self.next_seq_id = 0
        self.exited = False

    def add_request(self, prompt, sampling_params, *, max_tokens=None):
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        self.added_prompts.append(prompt)
        self.added_max_tokens.append(max_tokens)
        self.active.append(seq_id)
        return seq_id

    def is_finished(self):
        return not self.active

    def step(self):
        self.first_step_started.set()
        if self.block_first_step:
            self.block_first_step = False
            if not self.release_first_step.wait(timeout=2):
                raise TimeoutError("test did not release the first engine step")
        if self.fail_step:
            raise ValueError("engine step failed")
        seq_id = self.active.pop(0)
        return [(seq_id, [seq_id + 10], {"forwards": 1})], -1

    def finalize_output(self, token_ids, stats, sampling_params):
        return {
            "text": self.tokenizer.decode(token_ids),
            "token_ids": list(token_ids),
            "stats": dict(stats),
        }

    def exit(self):
        self.exited = True


class AsyncLLMEngineTest(unittest.TestCase):
    def test_admits_requests_arriving_during_generation(self):
        engine = FakeEngine(block_first_step=True)
        async_engine = AsyncLLMEngine(engine, SamplingParams())
        async_engine.start()
        try:
            first = async_engine.submit([1, 2], request_id="first")
            self.assertTrue(engine.first_step_started.wait(timeout=2))
            second = async_engine.submit(
                [3, 4],
                request_id="second",
                max_tokens=17,
            )
            engine.release_first_step.set()

            first_result = first.result(timeout=2)
            second_result = second.result(timeout=2)

            self.assertEqual(first_result.request_id, "first")
            self.assertEqual(first_result.token_ids, [10])
            self.assertEqual(first_result.text, "10")
            self.assertEqual(second_result.request_id, "second")
            self.assertEqual(second_result.token_ids, [11])
            self.assertEqual(engine.added_prompts, [[1, 2], [3, 4]])
            self.assertEqual(engine.added_max_tokens, [None, 17])
        finally:
            engine.release_first_step.set()
            async_engine.shutdown()
        self.assertTrue(engine.exited)

    def test_fatal_engine_error_fails_request_and_health(self):
        engine = FakeEngine(fail_step=True)
        async_engine = AsyncLLMEngine(engine, SamplingParams())
        async_engine.start()
        future = async_engine.submit([1], request_id="failed")

        with self.assertRaisesRegex(ValueError, "engine step failed"):
            future.result(timeout=2)

        self.assertFalse(async_engine.is_running)
        self.assertIsInstance(async_engine.fatal_error, ValueError)
        with self.assertRaisesRegex(RuntimeError, "has failed"):
            async_engine.submit([2])
        async_engine.shutdown()
        self.assertTrue(engine.exited)


if __name__ == "__main__":
    unittest.main()
