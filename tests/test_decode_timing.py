from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from nano_vllm_uno.engine.llm_engine import LLMEngine
from nano_vllm_uno.engine.model_runner import ModelRunner


class DecodeTimingTest(unittest.TestCase):
    def test_engine_aggregates_forwards_by_decode_step(self):
        engine = object.__new__(LLMEngine)
        engine._decode_step_wall_ns = [10_000_000, 8_000_000]
        engine.model_runner = Mock()
        engine.model_runner.call.return_value = {
            "enabled": True,
            "forward_records": [
                {"step_index": 1, "gpu_ms": 3.0},
                {"step_index": 1, "gpu_ms": 4.0},
                {"step_index": 2, "gpu_ms": 5.0},
            ],
        }

        timing = engine.get_decode_timing()

        self.assertEqual(timing["step_wall_ms"], [10.0, 8.0])
        self.assertEqual(timing["model_gpu_ms"], [7.0, 5.0])
        self.assertEqual(timing["exposed_non_model_ms"], [3.0, 3.0])
        self.assertEqual(timing["model_gpu_scope"], "backbone_and_lm_head")

    def test_reset_clears_engine_and_runner_records(self):
        engine = object.__new__(LLMEngine)
        engine._decode_step_wall_ns = [1]
        engine.model_runner = Mock()

        engine.reset_decode_timing()

        self.assertEqual(engine._decode_step_wall_ns, [])
        engine.model_runner.call.assert_called_once_with("reset_decode_timing")

    def test_runner_synchronizes_only_when_results_are_read(self):
        start = Mock()
        end = Mock()
        start.elapsed_time.return_value = 2.5
        runner = object.__new__(ModelRunner)
        runner.config = SimpleNamespace(decode_timing=True)
        runner._decode_forward_events = [
            {
                "start": start,
                "end": end,
                "step_index": 1,
                "batch_size": 4,
                "block_len": 8,
                "cuda_graph_hit": True,
            }
        ]

        with patch("torch.cuda.synchronize") as synchronize:
            timing = runner.get_decode_timing()

        synchronize.assert_called_once_with()
        start.elapsed_time.assert_called_once_with(end)
        self.assertEqual(timing["forward_records"][0]["gpu_ms"], 2.5)


if __name__ == "__main__":
    unittest.main()
