from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nano_vllm_uno.engine.block_cuda_graph_runner import BlockCudaGraphRunner


class CompileRunnerTest(unittest.TestCase):
    def make_runner(self, *, torch_compile: bool) -> BlockCudaGraphRunner:
        config = SimpleNamespace(torch_compile=torch_compile)
        model = SimpleNamespace()
        return BlockCudaGraphRunner(config, 256, model, object())

    def test_compile_uses_segmented_backbone_without_nested_cuda_graphs(self):
        runner = self.make_runner(torch_compile=True)

        def identity_compile(fn, **kwargs):
            return fn

        with patch("torch.compile", side_effect=identity_compile) as compile_mock:
            runner._prepare_compiled_forward()

        compile_mock.assert_called_once()
        backbone_call = compile_mock.call_args
        self.assertFalse(backbone_call.kwargs["fullgraph"])
        self.assertFalse(backbone_call.kwargs["dynamic"])
        self.assertEqual(
            backbone_call.kwargs["mode"],
            "max-autotune-no-cudagraphs",
        )

    def test_compile_disabled_keeps_eager_callables(self):
        runner = self.make_runner(torch_compile=False)
        with patch("torch.compile") as compile_mock:
            runner._prepare_compiled_forward()
        compile_mock.assert_not_called()
        self.assertIsNone(runner.compiled_forward)


if __name__ == "__main__":
    unittest.main()
