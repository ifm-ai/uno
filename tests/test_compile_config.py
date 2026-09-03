from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from nano_vllm_uno.config import Config


class CompileConfigTest(unittest.TestCase):
    def make_config(self, **kwargs) -> Config:
        hf_config = SimpleNamespace(
            dtype=torch.bfloat16,
            max_position_embeddings=8192,
        )
        with (
            patch("nano_vllm_uno.config.os.path.isdir", return_value=True),
            patch("nano_vllm_uno.config.os.listdir", return_value=[]),
            patch(
                "nano_vllm_uno.config.AutoConfig.from_pretrained",
                return_value=hf_config,
            ),
        ):
            return Config("/model", **kwargs)

    def test_compile_accepts_cuda_graph(self):
        config = self.make_config(torch_compile=True)
        self.assertTrue(config.torch_compile)

    def test_compile_rejects_eager_mode(self):
        with self.assertRaisesRegex(ValueError, "requires CUDA graphs"):
            self.make_config(torch_compile=True, enforce_eager=True)

    def test_compile_accepts_tensor_parallel(self):
        config = self.make_config(
            torch_compile=True,
            tensor_parallel_size=2,
        )
        self.assertEqual(config.tensor_parallel_size, 2)

    def test_tree_rejects_compile(self):
        with self.assertRaisesRegex(ValueError, "only by linear verification"):
            self.make_config(torch_compile=True, tree_verify_size=16)


if __name__ == "__main__":
    unittest.main()
