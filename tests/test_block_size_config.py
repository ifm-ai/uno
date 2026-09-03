from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from nano_vllm_uno.config import Config
from nano_vllm_uno.engine.llm_engine import LLMEngine
from nano_vllm_uno.sampling_params import SamplingParams


class BlockSizeConfigTest(unittest.TestCase):
    def make_config(self, **kwargs) -> Config:
        hf_config = SimpleNamespace(
            dtype=torch.float16,
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

    def test_default_graphs_cover_every_width_through_maximum(self):
        config = self.make_config(max_diffusion_block_size=4)
        self.assertEqual(config.cuda_graph_block_sizes, [1, 2, 3, 4])

    def test_explicit_graph_widths_are_preserved(self):
        config = self.make_config(
            max_diffusion_block_size=4,
            cuda_graph_block_sizes=[4, 1, 4],
        )
        self.assertEqual(config.cuda_graph_block_sizes, [1, 4])

    def test_graph_width_cannot_exceed_runtime_maximum(self):
        with self.assertRaisesRegex(
            ValueError,
            "cannot exceed max_diffusion_block_size",
        ):
            self.make_config(
                max_diffusion_block_size=4,
                cuda_graph_block_sizes=[1, 8],
            )

    def test_engine_rejects_unknown_arguments(self):
        with self.assertRaisesRegex(
            TypeError,
            "unexpected keyword argument 'max_diffusion_block_sizes'",
        ):
            LLMEngine("/model", max_diffusion_block_sizes=4)

    def test_engine_rejects_removed_lora_arguments(self):
        for name in ("gated_lora", "lora_path", "lora_mode"):
            with self.subTest(name=name), self.assertRaises(TypeError):
                LLMEngine("/model", **{name: 4})

    def test_legacy_diffusion_block_size_alias_is_supported(self):
        with patch(
            "nano_vllm_uno.engine.llm_engine.Config",
            side_effect=RuntimeError("sentinel"),
        ) as config, self.assertRaisesRegex(RuntimeError, "sentinel"):
            LLMEngine("/model", diffusion_block_size=4)
        config.assert_called_once_with(
            "/model",
            max_diffusion_block_size=4,
        )

    def test_generate_rejects_width_above_engine_maximum(self):
        engine = object.__new__(LLMEngine)
        engine.config = SimpleNamespace(max_diffusion_block_size=4)
        params = SamplingParams(diffusion_block_size=5, mask_token_id=2)
        with self.assertRaisesRegex(ValueError, "exceeds engine capacity"):
            engine.generate([], params, use_tqdm=False)

    def test_add_request_rejects_width_above_engine_maximum(self):
        engine = object.__new__(LLMEngine)
        engine.config = SimpleNamespace(max_diffusion_block_size=4)
        params = SamplingParams(diffusion_block_size=5, mask_token_id=2)
        with self.assertRaisesRegex(ValueError, "exceeds engine capacity"):
            engine.add_request([], params)

    def test_uniform_noise_does_not_require_mask_token(self):
        for noise_mode in ("random_uniform", "deterministic_uniform"):
            with self.subTest(noise_mode=noise_mode):
                params = SamplingParams(
                    diffusion_block_size=4,
                    noise_mode=noise_mode,
                )
                self.assertIsNone(params.mask_token_id)

    def test_mask_noise_requires_mask_token(self):
        with self.assertRaisesRegex(ValueError, "Mask noise requires"):
            SamplingParams(diffusion_block_size=4, noise_mode="mask")


if __name__ == "__main__":
    unittest.main()
