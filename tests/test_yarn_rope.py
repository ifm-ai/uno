from types import SimpleNamespace
import unittest

import torch
from transformers.modeling_rope_utils import _compute_yarn_parameters

from nano_vllm_uno.layers.rotary_embedding import get_rope


class YaRNRotaryEmbeddingTest(unittest.TestCase):
    def test_matches_transformers_reference(self):
        parameters = {
            "rope_type": "yarn",
            "factor": 16.0,
            "original_max_position_embeddings": 8192,
            "attention_factor": 1.2772588722239782,
            "beta_fast": 128.0,
            "beta_slow": 4.0,
            "truncate": True,
        }
        config = SimpleNamespace(
            rope_theta=1_000_000.0,
            rope_scaling=parameters,
            head_dim=64,
            hidden_size=1536,
            num_attention_heads=32,
            max_position_embeddings=131072,
        )
        expected_inv_freq, expected_attention_factor = _compute_yarn_parameters(
            config,
            torch.device("cpu"),
        )
        rope = get_rope(
            head_size=64,
            rotary_dim=64,
            max_position=131072,
            base=1_000_000.0,
            rope_scaling=parameters,
        )

        positions = torch.tensor([0, 1, 8191, 8192, 16384, 65535, 131071])
        frequencies = torch.outer(positions.float(), expected_inv_freq)
        expected = torch.cat(
            (
                frequencies.cos() * expected_attention_factor,
                frequencies.sin() * expected_attention_factor,
            ),
            dim=-1,
        ).unsqueeze(1)

        torch.testing.assert_close(
            rope.cos_sin_cache[positions],
            expected,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_default_rope_is_unchanged(self):
        rope = get_rope(64, 64, 32, 1_000_000.0, {"rope_type": "default"})
        positions = torch.arange(32, dtype=torch.float)
        inv_freq = 1.0 / (
            1_000_000.0
            ** (torch.arange(0, 64, 2, dtype=torch.float) / 64)
        )
        frequencies = torch.outer(positions, inv_freq)
        expected = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)
        torch.testing.assert_close(
            rope.cos_sin_cache.squeeze(1),
            expected,
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
