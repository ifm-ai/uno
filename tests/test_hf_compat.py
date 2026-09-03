import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nano_vllm_uno.utils.hf_compat import (
    is_native_k2_model,
    load_model_config,
    load_tokenizer,
)


class HuggingFaceCompatibilityTest(unittest.TestCase):
    def test_native_k2_config_loads_without_remote_config_code(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "k2_horizon",
                        "vocab_size": 250624,
                        "hidden_size": 4096,
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(is_native_k2_model(str(model_path)))
            config = load_model_config(str(model_path))

            self.assertEqual(config.model_type, "k2_horizon")
            self.assertEqual(config.vocab_size, 250624)
            self.assertEqual(config.hidden_size, 4096)

    @patch("nano_vllm_uno.utils.hf_compat.PreTrainedTokenizerFast.from_pretrained")
    @patch("nano_vllm_uno.utils.hf_compat.AutoTokenizer.from_pretrained")
    def test_tokenizers_backend_falls_back_to_fast_tokenizer(
        self,
        auto_from_pretrained,
        fast_from_pretrained,
    ):
        auto_from_pretrained.side_effect = ValueError(
            "Tokenizer class TokenizersBackend does not exist"
        )
        expected = object()
        fast_from_pretrained.return_value = expected

        tokenizer = load_tokenizer("IFM/K2-Horizon-7B", use_fast=True)

        self.assertIs(tokenizer, expected)
        fast_from_pretrained.assert_called_once_with(
            "IFM/K2-Horizon-7B",
            use_fast=True,
        )


if __name__ == "__main__":
    unittest.main()
