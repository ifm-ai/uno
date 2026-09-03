import base64
import json
import pickle
import runpy
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nano_vllm_uno.config import Config
from nano_vllm_uno.eval.benchmarks import BENCHMARKS, DEFAULT_DATA_ROOT
from nano_vllm_uno.eval.data import (
    prepare_gpqa_diamond,
    prepare_humaneval,
    prepare_ifeval,
    prepare_lcbv6,
)
from nano_vllm_uno.utils.hub import ADAPTER_ALLOW_PATTERNS, resolve_hf_snapshot


class HuggingFaceResolutionTest(unittest.TestCase):
    def test_local_directory_is_returned_without_hub_access(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "nano_vllm_uno.utils.hub.snapshot_download"
            ) as download:
                resolved = resolve_hf_snapshot(directory)
            self.assertEqual(resolved, str(Path(directory).resolve()))
            download.assert_not_called()

    def test_remote_repository_uses_revision_and_adapter_filter(self):
        with patch(
            "nano_vllm_uno.utils.hub.snapshot_download",
            return_value="/cache/snapshot",
        ) as download:
            resolved = resolve_hf_snapshot(
                "org/adapter",
                revision="revision-sha",
                cache_dir="/cache",
                local_files_only=True,
                allow_patterns=ADAPTER_ALLOW_PATTERNS,
            )
        self.assertEqual(resolved, "/cache/snapshot")
        download.assert_called_once_with(
            repo_id="org/adapter",
            revision="revision-sha",
            cache_dir="/cache",
            local_files_only=True,
            allow_patterns=list(ADAPTER_ALLOW_PATTERNS),
        )

    def test_engine_config_preserves_revision_pinned_hub_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            adapter = root / "adapter"
            model.mkdir()
            adapter.mkdir()
            hf_config = SimpleNamespace(
                dtype=torch.bfloat16,
                max_position_embeddings=32768,
            )
            with (
                patch(
                    "nano_vllm_uno.config.resolve_hf_snapshot",
                    side_effect=[str(model), str(adapter)],
                ) as resolve,
                patch(
                    "nano_vllm_uno.config.AutoConfig.from_pretrained",
                    return_value=hf_config,
                ) as load_config,
            ):
                config = Config(
                    "org/base",
                    model_revision="base-sha",
                    gated_lora_path="org/adapter",
                    gated_lora_revision="adapter-sha",
                    hf_cache_dir=str(root / "cache"),
                    hf_local_files_only=True,
                )

        self.assertEqual(config.model_source, "org/base")
        self.assertEqual(config.gated_lora_source, "org/adapter")
        self.assertEqual(resolve.call_count, 2)
        self.assertEqual(resolve.call_args_list[0].kwargs["revision"], "base-sha")
        self.assertEqual(
            resolve.call_args_list[1].kwargs["revision"],
            "adapter-sha",
        )
        load_config.assert_called_once_with(
            "org/base",
            trust_remote_code=True,
            revision="base-sha",
            cache_dir=str(root / "cache"),
            local_files_only=True,
        )


class PublicDatasetBuilderTest(unittest.TestCase):
    def test_prepare_cli_accepts_omitted_benchmarks_as_all(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "qwen"
            / "prepare_benchmark_data.py"
        )
        parse_args = runpy.run_path(str(script))["parse_args"]

        self.assertEqual(parse_args([]).benchmarks, [])
        self.assertEqual(parse_args(["gpqa-main"]).benchmarks, ["gpqa"])

    def test_all_canonical_paths_use_the_public_data_cache(self):
        self.assertTrue(
            all(
                config.data_path.parent == DEFAULT_DATA_ROOT
                for config in BENCHMARKS.values()
            )
        )

    def test_humaneval_preserves_executable_metadata(self):
        source = {
            "prompt": "def add(a, b):\n    pass\n",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
            "entry_point": "add",
        }
        with patch(
            "nano_vllm_uno.eval.data._load_dataset",
            return_value=[source],
        ):
            record = prepare_humaneval()[0]
        self.assertEqual(record["completion_input"], source["prompt"])
        self.assertEqual(
            record["ground_truth"],
            {"test": source["test"], "entry_point": "add"},
        )
        self.assertIn(source["prompt"], record["chat_input"][0]["content"])

    def test_gpqa_diamond_uses_the_official_gated_configuration(self):
        source = {
            "Question": "Which answer is correct?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong one",
            "Incorrect Answer 2": "Wrong two",
            "Incorrect Answer 3": "Wrong three",
        }
        with patch(
            "nano_vllm_uno.eval.data._load_dataset",
            return_value=[source],
        ) as load:
            records = prepare_gpqa_diamond()

        load.assert_called_once_with(
            "Idavidrein/gpqa",
            "gpqa_diamond",
            split="train",
        )
        self.assertEqual(len(records), 1)
        self.assertIn(records[0]["ground_truth"], "ABCD")

    def test_ifeval_uses_official_rows_and_removes_null_kwargs(self):
        source = {
            "key": 1000,
            "prompt": "Write at least three words.",
            "instruction_id_list": ["length_constraints:number_words"],
            "kwargs": [
                {
                    "relation": "at least",
                    "num_words": 3,
                    "unused": None,
                }
            ],
        }
        with patch(
            "nano_vllm_uno.eval.data._load_dataset",
            return_value=[source],
        ) as load:
            records = prepare_ifeval()

        load.assert_called_once_with("google/IFEval", split="train")
        self.assertEqual(
            records[0]["ground_truth"],
            {
                "instruction_id_list": ["length_constraints:number_words"],
                "kwargs": [{"relation": "at least", "num_words": 3}],
            },
        )
        self.assertEqual(records[0]["completion_input"], source["prompt"])

    def test_lcbv6_decodes_tests_and_marks_functional_method(self):
        private_tests = [
            {"input": "[2]", "output": "2", "testtype": "functional"}
        ]
        encoded_private = base64.b64encode(
            zlib.compress(pickle.dumps(json.dumps(private_tests)))
        ).decode()
        source = {
            "question_content": "Return the first list element.",
            "starter_code": "class Solution:\n    def first(self, values):\n        ",
            "public_test_cases": json.dumps(
                [{"input": "[1]", "output": "1", "testtype": "functional"}]
            ),
            "private_test_cases": encoded_private,
            "metadata": json.dumps({"func_name": "first"}),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test6.jsonl"
            path.write_text(json.dumps(source) + "\n", encoding="utf-8")
            with patch(
                "nano_vllm_uno.eval.data.hf_hub_download",
                return_value=str(path),
            ):
                record = prepare_lcbv6()[0]

        tests = json.loads(record["ground_truth"])
        self.assertEqual(len(tests), 2)
        self.assertTrue(all(test["method_name"] == "first" for test in tests))
        self.assertNotIn("        \n```", record["completion_input"])
        self.assertEqual(record["row"], "0")


if __name__ == "__main__":
    unittest.main()
