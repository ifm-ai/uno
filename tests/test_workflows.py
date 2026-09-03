import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import evaluation.run
import generation
import inference
from evaluation.graders import lcr
from nano_vllm_uno import SamplingParams


class FakeTokenizer:
    chat_template = "test"

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        del messages, kwargs
        return [11, 12] if tokenize else "rendered prompt"


class GenerationTest(unittest.TestCase):
    @patch("generation.LLM")
    def test_shared_generation_reports_tpf_and_tps(self, llm_class):
        llm = llm_class.return_value
        llm.generate.return_value = [
            {"text": "answer", "token_ids": [1, 2, 3], "stats": {}}
        ]
        llm.last_generate_stats = {"accepts": 6, "forwards": 2}
        outputs, metrics = generation.generate(
            [[11, 12]],
            SamplingParams(max_tokens=3),
            model="model",
            use_tqdm=False,
        )
        self.assertEqual(outputs[0]["text"], "answer")
        self.assertEqual(metrics["total_output_tokens"], 3)
        self.assertEqual(metrics["decoder_tokens_per_sequence_forward"], 3.0)
        self.assertGreater(metrics["output_tokens_per_second"], 0)
        llm.exit.assert_called_once()


class InferenceWorkflowTest(unittest.TestCase):
    @patch("inference.generate")
    @patch("inference.resolve_model_token_ids", return_value=(99, [1], 100))
    @patch("inference.load_tokenizer", return_value=FakeTokenizer())
    def test_cli_uses_shared_generation(self, _tokenizer, _tokens, run_generation):
        run_generation.return_value = (
            [{"text": "four", "token_ids": [4], "stats": {}}],
            {
                "total_output_tokens": 1,
                "elapsed_seconds": 1.0,
                "output_tokens_per_second": 1.0,
                "decoder_stats": {"accepts": 1, "forwards": 1},
                "decoder_tokens_per_sequence_forward": 1.0,
            },
        )
        with patch("builtins.print") as output:
            inference.main(
                [
                    "--model",
                    "model",
                    "--gated-lora-path",
                    "adapter",
                    "--prompt",
                    "2 + 2?",
                    "--no-progress",
                ]
            )
        payload = json.loads(output.call_args.args[0])
        self.assertEqual(payload["generations"], ["four"])
        run_generation.assert_called_once()


class EvaluationWorkflowTest(unittest.TestCase):
    @patch("evaluation.run.generate")
    @patch("evaluation.run.resolve_model_token_ids", return_value=(99, [1], 100))
    @patch("evaluation.run.load_tokenizer", return_value=FakeTokenizer())
    def test_cli_writes_generation_artifacts(
        self, _tokenizer, _tokens, run_generation
    ):
        run_generation.return_value = (
            [{"text": "4", "token_ids": [4], "stats": {"accepts": 1, "forwards": 1}}],
            {
                "total_output_tokens": 1,
                "elapsed_seconds": 1.0,
                "output_tokens_per_second": 1.0,
                "decoder_stats": {"accepts": 1, "forwards": 1},
                "decoder_tokens_per_sequence_forward": 1.0,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            output = root / "generations.jsonl"
            summary = root / "generation_summary.json"
            data.write_text(
                json.dumps(
                    {
                        "row": 0,
                        "ground_truth": "4",
                        "chat_input": [{"role": "user", "content": "2 + 2?"}],
                    }
                )
                + "\n"
            )
            evaluation.run.main(
                [
                    "--benchmark",
                    "gsm8k",
                    "--model",
                    "model",
                    "--gated-lora-path",
                    "adapter",
                    "--data",
                    str(data),
                    "--output",
                    str(output),
                    "--summary-output",
                    str(summary),
                    "--limit",
                    "1",
                    "--skip-grading",
                    "--no-progress",
                ]
            )
            self.assertEqual(len(output.read_text().splitlines()), 1)
            self.assertEqual(
                json.loads(summary.read_text())["num_generations"], 1
            )


class LcrGraderTest(unittest.TestCase):
    @patch("evaluation.graders.lcr._judge", return_value=(True, '{"GRADE":"CORRECT"}'))
    def test_lcr_grader_scores_every_generation(self, judge):
        rows = [
            {
                "problem": "question",
                "ground_truth": "reference",
                "generations": ["answer one", "answer two"],
            }
        ]
        graded, summary = lcr.score_lcr(rows)
        self.assertEqual(graded[0]["correct"], [True, True])
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["num_graded"], 2)
        self.assertEqual(judge.call_count, 2)


class ExampleLauncherTest(unittest.TestCase):
    def test_all_model_launchers_select_the_intended_release(self):
        repository = Path(__file__).resolve().parents[1]
        expected = {
            "uno_qwen3_8B": ("s-sahoo/uno-qwen3-8B", "--gated-lora-subfolder"),
            "uno_8B": ("IFM/K2-Horizon-7B", "IFM/K2-Horizon-7B-Uno"),
            "uno_1B": ("IFM/K2-Horizon-0.9B", "IFM/K2-Horizon-0.9B-Uno"),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_python = root / "python"
            fake_python.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$TEST_OUTPUT"\n'
            )
            fake_python.chmod(0o755)
            for model, required in expected.items():
                for workflow, arguments in (
                    ("run_inference.sh", ["--prompt", "test"]),
                    ("run_eval.sh", ["gsm8k"]),
                ):
                    output = root / f"{model}-{workflow}.txt"
                    subprocess.run(
                        [str(repository / "examples" / model / workflow), *arguments],
                        check=True,
                        env={
                            **os.environ,
                            "PYTHON": str(fake_python),
                            "TEST_OUTPUT": str(output),
                            "RESULTS_ROOT": str(root / "results"),
                        },
                    )
                    rendered = output.read_text()
                    self.assertIn(required[0], rendered)
                    self.assertIn(required[1], rendered)


if __name__ == "__main__":
    unittest.main()
