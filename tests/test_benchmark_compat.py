import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from nano_vllm_uno.engine.noise import _noise_bounds
from nano_vllm_uno.engine.llm_engine import _trim_terminal_stop_tokens
from nano_vllm_uno.eval.benchmarks import BENCHMARKS
from nano_vllm_uno.eval.code_grader import grade_humaneval_row
from nano_vllm_uno.eval.lcb_grader import score_lcbv6
from nano_vllm_uno.eval.model_tokens import resolve_model_token_ids
from nano_vllm_uno.eval.parsers import (
    parse_code_completion,
    parse_gpqa_answer,
)
from nano_vllm_uno.layers.sampler import build_sparse_top_k_probs
from nano_vllm_uno.sampling_params import SamplingParams
from nano_vllm_uno.eval.generate_benchmark import format_prompt


class BenchmarkCompatibilityTest(unittest.TestCase):
    def test_format_prompt_requests_plain_token_ids(self):
        class FakeTokenizer:
            chat_template = "unused"

            def __init__(self):
                self.tokenize_kwargs = None

            def apply_chat_template(
                self,
                messages,
                *,
                tokenize,
                add_generation_prompt,
                **kwargs,
            ):
                self.assert_generation_prompt = add_generation_prompt
                if not tokenize:
                    return "rendered prompt"
                self.tokenize_kwargs = kwargs
                return (
                    {"input_ids": [11, 12, 13]}
                    if kwargs.get("return_dict", True)
                    else [11, 12, 13]
                )

        tokenizer = FakeTokenizer()
        token_ids, rendered, messages = format_prompt(
            tokenizer,
            [{"role": "user", "content": "test"}],
            "",
        )

        self.assertEqual(token_ids, [11, 12, 13])
        self.assertEqual(rendered, "rendered prompt")
        self.assertEqual(messages, [{"role": "user", "content": "test"}])
        self.assertFalse(tokenizer.tokenize_kwargs["return_dict"])

    def test_canonical_suite_contains_all_twelve_benchmarks(self):
        self.assertEqual(
            tuple(BENCHMARKS),
            (
                "gsm8k",
                "math500",
                "aime24",
                "aime25",
                "aime26",
                "humaneval",
                "mbpp",
                "lcbv6",
                "gpqa",
                "gpqa_diamond",
                "mmlu_pro",
                "ifeval",
                "lcr",
            ),
        )
        self.assertTrue(all(config.expected_rows > 0 for config in BENCHMARKS.values()))
        self.assertTrue(
            all(not hasattr(config, "max_tokens") for config in BENCHMARKS.values())
        )

    def test_canonical_suite_has_a_launcher_for_every_benchmark(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts" / "qwen"
        wrappers = {
            "gsm8k": "run_gsm8k_eval.sh",
            "math500": "run_math500_eval.sh",
            "aime24": "run_aime24_eval.sh",
            "aime25": "run_aime25_eval.sh",
            "aime26": "run_aime26_eval.sh",
            "humaneval": "run_humaneval_eval.sh",
            "mbpp": "run_mbpp_eval.sh",
            "lcbv6": "run_lcbv6_eval.sh",
            "gpqa": "run_gpqa_eval.sh",
            "gpqa_diamond": "run_gpqa_diamond_eval.sh",
            "mmlu_pro": "run_mmlu_pro_eval.sh",
            "ifeval": "run_ifeval_eval.sh",
            "lcr": "run_lcr_eval.sh",
        }
        self.assertEqual(set(wrappers), set(BENCHMARKS))
        self.assertTrue(
            all((scripts / wrapper).is_file() for wrapper in wrappers.values())
        )

    def test_per_benchmark_launcher_forwards_attention_backend(self):
        repository = Path(__file__).resolve().parents[1]
        runner = repository / "scripts" / "qwen" / "run_benchmark.sh"
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            fake_python = temporary / "python"
            fake_python.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${TEST_OUTPUT}"\n',
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            output = temporary / "arguments"
            subprocess.run(
                [str(runner), "aime24"],
                check=True,
                env={
                    **os.environ,
                    "PYTHON": str(fake_python),
                    "TEST_OUTPUT": str(output),
                    "RESULTS_ROOT": str(temporary / "results"),
                    "RUN_NAME": "launcher-test",
                    "ATTENTION_BACKEND": "fa2",
                    "SKIP_GRADING": "1",
                },
            )
            arguments = output.read_text(encoding="utf-8").splitlines()
            backend_index = arguments.index("--attention-backend")
            self.assertEqual(arguments[backend_index + 1], "fa2")

    def test_model_entrypoints_delegate_to_shared_runner(self):
        repository = Path(__file__).resolve().parents[1]
        for family in ("qwen", "k2_horizon"):
            entrypoint = repository / "scripts" / family / "run_benchmark.sh"
            self.assertTrue(entrypoint.is_file())
            self.assertIn(
                'exec "${SCRIPT_DIR}/../common/run_benchmark.sh" "$@"',
                entrypoint.read_text(encoding="utf-8"),
            )

    def test_k2_horizon_linear_wrappers_match_reference_protocol(self):
        repository = Path(__file__).resolve().parents[1]
        scripts = repository / "scripts" / "k2_horizon"
        cases = {
            "gsm8k": (262144, 131072),
            "math500": (262144, 131072),
            "aime24": (262144, 131072),
            "aime25": (500000, 500000),
            "aime26": (500000, 500000),
        }

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            fake_python = temporary / "python"
            fake_python.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${TEST_OUTPUT}"\n',
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            for benchmark, (context, max_tokens) in cases.items():
                output = temporary / f"{benchmark}.arguments"
                wrapper = scripts / f"run_{benchmark}_linear_b8.sh"
                subprocess.run(
                    [str(wrapper)],
                    check=True,
                    env={
                        **os.environ,
                        "PYTHON": str(fake_python),
                        "TEST_OUTPUT": str(output),
                        "RESULTS_ROOT": str(temporary / "results"),
                        "MODEL": "test-k2-base",
                        "GATED_LORA_PATH": "test-k2-adapter",
                        "MODEL_REVISION": "",
                        "GATED_LORA_REVISION": "",
                        "SKIP_GRADING": "1",
                    },
                )
                arguments = output.read_text(encoding="utf-8").splitlines()

                def value(flag: str) -> str:
                    index = arguments.index(flag)
                    return arguments[index + 1]

                self.assertEqual(value("--benchmark"), benchmark)
                self.assertEqual(value("--data-parallel-size"), "8")
                self.assertEqual(value("--tensor-parallel-size"), "1")
                self.assertEqual(value("--max-num-seqs"), "4")
                self.assertEqual(value("--max-model-len"), str(context))
                self.assertEqual(value("--max-num-batched-tokens"), str(context))
                self.assertEqual(value("--max-tokens"), str(max_tokens))
                self.assertEqual(value("--num-samples"), "1")
                self.assertEqual(value("--temperature"), "1.0")
                self.assertEqual(value("--top-k"), "50")
                self.assertEqual(value("--top-p"), "0.95")
                self.assertEqual(value("--attention-backend"), "fa3")
                self.assertEqual(value("--diffusion-block-size"), "8")
                self.assertEqual(value("--cuda-graph-block-sizes"), "1,8")
                self.assertEqual(value("--cuda-graph-batch-sizes"), "1,2,4")
                self.assertEqual(value("--noise-mode"), "random_uniform")
                self.assertEqual(value("--mask-token-id"), "250624")
                self.assertEqual(value("--stop-token-ids"), "250019,1")
                self.assertEqual(value("--instruction"), "")
                self.assertEqual(
                    BENCHMARKS[benchmark].chat_template_kwargs,
                    {"reasoning_effort": "high"},
                )

    def test_copied_slurm_wrapper_uses_exported_repository_root(self):
        repository = Path(__file__).resolve().parents[1]
        source_wrapper = (
            repository
            / "scripts"
            / "qwen"
            / "run_benchmark_slurm.sh"
        )
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            fake_repository = temporary / "checkout"
            runner = (
                fake_repository
                / "scripts"
                / "qwen"
                / "run_benchmark.sh"
            )
            runner.parent.mkdir(parents=True)
            runner.write_text(
                '#!/usr/bin/env bash\nprintf "%s" "$1" > "${TEST_OUTPUT}"\n',
                encoding="utf-8",
            )
            runner.chmod(0o755)

            copied_wrapper = temporary / "slurm_script"
            copied_wrapper.write_bytes(source_wrapper.read_bytes())
            copied_wrapper.chmod(0o755)
            output = temporary / "selected-benchmark"
            subprocess.run(
                [str(copied_wrapper), "aime24"],
                check=True,
                env={
                    **os.environ,
                    "NANO_VLLM_UNO_REPO_ROOT": str(fake_repository),
                    "TEST_OUTPUT": str(output),
                },
            )
            self.assertEqual(output.read_text(encoding="utf-8"), "aime24")

    def test_copied_pull_wrapper_uses_exported_repository_root(self):
        repository = Path(__file__).resolve().parents[1]
        source_wrapper = (
            repository
            / "scripts"
            / "qwen"
            / "run_pull_suite_slurm.sh"
        )
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            fake_repository = temporary / "checkout"
            script_directory = fake_repository / "scripts" / "qwen"
            common_directory = fake_repository / "scripts" / "common"
            script_directory.mkdir(parents=True)
            common_directory.mkdir(parents=True)
            (script_directory / "release_model_defaults.sh").write_text(
                "MODEL=fake-model\nGATED_LORA_PATH=\n"
                "export MODEL GATED_LORA_PATH\n",
                encoding="utf-8",
            )
            (common_directory / "eval_defaults.sh").write_bytes(
                (
                    repository
                    / "scripts"
                    / "common"
                    / "eval_defaults.sh"
                ).read_bytes()
            )

            fake_bin = temporary / "bin"
            fake_bin.mkdir()
            (fake_bin / "scontrol").write_text(
                "#!/usr/bin/env bash\necho fake-node\n",
                encoding="utf-8",
            )
            (fake_bin / "srun").write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n%s\\n' \"$NANO_MAX_NUM_SEQS\" "
                "\"$NANO_CUDA_GRAPH_BATCH_SIZES\" > \"$TEST_OUTPUT\"\n",
                encoding="utf-8",
            )
            (fake_bin / "scontrol").chmod(0o755)
            (fake_bin / "srun").chmod(0o755)

            copied_wrapper = temporary / "slurm_script"
            copied_wrapper.write_bytes(source_wrapper.read_bytes())
            copied_wrapper.chmod(0o755)
            output = temporary / "pull-defaults"
            subprocess.run(
                [str(copied_wrapper)],
                check=True,
                env={
                    **os.environ,
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "NANO_VLLM_UNO_REPO_ROOT": str(fake_repository),
                    "RESULTS_ROOT": str(temporary / "results"),
                    "RUN_NAME": "copied-wrapper-test",
                    "SLURM_JOB_ID": "123",
                    "SLURM_JOB_NODELIST": "fake-node",
                    "SLURM_NNODES": "1",
                    "SLURM_NTASKS": "1",
                    "SKIP_GRADING": "1",
                    "TEST_OUTPUT": str(output),
                },
            )
            self.assertEqual(
                output.read_text(encoding="utf-8").splitlines(),
                ["64", "1,2,4,8,16,32,64"],
            )

    def test_uniform_noise_matches_training_token_range(self):
        params = SamplingParams(
            diffusion_block_size=8,
            mask_token_id=151669,
            noise_mode="random_uniform",
        )
        self.assertEqual(_noise_bounds(params, vocab_size=151936), (1, 151669))

    def test_top_p_uses_renormalized_top_k_mass(self):
        probabilities = torch.tensor([[0.50, 0.20, 0.15, 0.15]])
        logits = probabilities.log()
        eager_fn = getattr(
            build_sparse_top_k_probs,
            "_torchdynamo_orig_callable",
            build_sparse_top_k_probs,
        )
        token_ids, filtered = eager_fn(
            logits,
            temperature=1.0,
            top_k=2,
            top_p=0.6,
        )
        self.assertEqual(token_ids.tolist(), [[0, 1]])
        torch.testing.assert_close(
            filtered,
            torch.tensor([[1.0, 0.0]]),
        )

    def test_model_token_ids_are_resolved_and_range_checked(self):
        config = SimpleNamespace(
            vocab_size=151936,
            mask_token_id=151669,
            eos_token_id=151645,
        )
        tokenizer = SimpleNamespace(
            mask_token_id=None,
            eos_token_id=151643,
        )
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "generation_config.json").write_text(
                json.dumps({"eos_token_id": [151645, 151643]}),
                encoding="utf-8",
            )
            with patch(
                "nano_vllm_uno.eval.model_tokens.AutoConfig.from_pretrained",
                return_value=config,
            ):
                mask_id, stop_ids, vocab_size = resolve_model_token_ids(
                    model_path,
                    tokenizer,
                )
                self.assertEqual(mask_id, 151669)
                self.assertEqual(stop_ids, [151645, 151643])
                self.assertEqual(vocab_size, 151936)
                with self.assertRaisesRegex(ValueError, "outside model vocabulary"):
                    resolve_model_token_ids(
                        model_path,
                        tokenizer,
                        mask_token_id=250624,
                    )

    def test_k2_uniform_noise_uses_vocab_size_as_exclusive_upper_bound(self):
        tokenizer = SimpleNamespace(
            mask_token_id=None,
            eos_token_id=250019,
            unk_token_id=None,
            convert_tokens_to_ids=lambda _: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "k2_horizon",
                        "vocab_size": 250624,
                        "eos_token_id": 1,
                    }
                ),
                encoding="utf-8",
            )

            mask_id, stop_ids, vocab_size = resolve_model_token_ids(
                model_path,
                tokenizer,
                noise_mode="random_uniform",
            )

            self.assertEqual(mask_id, 250624)
            self.assertEqual(stop_ids, [1, 250019])
            self.assertEqual(vocab_size, 250624)
            with self.assertRaisesRegex(ValueError, "outside model vocabulary"):
                resolve_model_token_ids(
                    model_path,
                    tokenizer,
                    mask_token_id=250624,
                    noise_mode="mask",
                )

    def test_gpqa_parser_matches_terminal_pattern(self):
        self.assertEqual(parse_gpqa_answer("analysis\n**(C)**"), "C")

    def test_terminal_stop_tokens_are_not_returned_to_clients(self):
        self.assertEqual(
            _trim_terminal_stop_tokens([10, 151643], {151643, 151645}),
            [10],
        )
        self.assertEqual(
            _trim_terminal_stop_tokens(
                [151643, 10, 151645, 151643],
                {151643, 151645},
            ),
            [151643, 10],
        )

    def test_code_parser_accepts_legacy_terminal_chat_token(self):
        self.assertEqual(
            parse_code_completion("return x\n<|im_end|>"),
            "return x\n",
        )

    def test_humaneval_preserves_reemitted_parameter_names(self):
        row = {
            "completion_input": (
                "def identity(x):\n"
                '    """Return the argument unchanged."""\n'
            ),
            "ground_truth": {
                "entry_point": "identity",
                "test": (
                    "def check(candidate):\n"
                    "    assert candidate(7) == 7\n"
                ),
            },
            "generation": "def identity(value):\n    return value\n",
        }
        graded = grade_humaneval_row(row)
        self.assertEqual(graded["correct"], [True])

    @unittest.skipUnless(
        importlib.util.find_spec("langdetect"),
        "IFEval dependencies are not installed",
    )
    def test_ifeval_grading_is_deterministic(self):
        from nano_vllm_uno.eval.ifeval_grader import score_ifeval

        rows = [
            {
                "row": 0,
                "completion_input": "Use at least two exclamation marks.",
                "ground_truth": {
                    "instruction_id_list": ["keywords:letter_frequency"],
                    "kwargs": [
                        {
                            "let_relation": "at least",
                            "letter": "!",
                            "let_frequency": 2,
                        }
                    ],
                },
                "generation": "A deterministic response!!",
            }
        ]
        first, first_summary = score_ifeval(rows)
        second, second_summary = score_ifeval(rows)
        self.assertEqual(first, second)
        self.assertEqual(first_summary, second_summary)

    @patch("nano_vllm_uno.eval.lcb_grader.codegen_metrics")
    def test_lcb_grades_raw_generation(self, metrics_mock):
        metrics_mock.return_value = (
            {"pass@1": 1.0},
            {0: [[1]]},
        )
        rows = [
            {
                "ground_truth": [
                    {
                        "testtype": "stdin",
                        "input": "1\n",
                        "output": "1\n",
                    }
                ],
                "generation": "```python\nprint(1)\n```",
                "parsed_generations": ["print(0)"],
            }
        ]
        _, summary = score_lcbv6(rows, num_processes=1)
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(
            metrics_mock.call_args.kwargs["generations"],
            [["print(1)"]],
        )


if __name__ == "__main__":
    unittest.main()
