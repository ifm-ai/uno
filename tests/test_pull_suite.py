import json
import os
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nano_vllm_uno.eval.benchmarks import BenchmarkConfig
from nano_vllm_uno.pull_suite import (
    SuiteSettings,
    _aggregate_rows,
    _cuda_graph_batch_ladder,
    _completed_ids,
    _resolve_worker_model_token_ids,
    _torch_distributed_port,
    build_manifest,
    prepare_suite,
    write_or_validate_manifest,
)


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict=False,
        **kwargs,
    ):
        del add_generation_prompt, return_dict, kwargs
        token_count = int(messages[-1]["content"])
        return list(range(token_count)) if tokenize else f"rendered:{token_count}"


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class PullSuiteTest(unittest.TestCase):
    def test_default_concurrency_and_cuda_graph_batches(self):
        with patch.dict(
            os.environ,
            {
                "NANO_MODEL": "/model",
                "NANO_RESULTS_ROOT": "/results",
                "NANO_RUN_NAME": "defaults",
                "NANO_BENCHMARKS": "aime24",
            },
            clear=True,
        ):
            settings = SuiteSettings.from_env()

        self.assertEqual(settings.max_num_seqs, 64)
        self.assertEqual(
            settings.cuda_graph_batch_sizes,
            (1, 2, 4, 8, 16, 32, 64),
        )

    def test_cuda_graph_batch_ladder_includes_non_power_of_two_limit(self):
        self.assertEqual(_cuda_graph_batch_ladder(48), [1, 2, 4, 8, 16, 32, 48])

    def test_adjacent_jobs_use_disjoint_worker_port_blocks(self):
        first = {_torch_distributed_port(100, rank) for rank in range(8)}
        second = {_torch_distributed_port(101, rank) for rank in range(8)}

        self.assertTrue(first.isdisjoint(second))
        self.assertNotEqual(
            _torch_distributed_port(100, 7),
            _torch_distributed_port(101, 6),
        )

    def test_worker_port_stays_in_allocated_range(self):
        for job_id in (0, 1, 100, 999999999):
            for local_rank in (0, 7, 63):
                port = _torch_distributed_port(job_id, local_rank)
                self.assertGreaterEqual(port, 40000)
                self.assertLess(port, 60000)

        with self.assertRaisesRegex(ValueError, "local_rank"):
            _torch_distributed_port(100, 64)

    def make_settings(self, root: Path) -> SuiteSettings:
        return SuiteSettings(
            model="/model",
            model_revision=None,
            tokenizer_path="/model",
            tokenizer_revision=None,
            gated_lora_path="/adapter",
            gated_lora_revision=None,
            hf_cache_dir=None,
            hf_local_files_only=False,
            results_root=root / "results",
            run_name="smoke",
            benchmarks=("smoke",),
            data_root=None,
            num_samples=1,
            limit=None,
            max_num_seqs=4,
            context_length=10,
            max_num_batched_tokens=10,
            gpu_memory_utilization=0.9,
            attention_backend="fa2",
            global_max_tokens=None,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            diffusion_block_size=4,
            tree_verify_size=None,
            tree_candidate_top_k=16,
            torch_compile=False,
            noise_mode="random_uniform",
            noise_salt=None,
            ignore_eos=False,
            mask_token_id=None,
            stop_token_ids=None,
            cuda_graph_block_sizes=(1, 4),
            cuda_graph_batch_sizes=(1, 2, 4),
            save_token_ids=False,
        )

    def test_prepare_queues_valid_prompt_and_records_context_failure(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "smoke.jsonl"
            write_jsonl(
                data,
                [
                    {"id": "short", "chat_input": [{"role": "user", "content": "2"}], "answer": "a"},
                    {"id": "long", "chat_input": [{"role": "user", "content": "6"}], "answer": "b"},
                ],
            )
            config = BenchmarkConfig(
                name="smoke",
                task="smoke",
                expected_rows=2,
                data_path=data,
            )
            settings = self.make_settings(root)
            with patch.dict("nano_vllm_uno.pull_suite.BENCHMARKS", {"smoke": config}, clear=True):
                suite = prepare_suite(settings, FakeTokenizer())

            self.assertEqual(len(suite.jobs), 1)
            self.assertEqual(suite.jobs[0]["max_tokens"], 4)
            self.assertEqual(len(suite.immediate_results), 1)
            failed = suite.immediate_results[0]["row"]
            self.assertEqual(failed["error_type"], "context_length_exceeded")
            self.assertEqual(failed["resolved_max_tokens"], 0)
            self.assertEqual(failed["prompt_token_count"], 6)

    def test_tree_requires_a_tree_capable_attention_backend(self):
        with TemporaryDirectory() as temporary:
            settings = replace(
                self.make_settings(Path(temporary)),
                tree_verify_size=8,
                attention_backend="fa2",
            )
            with self.assertRaisesRegex(ValueError, "fa3 or fa4"):
                settings.validate()

    def test_worker_resolves_tokens_from_original_hub_source(self):
        with TemporaryDirectory() as temporary:
            settings = replace(
                self.make_settings(Path(temporary)),
                mask_token_id=12,
                stop_token_ids=(13, 14),
            )
            engine = SimpleNamespace(
                config=SimpleNamespace(
                    model="/cache/snapshots/revision-sha",
                    model_source="org/base",
                    model_revision="revision-sha",
                    hf_cache_dir="/cache",
                    hf_local_files_only=True,
                ),
                tokenizer=object(),
            )
            with patch(
                "nano_vllm_uno.pull_suite.resolve_model_token_ids",
                return_value=(12, [13, 14], 20),
            ) as resolve:
                result = _resolve_worker_model_token_ids(engine, settings)

        self.assertEqual(result, (12, [13, 14], 20))
        resolve.assert_called_once_with(
            "org/base",
            engine.tokenizer,
            mask_token_id=12,
            stop_token_ids=[13, 14],
            revision="revision-sha",
            cache_dir="/cache",
            local_files_only=True,
        )

    def test_manifest_allows_exact_resume_and_rejects_changes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "smoke.jsonl"
            write_jsonl(
                data,
                [{"id": 0, "chat_input": [{"role": "user", "content": "2"}]}],
            )
            config = BenchmarkConfig(
                name="smoke",
                task="smoke",
                expected_rows=1,
                data_path=data,
            )
            settings = self.make_settings(root)
            with patch.dict("nano_vllm_uno.pull_suite.BENCHMARKS", {"smoke": config}, clear=True):
                suite = prepare_suite(settings, FakeTokenizer())
                manifest = build_manifest(
                    settings,
                    suite,
                    worker_count=8,
                    mask_token_id=9,
                    stop_token_ids=[1],
                    vocab_size=10,
                )
            path = write_or_validate_manifest(settings.run_dir, manifest)
            self.assertEqual(write_or_validate_manifest(settings.run_dir, manifest), path)
            changed = {**manifest, "num_samples": 2}
            with self.assertRaisesRegex(ValueError, "incompatible resume"):
                write_or_validate_manifest(settings.run_dir, changed)

    def test_empty_generation_file_has_no_completed_ids(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "generations.jsonl"
            path.touch()
            self.assertEqual(_completed_ids(path), set())

    def test_per_sequence_counters_reproduce_legacy_aggregate_tpf(self):
        summary = _aggregate_rows(
            [
                {
                    "output_token_count": 4,
                    "stats": {"accepts": 4, "forwards": 2, "lookaheads": 0},
                },
                {
                    "output_token_count": 9,
                    "stats": {"accepts": 9, "forwards": 3, "lookaheads": 1},
                },
            ]
        )
        expected = 13 / 5
        self.assertEqual(summary["decoder_tokens_per_sequence_forward"], expected)
        self.assertEqual(summary["forward_weighted_mean_sequence_tpf"], expected)
        self.assertEqual(summary["unweighted_mean_sequence_tpf"], 2.5)


if __name__ == "__main__":
    unittest.main()
