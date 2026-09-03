import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from training.config import LoraSettings, UnoObjectiveConfig
from training.constants import DEFAULT_LORA_ALPHA, DEFAULT_LORA_RANK
from training.curriculum import BlockCurriculumPlan
from training.create_fixed_curriculum import build_fixed_curriculum
from training.data import (
    UnoDataCollator,
    content_hash_rows,
    encode_qwen3_sharegpt_example,
    infer_sequence_lengths,
)
from training.lora import (
    TokenwiseLoraRouter,
    make_saved_adapter_portable,
    resolve_lora_targets,
)
from training.losses import (
    resolve_ce_targets,
    token_normalized_reverse_kl,
    token_normalized_total_variation,
)
from training.modeling import configure_uniform_noise_config
from training.train import (
    _load_base_model,
    _resolve_model_snapshot,
    build_parser,
)
from training.trainer import combine_objective_losses


ROOT = Path(__file__).resolve().parents[1]


class RecordingTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def __init__(self):
        self.texts = []

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("The Uno template must not add special tokens.")
        self.texts.append(text)
        return [ord(character) % 251 + 1 for character in text]


class ModelResolutionTest(unittest.TestCase):
    @patch("training.train.snapshot_download")
    def test_resolves_the_pinned_model_revision_to_a_local_snapshot(self, download):
        download.return_value = "/cache/resolved-snapshot"
        result = _resolve_model_snapshot(
            Path("/cache"),
            local_files_only=True,
        )
        self.assertEqual(result, Path("/cache/resolved-snapshot"))
        download.assert_called_once_with(
            repo_id="s-sahoo/uno-qwen3-8B",
            revision="b8a7577b3223bdcf2b3af0f2fc6e95258b3bbc29",
            cache_dir="/cache",
            local_files_only=True,
        )

    @patch("training.train.AutoModelForCausalLM.from_pretrained")
    @patch("training.train.AutoConfig.from_pretrained")
    @patch("training.train.AutoTokenizer.from_pretrained")
    def test_all_transformers_loaders_use_the_resolved_local_snapshot(
        self,
        load_tokenizer,
        load_config,
        load_model,
    ):
        tokenizer = SimpleNamespace(
            pad_token_id=0,
            pad_token=None,
            eos_token="<eos>",
            padding_side="left",
        )
        model_config = SimpleNamespace()
        model = object()
        load_tokenizer.return_value = tokenizer
        load_config.return_value = model_config
        load_model.return_value = model

        actual_tokenizer, actual_model = _load_base_model(Path("/cache/snapshot"))

        self.assertIs(actual_tokenizer, tokenizer)
        self.assertIs(actual_model, model)
        self.assertEqual(tokenizer.padding_side, "right")
        self.assertEqual(model_config.noise, "uniform")
        for loader in (load_tokenizer, load_config, load_model):
            self.assertEqual(loader.call_args.args[0], "/cache/snapshot")
            self.assertTrue(loader.call_args.kwargs["local_files_only"])


class PreprocessingTest(unittest.TestCase):
    def test_uno_template_masks_prompt_and_inserts_empty_think(self):
        tokenizer = RecordingTokenizer()
        encoded = encode_qwen3_sharegpt_example(
            [
                {"from": "human", "value": "Question"},
                {"from": "gpt", "value": "Answer"},
            ],
            tokenizer,
        )
        self.assertEqual(len(encoded["input_ids"]) % 3, 0)
        self.assertLessEqual(len(encoded["input_ids"]), 4095)
        self.assertIn("<think>\n\n</think>\n\nAnswer", tokenizer.texts[1])
        source_length = len(tokenizer.texts[0])
        self.assertEqual(encoded["labels"][:source_length], [-100] * source_length)
        self.assertTrue(any(label != -100 for label in encoded["labels"]))

    def test_long_rows_are_rounded_to_4095(self):
        tokenizer = RecordingTokenizer()
        encoded = encode_qwen3_sharegpt_example(
            [
                {"from": "human", "value": "q" * 5000},
                {"from": "gpt", "value": "a" * 5000},
            ],
            tokenizer,
        )
        self.assertEqual(len(encoded["input_ids"]), 4095)

    def test_infer_sequence_lengths_matches_balanced_truncation(self):
        self.assertEqual(infer_sequence_lengths(100, 300, 200), (50, 150))
        self.assertEqual(infer_sequence_lengths(50, 1000, 200), (50, 150))

    def test_collator_adds_only_ignored_batch_padding(self):
        tokenizer = RecordingTokenizer()
        collator = UnoDataCollator(tokenizer, sequence_length=6)
        batch = collator(
            [{"input_ids": [1, 2, 0], "attention_mask": [1, 1, 1], "labels": [-100, 2, 0]}]
        )
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 0, 0, 0, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2, 0, -100, -100, -100]])

    def test_content_hash_is_order_sensitive(self):
        rows = [
            {"input_ids": [1, 2], "labels": [-100, 2]},
            {"input_ids": [3], "labels": [3]},
        ]
        forward = content_hash_rows(rows)
        reverse = content_hash_rows(reversed(rows))
        self.assertNotEqual(forward["content_sha256"], reverse["content_sha256"])
        self.assertEqual(forward["row_count"], 2)

    def test_reference_hash_accepts_train_dataset_dict(self):
        from datasets import Dataset, DatasetDict

        from training.prepare_openthoughts import _reference_hash

        rows = [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3], "attention_mask": [1], "labels": [3]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference"
            DatasetDict({"train": Dataset.from_list(rows)}).save_to_disk(str(path))
            reference_hash = _reference_hash(path)
        self.assertEqual(reference_hash, content_hash_rows(rows))


class ObjectiveTest(unittest.TestCase):
    def test_uniform_noise_is_set_before_model_construction(self):
        config = SimpleNamespace()
        self.assertIs(configure_uniform_noise_config(config), config)
        self.assertEqual(config.noise, "uniform")
        self.assertEqual(
            configure_uniform_noise_config(SimpleNamespace(noise="uniform")).noise,
            "uniform",
        )
        with self.assertRaisesRegex(RuntimeError, "requires uniform noise"):
            configure_uniform_noise_config(SimpleNamespace(noise="masking"))

    def test_teacher_and_ground_truth_targets(self):
        shifted = torch.tensor([[4, 5, -100]])
        mask = torch.tensor([[True, True, False]])
        teacher = torch.tensor([[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]])
        self.assertEqual(
            resolve_ce_targets("data", shifted, mask, teacher).tolist(), [4, 5]
        )
        self.assertEqual(
            resolve_ce_targets("clean_argmax", shifted, mask, teacher).tolist(),
            [1, 0],
        )

    def test_reverse_kl_matches_literal_formula(self):
        student = torch.tensor([[0.2, -0.4, 0.7]], requires_grad=True)
        teacher = torch.tensor([[0.6, 0.0, -0.5]])
        actual = token_normalized_reverse_kl(student, teacher, 1)
        student_log = F.log_softmax(student, dim=-1)
        teacher_log = F.log_softmax(teacher, dim=-1)
        expected = (student_log.exp() * (student_log - teacher_log)).sum()
        torch.testing.assert_close(actual, expected)

    def test_tv_matches_literal_l1_and_has_student_gradient_only(self):
        student = torch.tensor([[0.2, -0.4, 0.7]], requires_grad=True)
        teacher = torch.tensor([[0.6, 0.0, -0.5]], requires_grad=True)
        actual = token_normalized_total_variation(student, teacher, 1, chunk_size=2)
        expected = (student.softmax(-1) - teacher.softmax(-1)).abs().sum()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_unsupported_objectives_are_rejected(self):
        objective = UnoObjectiveConfig()
        self.assertEqual(
            (objective.ce_weight, objective.kl_weight, objective.tv_weight),
            (0.0, 0.0, 1.0),
        )
        with self.assertRaises(ValueError):
            UnoObjectiveConfig(noise="mask").validate()
        with self.assertRaises(ValueError):
            UnoObjectiveConfig(reverse_kl=False).validate()
        with self.assertRaises(ValueError):
            UnoObjectiveConfig(
                ce_weight=0.0, kl_weight=0.0, tv_weight=0.0
            ).validate()
        with self.assertRaises(ValueError):
            UnoObjectiveConfig(ce_weight=-1.0).validate()

    def test_objective_weights_select_tv_only_by_default(self):
        ce = torch.tensor(2.0)
        kl = torch.tensor(3.0)
        tv = torch.tensor(5.0)
        self.assertEqual(
            combine_objective_losses(ce, kl, tv, UnoObjectiveConfig()), tv
        )
        weighted = combine_objective_losses(
            ce,
            kl,
            tv,
            UnoObjectiveConfig(ce_weight=2.0, kl_weight=3.0, tv_weight=4.0),
        )
        self.assertEqual(weighted, torch.tensor(33.0))


class LoraTest(unittest.TestCase):
    def test_saved_adapter_metadata_uses_the_public_pinned_base(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text(
                '{"base_model_name_or_path": "/private/cache/snapshot", "revision": null}\n'
            )
            (adapter / "README.md").write_text(
                "---\nbase_model: /private/cache/snapshot\nlibrary_name: peft\n---\n"
            )

            make_saved_adapter_portable(adapter)

            config = json.loads((adapter / "adapter_config.json").read_text())
            self.assertEqual(config["base_model_name_or_path"], "s-sahoo/uno-qwen3-8B")
            self.assertEqual(
                config["revision"],
                "b8a7577b3223bdcf2b3af0f2fc6e95258b3bbc29",
            )
            self.assertIn(
                "base_model: s-sahoo/uno-qwen3-8B",
                (adapter / "README.md").read_text(),
            )

    def test_target_presets_and_default_alpha_ratio(self):
        self.assertEqual(DEFAULT_LORA_RANK, 128)
        self.assertEqual(DEFAULT_LORA_ALPHA, 2048)
        self.assertEqual(resolve_lora_targets("o"), ("o_proj",))
        self.assertEqual(resolve_lora_targets("qkvo"), ("q_proj", "k_proj", "v_proj", "o_proj"))
        self.assertEqual(resolve_lora_targets("gate_proj,down_proj"), ("gate_proj", "down_proj"))
        self.assertEqual(LoraSettings("all", 256, None, 0.0).resolved_alpha, 4096)
        with self.assertRaises(ValueError):
            resolve_lora_targets("lm_head")

    def test_token_router_zeros_clean_rows(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = torch.nn.ModuleDict(
                    {"default": torch.nn.Linear(3, 2, bias=False)}
                )

        model = FakeModel()
        router = TokenwiseLoraRouter(model)
        router.set_token_mask(torch.tensor([[True, False]]))
        output = model.lora_A["default"](torch.ones(1, 2, 3))
        self.assertFalse(torch.equal(output[0, 0], torch.zeros(2)))
        torch.testing.assert_close(output[0, 1], torch.zeros(2))
        router.close()


class CurriculumTest(unittest.TestCase):
    def test_fixed_b8_epoch_schedule(self):
        one_epoch = build_fixed_curriculum("1")
        tenth_epoch = build_fixed_curriculum("0.1")

        self.assertEqual(one_epoch["stages"], [{
            "block_size": 8,
            "tokens": 9_375 * 4_096 * 128,
        }])
        self.assertEqual(tenth_epoch["stages"][0]["tokens"], 938 * 4_096 * 128)
        self.assertEqual(tenth_epoch["requested_epochs"], "0.1")

    def test_fixed_curriculum_rejects_invalid_epochs(self):
        for epochs in ("0", "-1", "nan", "not-a-number"):
            with self.subTest(epochs=epochs):
                with self.assertRaises(ValueError):
                    build_fixed_curriculum(epochs)

    def test_default_six_stage_boundaries(self):
        plan = BlockCurriculumPlan.from_yaml(
            ROOT / "training/configs/uno_3epoch_curriculum.yaml"
        )
        self.assertEqual(plan.max_steps, 28125)
        self.assertEqual(
            [stage.end_step for stage in plan.stages],
            [4688, 9375, 14063, 18750, 23438, 28125],
        )
        self.assertEqual(plan.stage_for_step(0).block_size, 2)
        self.assertEqual(plan.stage_for_step(4688).block_size, 4)
        plan.validate_runtime(world_size=16, per_device_batch=8, accumulation=1)
        with self.assertRaises(ValueError):
            plan.validate_runtime(world_size=8, per_device_batch=8, accumulation=1)

    def test_replay_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curriculum.yaml"
            path.write_text(
                "tokens_per_step: 8\nsequence_length: 4\nglobal_batch_size: 2\n"
                "seed: 1\nreplay: {enabled: true}\nstages:\n"
                "  - {block_size: 2, tokens: 8}\n"
            )
            with self.assertRaises(ValueError):
                BlockCurriculumPlan.from_yaml(path)


class LauncherTest(unittest.TestCase):
    def test_training_cli_defaults_match_released_recipe(self):
        args = build_parser().parse_args(
            ["--dataset-path", "/data", "--output-dir", "/output"]
        )
        self.assertEqual(
            (args.ce_alpha, args.kl_beta, args.tv_gamma), (0.0, 0.0, 1.0)
        )
        self.assertEqual(args.lora_rank, 128)
        self.assertIsNone(args.lora_alpha)

    def test_multinode_launcher_uses_shared_repository_path(self):
        launcher = (ROOT / "training" / "run_slurm.sh").read_text()
        self.assertIn(
            'LAUNCHER_PATH="${REPO_ROOT}/training/run_slurm.sh"',
            launcher,
        )
        self.assertIn('if [[ -n "${UNO_REPO_ROOT:-}" ]]', launcher)
        self.assertIn('export UNO_REPO_ROOT="${REPO_ROOT}"', launcher)
        self.assertIn('worker_tmpdir="${SLURM_TMPDIR}/uno-${SLURM_PROCID}"', launcher)
        self.assertIn(
            'worker_tmpdir="/tmp/uno-${USER:-user}-${SLURM_JOB_ID}-${SLURM_PROCID}"',
            launcher,
        )
        self.assertIn('export TMPDIR="${worker_tmpdir}"', launcher)
        self.assertIn('bash "${LAUNCHER_PATH}"', launcher)
        self.assertNotIn(
            'srun --nodes="${NODES}" --ntasks="${NODES}" '
            '--ntasks-per-node=1 "${BASH_SOURCE[0]}"',
            launcher,
        )

    def test_schedule_wrappers_delegate_to_generic_launcher(self):
        three_epoch = (ROOT / "examples" / "uno_qwen3_8B" / "run_train.sh").read_text()
        fixed_b8 = (ROOT / "training" / "run_fixed_b8_slurm.sh").read_text()
        self.assertIn("training/configs/uno_3epoch_curriculum.yaml", three_epoch)
        self.assertIn('training/run_slurm.sh"', three_epoch)
        self.assertIn('training/run_slurm.sh"', fixed_b8)

    def test_batch_spool_copy_preserves_shared_repository_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spool_copy = root / "slurm_script"
            spool_copy.write_text(
                (ROOT / "training" / "run_slurm.sh").read_text()
            )

            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "uno_dataset_manifest.json").write_text("{}\n")
            storage = root / "training"
            commands = root / "commands.log"
            mock_bin = root / "bin"
            mock_bin.mkdir()
            (mock_bin / "srun").write_text(
                "#!/usr/bin/env bash\n"
                'printf \'%s\\n\' "$*" >> "$UNO_LAUNCH_LOG"\n'
            )
            (mock_bin / "scontrol").write_text(
                "#!/usr/bin/env bash\nprintf 'node0\\nnode1\\n'\n"
            )
            (mock_bin / "srun").chmod(0o755)
            (mock_bin / "scontrol").chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{mock_bin}:{env['PATH']}",
                    "UNO_REPO_ROOT": str(ROOT),
                    "UNO_LAUNCH_LOG": str(commands),
                    "DATASET_PATH": str(dataset),
                    "TRAIN_STORAGE_ROOT": str(storage),
                    "CURRICULUM_PATH": str(
                        ROOT / "training" / "configs" / "uno_3epoch_curriculum.yaml"
                    ),
                    "SLURM_JOB_ID": "123",
                    "SLURM_JOB_NODELIST": "fake",
                    "SLURM_NNODES": "2",
                    "NODES": "2",
                    "GPUS_PER_NODE": "1",
                    "PER_DEVICE_BATCH_SIZE": "8",
                    "GLOBAL_BATCH_SIZE": "16",
                }
            )
            result = subprocess.run(
                ["bash", str(spool_copy)],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            launched = commands.read_text()
            self.assertIn(
                f"bash {ROOT}/training/run_slurm.sh",
                launched,
            )
            self.assertNotIn(str(root / "scripts"), launched)

    def test_worker_uses_short_local_ipc_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spool_copy = root / "slurm_script"
            spool_copy.write_text(
                (ROOT / "training" / "run_slurm.sh").read_text()
            )
            command_log = root / "worker.log"
            fake_python = root / "python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                'printf \'TMPDIR=%s\\nARGS=%s\\n\' "$TMPDIR" "$*" '
                '> "$UNO_LAUNCH_LOG"\n'
            )
            fake_python.chmod(0o755)
            local_tmp = root / "ipc"

            env = os.environ.copy()
            env.update(
                {
                    "UNO_REPO_ROOT": str(ROOT),
                    "UNO_NODE_WORKER": "1",
                    "UNO_LOCAL_TMPDIR": str(local_tmp),
                    "UNO_LAUNCH_LOG": str(command_log),
                    "PYTHON_BIN": str(fake_python),
                    "DATASET_PATH": str(root / "dataset"),
                    "TRAIN_STORAGE_ROOT": str(root / "training"),
                    "CURRICULUM_PATH": str(
                        ROOT / "training" / "configs" / "uno_3epoch_curriculum.yaml"
                    ),
                    "SLURM_JOB_ID": "123",
                    "SLURM_NNODES": "1",
                    "SLURM_PROCID": "0",
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": "23456",
                    "NODES": "1",
                    "GPUS_PER_NODE": "1",
                    "PER_DEVICE_BATCH_SIZE": "1",
                    "GLOBAL_BATCH_SIZE": "1",
                }
            )
            result = subprocess.run(
                ["bash", str(spool_copy)],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            launched = command_log.read_text()
            self.assertIn(f"TMPDIR={local_tmp}\n", launched)
            self.assertIn("--ce-alpha 0.0", launched)
            self.assertIn("--kl-beta 0.0", launched)
            self.assertIn("--tv-gamma 1.0", launched)
            self.assertIn("--lora-rank 128", launched)
            self.assertTrue(local_tmp.is_dir())


if __name__ == "__main__":
    unittest.main()
