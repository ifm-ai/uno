# K2 Horizon evaluation scripts

These wrappers pin the K2 Horizon model family, linear block size 8, and the
K2 token IDs used by the published checkpoints. They delegate to
`scripts/common/run_benchmark.sh` so Qwen and K2 share one generation and
scoring implementation.

Run a single benchmark directly, for example:

```bash
RESULTS_ROOT=/path/to/results \
  bash scripts/k2_horizon/run_gsm8k_linear_b8.sh
```

The Slurm array launcher runs GSM8K, MATH500, and AIME 2024--2026:

```bash
RESULTS_ROOT=/path/to/shared/results \
  sbatch scripts/k2_horizon/run_k2_horizon_7b_tpf_suite.sbatch
```
