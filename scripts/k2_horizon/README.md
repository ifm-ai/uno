# K2 Horizon evaluation scripts

These wrappers pin the K2 Horizon model family, linear block size 8, and the
K2 token IDs used by the published checkpoints. They delegate to
`scripts/common/run_benchmark.sh` so Qwen and K2 share one generation and
scoring implementation.

## K2 Horizon 7B

The wrappers use `IFM/K2-Horizon-7B` with the conditional adapter from
`IFM/K2-Horizon-7B-Uno` by default. Run a single benchmark on an allocated
eight-GPU node, for example:

```bash
RESULTS_ROOT=/path/to/results \
  bash scripts/k2_horizon/run_gsm8k_linear_b8.sh
```

The Slurm array launcher runs GSM8K, MATH500, and AIME 2024--2026 with the 7B
defaults:

```bash
RESULTS_ROOT=/path/to/shared/results \
  sbatch scripts/k2_horizon/run_k2_horizon_7b_tpf_suite.sbatch
```

## K2 Horizon 0.9B

The 0.9B release uses the same benchmark wrappers but different model paths,
token IDs, and a 131,072-token context limit. On an allocated eight-GPU node,
run a single benchmark with:

```bash
MODEL=IFM/K2-Horizon-0.9B \
GATED_LORA_PATH=IFM/K2-Horizon-0.9B-Uno \
MASK_TOKEN_ID=64256 \
STOP_TOKEN_IDS=64019,1 \
MAX_MODEL_LEN=131072 \
MAX_NUM_BATCHED_TOKENS=131072 \
MAX_TOKENS=131072 \
RESULTS_ROOT=/path/to/results \
RUN_NAME=k2-horizon-09b-uno-linear-b8-n1 \
  bash scripts/k2_horizon/run_gsm8k_linear_b8.sh
```

To run the five available K2 wrappers sequentially in the same allocation:

```bash
export MODEL=IFM/K2-Horizon-0.9B
export GATED_LORA_PATH=IFM/K2-Horizon-0.9B-Uno
export MASK_TOKEN_ID=64256
export STOP_TOKEN_IDS=64019,1
export MAX_MODEL_LEN=131072
export MAX_NUM_BATCHED_TOKENS=131072
export MAX_TOKENS=131072
export RESULTS_ROOT=/path/to/results
export RUN_NAME=k2-horizon-09b-uno-linear-b8-n1

for benchmark in gsm8k math500 aime24 aime25 aime26; do
  bash "scripts/k2_horizon/run_${benchmark}_linear_b8.sh"
done
```

Both model sizes use linear block size 8 and one sample per problem in these
wrappers. Set `DATA_PARALLEL_SIZE=1` when running on a single visible GPU.
