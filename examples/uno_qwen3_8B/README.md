# uno_qwen3_8B

This recipe uses the public `s-sahoo/uno-qwen3-8B` bundle. Base weights live
at the repository root and the conditional LoRA adapter lives under `adapter/`.

```bash
cd examples/uno_qwen3_8B
./run_inference.sh --prompt "Solve 2 + 2."
./run_eval.sh gsm8k
```

The launchers resolve the repository root automatically, so they work from
this directory without changing the current working directory themselves.

`run_train.sh` launches the published Qwen Uno LoRA training recipe. Set
`DATASET_PATH` and `TRAIN_STORAGE_ROOT` before running it.
