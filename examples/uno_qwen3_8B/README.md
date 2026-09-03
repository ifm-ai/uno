# uno_qwen3_8B

This recipe uses the public `s-sahoo/uno-qwen3-8B` bundle. Base weights live
at the repository root and the conditional LoRA adapter lives under `adapter/`.

```bash
bash examples/uno_qwen3_8B/run_inference.sh --prompt "Solve 2 + 2."
bash examples/uno_qwen3_8B/run_eval.sh gsm8k
```

`run_train.sh` launches the published Qwen Uno LoRA training recipe. Set
`DATASET_PATH` and `TRAIN_STORAGE_ROOT` before running it.
