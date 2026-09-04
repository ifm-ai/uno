# uno_1B

This recipe combines `IFM/K2-Horizon-0.9B` with the conditional LoRA adapter in
`IFM/K2-Horizon-0.9B-Uno`. The launcher uses the model's 131,072-token YaRN
context and K2 token IDs.

```bash
cd examples/uno_1B
./run_inference.sh --prompt "Solve 2 + 2."
./run_eval.sh gsm8k
```

The launchers resolve the repository root automatically, so they work from
this directory without changing the current working directory themselves.

The release provides inference and evaluation. This repository does not expose
a K2 training recipe, so this directory intentionally has no `run_train.sh`.
