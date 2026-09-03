# uno_8B

This recipe combines `IFM/K2-Horizon-7B` with the conditional LoRA adapter in
`IFM/K2-Horizon-7B-Uno`.

```bash
bash examples/uno_8B/run_inference.sh --prompt "Solve 2 + 2."
bash examples/uno_8B/run_eval.sh gsm8k
```

The release provides inference and evaluation. This repository does not expose
a K2 training recipe, so this directory intentionally has no `run_train.sh`.
