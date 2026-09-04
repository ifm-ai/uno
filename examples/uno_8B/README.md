# uno_8B

This recipe combines `IFM/K2-Horizon-7B` with the conditional LoRA adapter in
`IFM/K2-Horizon-7B-Uno`.

```bash
cd examples/uno_8B
./run_inference.sh --prompt "Solve 2 + 2."
./run_eval.sh gsm8k
```

The launchers resolve the repository root automatically, so they work from
this directory without changing the current working directory themselves.

The release provides inference and evaluation. This repository does not expose
a K2 training recipe, so this directory intentionally has no `run_train.sh`.
