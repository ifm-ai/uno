# Shared evaluation scripts

`run_benchmark.sh` is the single benchmark generation and scoring
implementation used by all model families. Model-specific entry points set
`UNO_MODEL_DEFAULTS` and `UNO_RESULTS_NAMESPACE` before delegating here.
