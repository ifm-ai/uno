#!/usr/bin/env bash

uno_cuda_graph_batch_sizes() {
  local max_num_seqs="$1"
  if [[ ! "${max_num_seqs}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_NUM_SEQS must be a positive integer, got ${max_num_seqs}" >&2
    return 2
  fi

  local -a batch_sizes=()
  local batch_size=1
  local last_batch_size=0
  while (( batch_size <= max_num_seqs )); do
    batch_sizes+=("${batch_size}")
    last_batch_size="${batch_size}"
    batch_size=$((batch_size * 2))
  done
  if (( last_batch_size != max_num_seqs )); then
    batch_sizes+=("${max_num_seqs}")
  fi

  local IFS=,
  printf '%s\n' "${batch_sizes[*]}"
}
