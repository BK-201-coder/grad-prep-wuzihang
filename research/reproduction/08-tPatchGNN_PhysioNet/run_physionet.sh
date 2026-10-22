#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$repo_root/source/data/physionet/processed"
if [[ ! -f "$data_dir/set-a_0.0.pt" || ! -f "$data_dir/set-b_0.0.pt" || ! -f "$data_dir/set-c_0.0.pt" ]]; then
  echo "Missing PhysioNet processed data. Follow README.md to download and unzip processed.zip." >&2
  exit 1
fi

mkdir -p "$repo_root/run_logs"

run_one() {
  local code_dir="$1" state="$2" seed="$3"
  (
    cd "$repo_root/source/$code_dir"
    python -u run_models.py \
      --dataset physionet --state "$state" --history 24 --patience 10 \
      --batch_size 32 --lr 1e-3 --patch_size 8 --stride 8 \
      --nhead 1 --tf_layer 1 --nlayer 1 --te_dim 10 --node_dim 10 \
      --hid_dim 64 --outlayer Linear --seed "$seed" --gpu 0 \
      --epoch 1000 --logmode w
  ) 2>&1 | tee "$repo_root/run_logs/${state}.console.log"
}

for seed in 1 2 3; do
  run_one tPatchGNN "seed${seed}" "$seed"
done
run_one tPatchGNN_wo_transformer wo_transformer_seed1 1
