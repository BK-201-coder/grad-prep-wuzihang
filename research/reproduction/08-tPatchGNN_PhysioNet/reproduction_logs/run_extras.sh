#!/usr/bin/env bash
set -o pipefail
cd /root/autodl-tmp/t-PatchGNN-main/tPatchGNN
for seed in 2 3; do
  python -u run_models.py --dataset physionet --state seed${seed} --history 24 --patience 10 --batch_size 32 --lr 1e-3 --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 --te_dim 10 --node_dim 10 --hid_dim 64 --outlayer Linear --seed ${seed} --gpu 0 --epoch 1000 --logmode w 2>&1 | tee reproduction_logs/seed${seed}.console.log
  code=${PIPESTATUS[0]}
  echo "$code" > reproduction_logs/seed${seed}.exit.txt
  if [ "$code" -ne 0 ]; then exit "$code"; fi
done
cd /root/autodl-tmp/t-PatchGNN-main/tPatchGNN_wo_transformer
python -u run_models.py --dataset physionet --state wo_transformer --history 24 --patience 10 --batch_size 32 --lr 1e-3 --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 --te_dim 10 --node_dim 10 --hid_dim 64 --outlayer Linear --seed 1 --gpu 0 --epoch 1000 --logmode w 2>&1 | tee ../tPatchGNN/reproduction_logs/wo_transformer_seed1.console.log
code=${PIPESTATUS[0]}
echo "$code" > ../tPatchGNN/reproduction_logs/wo_transformer_seed1.exit.txt
date -Is > ../tPatchGNN/reproduction_logs/extras_finished_at.txt
exit "$code"
