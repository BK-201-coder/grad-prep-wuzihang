#!/usr/bin/env bash
set -o pipefail
cd /root/autodl-tmp/t-PatchGNN-main/tPatchGNN
python -u run_models.py --dataset physionet --state def --history 24 --patience 10 --batch_size 32 --lr 1e-3 --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 --te_dim 10 --node_dim 10 --hid_dim 64 --outlayer Linear --seed 1 --gpu 0 --epoch 1000 --logmode w 2>&1 | tee reproduction_logs/B_physionet_seed1.console.log
status=${PIPESTATUS[0]}
echo "$status" > reproduction_logs/B_exit_code.txt
date -Is > reproduction_logs/B_finished_at.txt
