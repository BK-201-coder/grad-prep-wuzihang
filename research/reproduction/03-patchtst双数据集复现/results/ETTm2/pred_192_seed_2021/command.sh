#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/patchtst_repro/results/ETTm2/pred_192_seed_2021
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=2021
/root/autodl-tmp/patchtst_repro/environment/venv/bin/python -u /root/autodl-tmp/patchtst_repro/code/PatchTST_supervised/run_longExp.py --random_seed 2021 --is_training 1 --root_path /root/autodl-tmp/patchtst_repro/data --data_path ETTm2.csv --model_id ETTm2_336_192 --model PatchTST --data ETTm2 --features M --target OT --freq h --seq_len 336 --label_len 48 --pred_len 192 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 --patch_len 16 --stride 8 --padding_patch end --revin 1 --affine 0 --subtract_last 0 --decomposition 0 --kernel_size 25 --individual 0 --des Exp --train_epochs 100 --patience 20 --lradj TST --pct_start 0.4 --itr 1 --batch_size 128 --learning_rate 0.0001 --num_workers 4 --embed timeF
