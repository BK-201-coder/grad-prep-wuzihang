#!/usr/bin/env bash
set -u

pred_len="${1:?prediction length is required}"
case "$pred_len" in
  96|336)
    d_model=4
    k=2
    patch_sizes=(16 12 8 32 12 8 6 4 8 6 4 2)
    ;;
  192)
    d_model=8
    k=2
    patch_sizes=(16 12 8 32 12 8 6 4 8 6 4 2)
    ;;
  720)
    d_model=16
    k=3
    patch_sizes=(16 12 8 32 12 8 6 32 8 6 16 12)
    ;;
  *)
    echo "Unsupported prediction length: $pred_len" >&2
    exit 2
    ;;
esac

project=/root/autodl-tmp/pathformer_repro/pathformer
run_id="ETTh2_96_${pred_len}_repro_20261013"
run_dir="$project/reproductions/$run_id"
setting="${run_id}_PathFormer_ftETTh2_slM_pl96_${pred_len}"

if [ -e "$run_dir/started_at.txt" ]; then
  echo "Refusing to overwrite existing run: $run_dir" >&2
  exit 3
fi
mkdir -p "$run_dir"
cd "$project" || exit 1
export PATH="/root/miniconda3/bin:$PATH"

date -Is > "$run_dir/started_at.txt"
git rev-parse HEAD > "$run_dir/git_commit.txt"
git status --short > "$run_dir/git_status_before.txt"
sha256sum dataset/ETT/ETTh2.csv > "$run_dir/dataset_sha256.txt"
python -m pip freeze > "$run_dir/pip_freeze.txt"
nvidia-smi > "$run_dir/gpu_at_start.txt"
printf '%s\n' "$setting" > "$run_dir/setting.txt"
printf 'pred_len=%s\nd_model=%s\nk=%s\npatch_sizes=%s\n' \
  "$pred_len" "$d_model" "$k" "${patch_sizes[*]}" > "$run_dir/config.txt"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT/ \
  --data_path ETTh2.csv \
  --model_id "$run_id" \
  --model PathFormer \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len "$pred_len" \
  --patch_size_list "${patch_sizes[@]}" \
  --num_nodes 7 \
  --layer_nums 3 \
  --batch_norm 0 \
  --residual_connection 0 \
  --k "$k" \
  --d_model "$d_model" \
  --d_ff 64 \
  --train_epochs 30 \
  --patience 10 \
  --lradj TST \
  --itr 1 \
  --batch_size 512 \
  --learning_rate 0.0005 \
  --checkpoints "$run_dir/checkpoints/" \
  > "$run_dir/train.log" 2>&1
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
  grep -E '^mse:' "$run_dir/train.log" | tail -n 1 > "$run_dir/metrics.txt"
  if [ ! -s "$run_dir/metrics.txt" ] || [ ! -f "$run_dir/checkpoints/$setting/checkpoint.pth" ]; then
    exit_code=2
  else
    sha256sum "$run_dir/checkpoints/$setting/checkpoint.pth" > "$run_dir/checkpoint_sha256.txt"
    nvidia-smi > "$run_dir/gpu_at_end.txt"
  fi
fi

printf '%s\n' "$exit_code" > "$run_dir/exit_code.txt"
date -Is > "$run_dir/finished_at.txt"
exit "$exit_code"
