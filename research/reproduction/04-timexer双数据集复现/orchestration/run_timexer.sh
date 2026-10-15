#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/timexer_repro
DATASET=${1:?ETTh2 or ETTm2 required}
PRED=${2:?prediction length required}
ESTIMATE=${3:-unknown}
PY="$ROOT/venv/bin/python"
CODE="$ROOT/code"
RUN="$ROOT/results/$DATASET/pred_$PRED"

if [[ -f "$RUN/COMPLETE.json" ]]; then
  echo "Already complete: $DATASET/$PRED; refusing to overwrite."
  exit 0
fi
if [[ -e "$RUN/started_at.txt" ]]; then
  echo "Existing incomplete run: $RUN; refusing to overwrite."
  exit 2
fi

case "$DATASET:$PRED" in
  ETTh2:96)  E_LAYERS=1; D_MODEL=128; D_FF=128;  BATCH=128 ;;
  ETTh2:192) E_LAYERS=1; D_MODEL=128; D_FF=512;  BATCH=128 ;;
  ETTh2:336) E_LAYERS=2; D_MODEL=128; D_FF=256;  BATCH=16 ;;
  ETTh2:720) E_LAYERS=1; D_MODEL=256; D_FF=512;  BATCH=32 ;;
  ETTm2:96)  E_LAYERS=1; D_MODEL=512; D_FF=2048; BATCH=16 ;;
  ETTm2:192) E_LAYERS=1; D_MODEL=256; D_FF=2048; BATCH=4 ;;
  ETTm2:336) E_LAYERS=1; D_MODEL=128; D_FF=2048; BATCH=128 ;;
  ETTm2:720) E_LAYERS=1; D_MODEL=128; D_FF=2048; BATCH=128 ;;
  *) echo "Unsupported official-script configuration: $DATASET:$PRED"; exit 2 ;;
esac

mkdir -p "$RUN"
cd "$RUN"
date -Is > started_at.txt
exec > >(tee -a train.log) 2>&1
echo "$(date -Is) START $DATASET prediction=$PRED; estimated time: $ESTIMATE"
echo "Official code commit: $(git -C "$CODE" rev-parse HEAD)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

ARGS=(
  --task_name long_term_forecast --is_training 1
  --root_path "$ROOT/dataset/ETT-small/" --data_path "$DATASET.csv"
  --model_id "${DATASET}_96_${PRED}" --model TimeXer --data "$DATASET"
  --features MS --seq_len 96 --label_len 48 --pred_len "$PRED"
  --e_layers "$E_LAYERS" --factor 3 --enc_in 7 --dec_in 7 --c_out 7
  --d_model "$D_MODEL" --d_ff "$D_FF" --batch_size "$BATCH"
  --des Timexer-MS --itr 1 --gpu 0 --checkpoints "$RUN/checkpoints/"
)
printf '#!/usr/bin/env bash\n' > command.sh
printf '%q ' "$PY" "$CODE/run.py" "${ARGS[@]}" >> command.sh
printf '\n' >> command.sh
chmod +x command.sh

export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg
set +e
"$PY" -u "$CODE/run.py" "${ARGS[@]}"
CODE_EXIT=$?
set -e
printf '%s\n' "$CODE_EXIT" > exit_code.txt
date -Is > finished_at.txt
if [[ "$CODE_EXIT" -ne 0 ]]; then
  echo "$(date -Is) FAILED $DATASET/$PRED exit=$CODE_EXIT"
  exit "$CODE_EXIT"
fi
"$PY" "$ROOT/orchestration/summarize_timexer.py" "$RUN" "$DATASET" "$PRED"
echo "$(date -Is) VERIFIED $DATASET/$PRED"
