#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mesogeos_repro
results="$root/results"
code="$root/code/ml_tracks/a.fire_danger"
python="$root/venv/bin/python"
validation="$results/data_validation_2023.json"
config="$results/config_train.json"
log="$results/run.log"

if [[ ! -f "$validation" ]]; then
  echo "REFUSING_TRAINING: data validation report is missing"
  exit 2
fi
"$python" -c 'import json,sys; assert json.load(open(sys.argv[1]))["matches_paper_positive_splits_and_negative_pool"] is True' "$validation"
if [[ -e "$log" || -e "$results/COMPLETE.json" ]]; then
  echo "REFUSING_TRAINING: existing run artifacts must be preserved"
  exit 2
fi
exec > >(tee "$log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$results/exit_code.txt"; echo "EXIT_CODE $status $(date -Is)"' EXIT

echo "START $(date -Is) official Mesogeos Track A LSTM"
echo "CONFIG $config"
echo "SOURCE_COMMIT $(git -C "$root/code" rev-parse HEAD)"
cd "$code"
"$python" -u train.py -c "$config"
model_dir="$results/models/mesogeos_trackA_lstm_seed12345"
mapfile -t best_models < <(find "$model_dir" -name model_best.pth -type f)
if [[ ${#best_models[@]} -ne 1 ]]; then
  echo "REFUSING_TEST: expected exactly one best checkpoint, found ${#best_models[@]}"
  exit 3
fi
best="${best_models[0]}"
echo "$best" > "$results/model_path.txt"
"$python" - "$config" "$best" "$results/config_test.json" <<'PY'
import json
import pathlib
import sys
config = json.loads(pathlib.Path(sys.argv[1]).read_text())
config['model_path'] = sys.argv[2]
pathlib.Path(sys.argv[3]).write_text(json.dumps(config, indent=2) + '\n')
PY
echo "TEST_START $(date -Is) best_checkpoint=$best"
"$python" -u test.py -c "$results/config_test.json"
echo "RUN_FINISHED $(date -Is)"
