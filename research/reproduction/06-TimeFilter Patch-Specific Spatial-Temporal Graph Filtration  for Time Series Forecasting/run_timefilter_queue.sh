#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/timefilter_repro
CODE="$ROOT/code"
PY="$ROOT/venv/bin/python"
RUNS="$ROOT/runs"
PROGRESS="$ROOT/progress.log"
mkdir -p "$RUNS"
cd "$CODE"

stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
log() { printf '[%s] %s\n' "$(stamp)" "$*" | tee -a "$PROGRESS"; }

python() {
    local previous='' arg run_id='' estimate='' run_dir='' result_dir='' checkpoint_dir='' rc=0
    for arg in "$@"; do
        if [[ "$previous" == '--model_id' ]]; then run_id="$arg"; break; fi
        previous="$arg"
    done
    if [[ ! "$run_id" =~ ^(ETTh2|ETTm2)_96_(96|192|336|720)$ ]]; then
        log "ERROR: unexpected model_id: $run_id"
        return 2
    fi
    run_dir="$RUNS/$run_id"
    if [[ -e "$run_dir/COMPLETE.json" ]]; then
        log "SKIP $run_id: already verified"
        return 0
    fi
    if [[ -e "$run_dir/STARTED" ]]; then
        log "STOP $run_id: previous incomplete run found; inspect before any rerun"
        return 3
    fi
    if [[ "$run_id" == ETTh2* ]]; then estimate='about 20-75 minutes'; else estimate='about 35-120 minutes'; fi
    mkdir -p "$run_dir"
    log "START $run_id; estimate $estimate (10 epoch cap; early stopping may shorten it)"
    date -Is > "$run_dir/STARTED"
    printf '%q ' "$@" > "$run_dir/command.txt"
    printf '\n' >> "$run_dir/command.txt"
    set +e
    CUDA_VISIBLE_DEVICES=0 "$PY" "$@" 2>&1 | tee "$run_dir/train.log"
    rc=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$rc" > "$run_dir/exit_code.txt"
    if (( rc != 0 )); then
        log "FAIL $run_id: exit code $rc; queue stopped; see $run_dir/train.log"
        return "$rc"
    fi
    result_dir=$(find "$CODE/results" -maxdepth 1 -type d -name "long_term_forecast_${run_id}_TimeFilter_*" -print -quit)
    checkpoint_dir=$(find "$CODE/checkpoints" -maxdepth 1 -type d -name "long_term_forecast_${run_id}_TimeFilter_*" -print -quit)
    if [[ -z "$result_dir" || -z "$checkpoint_dir" || ! -f "$checkpoint_dir/checkpoint.pth" ]]; then
        log "FAIL $run_id: missing result directory or checkpoint; queue stopped"
        return 4
    fi
    "$PY" -c 'import json, math, os, sys, numpy as np; r,c=sys.argv[1:]; m=np.load(os.path.join(r,"metrics.npy")); p=np.load(os.path.join(r,"pred.npy"),mmap_mode="r"); t=np.load(os.path.join(r,"true.npy"),mmap_mode="r"); assert m.shape==(5,) and np.isfinite(m).all(); assert p.shape==t.shape and p.size>0; assert os.path.getsize(os.path.join(c,"checkpoint.pth"))>0; print(json.dumps({"verified":True,"metrics":{"mae":float(m[0]),"mse":float(m[1]),"rmse":float(m[2]),"mape":float(m[3]),"mspe":float(m[4])},"prediction_shape":list(p.shape),"result_dir":r,"checkpoint_dir":c},indent=2))' "$result_dir" "$checkpoint_dir" > "$run_dir/COMPLETE.json"
    log "DONE $run_id: verified model, predictions, targets and metrics"
}

log 'TimeFilter 8-run queue started (official ETTh2/ETTm2 hyperparameters; GPU remapped to visible device 0)'
source scripts/ETTh2.sh
source scripts/ETTm2.sh
log 'ALL 8 RUNS COMPLETE'
