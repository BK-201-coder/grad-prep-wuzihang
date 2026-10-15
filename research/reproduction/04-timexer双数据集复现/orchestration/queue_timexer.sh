#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/timexer_repro
RUN="$ROOT/orchestration/run_timexer.sh"
on_error() {
  code=$?
  echo "$(date -Is) QUEUE STOPPED after an error (exit $code). Existing results were not overwritten."
  exec bash -i
}
trap on_error ERR

echo "$(date -Is) TimeXer queue begins: 7 remaining official exogenous ETT runs."
echo "Watch this session with: screen -x timexer"

bash "$RUN" ETTh2 192 '1-3 minutes'
bash "$RUN" ETTh2 336 '3-10 minutes'
bash "$RUN" ETTh2 720 '2-6 minutes'
bash "$RUN" ETTm2 96 '10-40 minutes'
bash "$RUN" ETTm2 192 '20-90 minutes'
bash "$RUN" ETTm2 336 '4-12 minutes'
bash "$RUN" ETTm2 720 '4-12 minutes'

echo "$(date -Is) ALL 8 EXPERIMENTS COMPLETED AND VERIFIED"
echo 'The session will remain open for viewing. Detach with Ctrl+A, then D.'
exec bash -i
