import csv
import datetime as dt
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from patchtst_config import BASE, PAPER, command

TZ = dt.timezone(dt.timedelta(hours=8))
RESULTS = BASE/'results'
STATUS = RESULTS/'status.json'

def now():
    return dt.datetime.now(TZ).isoformat(timespec='seconds')

def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)

def log(message):
    print(f'[{now()}] {message}', flush=True)

def summarize(rows):
    if rows:
        with (RESULTS/'summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

def main():
    import fcntl
    lock=(RESULTS/'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    benchmark=json.loads((BASE/'environment/benchmark.json').read_text())
    rows=[]
    jobs=[(d,h) for d in ('ETTh2','ETTm2') for h in (96,192,336,720)]
    state=dict(status='running', started_at=now(), completed=[], total=len(jobs), pid=os.getpid())
    for index,(dataset,horizon) in enumerate(jobs,1):
        out=RESULTS/dataset/f'pred_{horizon}_seed_2021'
        # Do not overwrite any prior experiment, including partially completed runs.
        out.mkdir(parents=True,exist_ok=False)
        cmd=command(dataset,horizon)
        (out/'command.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(out))+'\nexport OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=2021\n'+shlex.join(cmd)+'\n')
        estimate=benchmark[dataset][str(horizon)]['estimated_100_epochs_minutes']
        lo=max(1,round(estimate*0.7))
        hi=max(lo+1,math.ceil(estimate*1.4+2))
        log(f'START {index}/{len(jobs)} {dataset} 336 -> {horizon}; estimated {lo}-{hi} minutes (100 epochs; ETTm2 may stop early)')
        started=time.monotonic()
        state.update(current=f'{dataset}/{horizon}',current_started_at=now(),epoch=0,eta_minutes=[lo,hi],log=str(out/'train.log'))
        write_json(STATUS,state)
        env=os.environ.copy()
        env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONHASHSEED='2021',PYTHONUNBUFFERED='1')
        epochs=[]
        with (out/'train.log').open('w',buffering=1) as training_log:
            child=subprocess.Popen(cmd,cwd=out,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            state['child_pid']=child.pid
            write_json(STATUS,state)
            last_epoch=time.monotonic()
            for line in child.stdout:
                stamped=f'[{now()}] {line.rstrip()}'
                print(stamped,flush=True)
                training_log.write(stamped+'\n')
                match=re.search(r'Epoch: (\d+), Steps: (\d+) \| Train Loss: ([\d.eE+-]+) Vali Loss: ([\d.eE+-]+) Test Loss: ([\d.eE+-]+)',line)
                if match:
                    epoch=int(match[1])
                    measured=time.monotonic()-last_epoch
                    last_epoch=time.monotonic()
                    epochs.append(dict(epoch=epoch,train_loss=float(match[3]),val_loss=float(match[4]),test_loss=float(match[5]),epoch_wall_seconds=measured))
                    write_json(out/'epochs.json',epochs)
                    average=sum(e['epoch_wall_seconds'] for e in epochs[-5:])/len(epochs[-5:])
                    state.update(epoch=epoch,last_update=now(),estimated_max_remaining_minutes=round(average*(100-epoch)/60,1))
                    write_json(STATUS,state)
            rc=child.wait()
        if rc:
            state.update(status='failed',exit_code=rc,finished_at=now())
            write_json(STATUS,state)
            raise RuntimeError(f'{dataset}/{horizon} exited {rc}; see {out}/train.log')
        metric_files=list((out/'results').glob('*/metrics.json'))
        checkpoint_files=list((out/'checkpoints').glob('*/checkpoint.pth'))
        assert len(metric_files)==len(checkpoint_files)==1
        metrics=json.loads(metric_files[0].read_text())
        pred=np.load(metric_files[0].parent/'pred.npy',mmap_mode='r')
        true=np.load(metric_files[0].parent/'true.npy',mmap_mode='r')
        assert pred.shape==true.shape and pred.shape[1:]==(horizon,7)
        assert np.isfinite(pred).all() and np.isfinite(true).all()
        # Independent, chunked float64 reduction of the saved predictions/targets.
        squared=absolute=count=0
        for i in range(0,len(pred),128):
            difference=pred[i:i+128].astype(np.float64)-true[i:i+128].astype(np.float64)
            squared+=np.square(difference).sum()
            absolute+=np.abs(difference).sum()
            count+=difference.size
        verified_mse,verified_mae=squared/count,absolute/count
        assert math.isclose(verified_mse,metrics['mse'],rel_tol=1e-5,abs_tol=1e-7)
        assert math.isclose(verified_mae,metrics['mae'],rel_tol=1e-5,abs_tol=1e-7)
        elapsed=(time.monotonic()-started)/60
        paper_mse,paper_mae=PAPER[dataset][horizon]
        row=dict(dataset=dataset,model='PatchTST/42',seq_len=336,pred_len=horizon,seed=2021,
            mse=metrics['mse'],mae=metrics['mae'],paper_mse=paper_mse,paper_mae=paper_mae,
            mse_delta=metrics['mse']-paper_mse,mae_delta=metrics['mae']-paper_mae,
            epochs=len(epochs),elapsed_minutes=elapsed,test_samples=len(pred),verified=True)
        rows.append(row)
        write_json(out/'completion.json',dict(**row,completed_at=now(),verified_mse=verified_mse,verified_mae=verified_mae))
        summarize(rows)
        state['completed'].append(row)
        write_json(STATUS,state)
        log(f'DONE {index}/{len(jobs)} {dataset}/{horizon}: MSE={metrics["mse"]:.6f}, MAE={metrics["mae"]:.6f}; {elapsed:.1f} minutes; artifacts verified')
    state.update(status='complete',finished_at=now(),current=None)
    write_json(STATUS,state)
    log('ALL 8 EXPERIMENTS COMPLETE AND VERIFIED')

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        log(f'ERROR: {type(exc).__name__}: {exc}')
        if STATUS.exists():
            state=json.loads(STATUS.read_text())
            state.update(status='failed',error=str(exc),finished_at=now())
            write_json(STATUS,state)
        raise
