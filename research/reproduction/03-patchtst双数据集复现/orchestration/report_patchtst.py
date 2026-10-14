"""Generate a compact report and portable archives after verified completion."""
import csv
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patchtst_config import BASE

results=BASE/'results'
state=json.loads((results/'status.json').read_text())
assert state['status']=='complete' and len(state['completed'])==8
rows=state['completed']
lines=['# PatchTST/42 reproduction results', '',
       'Two datasets, four prediction lengths each, one run per configuration (seed 2021).',
       'Reference: Table 3, supervised multivariate PatchTST/42, ICLR 2023.',
       'Metrics are on standardized data, matching the upstream evaluation protocol.', '',
       '| Dataset | Horizon | MSE | Paper MSE | Delta | MAE | Paper MAE | Minutes |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
for row in rows:
    d,h=row['dataset'],row['pred_len']
    run=results/d/f'pred_{h}_seed_2021'
    assert json.loads((run/'completion.json').read_text())['verified']
    epochs=json.loads((run/'epochs.json').read_text())
    fig,ax=plt.subplots(figsize=(7,4))
    for key,label in [('train_loss','Train'),('val_loss','Validation')]:
        ax.plot([e['epoch'] for e in epochs],[e[key] for e in epochs],label=label)
    ax.set(xlabel='Epoch',ylabel='MSE loss (standardized)',title=f'{d}: 336 -> {h}, seed 2021')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(run/'learning_curve.png',dpi=160); plt.close(fig)
    artifact=next((run/'results').glob('*/metrics.json')).parent
    pred=np.load(artifact/'pred.npy',mmap_mode='r')
    true=np.load(artifact/'true.npy',mmap_mode='r')
    fig,ax=plt.subplots(figsize=(8,3.5))
    ax.plot(true[0,:,-1],label='Ground truth',linewidth=1.5)
    ax.plot(pred[0,:,-1],label='Prediction',linewidth=1.2)
    ax.set(xlabel='Forecast step',ylabel='OT (standardized)',title=f'{d}: first test window, horizon {h}')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(run/'forecast_first_window.png',dpi=160); plt.close(fig)
    lines.append(f'| {d} | {h} | {row["mse"]:.6f} | {row["paper_mse"]:.3f} | {row["mse_delta"]:+.6f} | {row["mae"]:.6f} | {row["paper_mae"]:.3f} | {row["elapsed_minutes"]:.1f} |')
lines += ['', '## Verification and limitations', '',
    'All eight saved predictions/targets passed shape, finite-value and independently recomputed MSE/MAE checks. Best checkpoints and full training configs are retained.',
    'Only one seed was run; these results do not measure variation across seeds. This is not a full-paper reproduction.',
    'Python/PyTorch and worker-count differences from the upstream setup are documented in README.md and environment/.',
    'The upstream evaluator drops the final incomplete test batch. Metrics should be compared using the same evaluation protocol.',
    'All hyperparameters were fixed before training. Test scores were not used to tune or select a rerun.', '']
(results/'REPORT.md').write_text('\n'.join(lines))
archives=BASE/'archives'
archives.mkdir(exist_ok=True)
files=[]
for root in ('code','data','environment','results','orchestration'):
    for p in (BASE/root).rglob('*'):
        if not p.is_file():
            continue
        rel=p.relative_to(BASE)
        if any(x in rel.parts for x in ('.git','__pycache__','venv','wheels','patchtst_wheels')):
            continue
        if p.name in ('runner.lock',):
            continue
        files.append(p)
files.append(BASE/'README.md')
with (archives/'SHA256SUMS.txt').open('w') as f:
    for p in sorted(files):
        digest=hashlib.sha256()
        with p.open('rb') as source:
            for block in iter(lambda:source.read(8*1024*1024),b''):
                digest.update(block)
        f.write(f'{digest.hexdigest()}  {p.relative_to(BASE)}\n')
files.append(archives/'SHA256SUMS.txt')
for kind in ('full','light'):
    path=archives/f'patchtst_ETTh2_ETTm2_20261013_{kind}.tar.gz'
    with tarfile.open(path,'w:gz',compresslevel=3) as tar:
        for p in sorted(files):
            if kind=='light' and p.suffix in ('.npy','.pth'):
                continue
            tar.add(p,arcname=str(Path('patchtst_repro')/p.relative_to(BASE)))
    print('ARCHIVE',path,path.stat().st_size,flush=True)
(archives/'COMPLETE.json').write_text(json.dumps({'report':str(results/'REPORT.md'),'archives':[p.name for p in archives.glob('*.tar.gz')]},indent=2))
print('REPORT_AND_ARCHIVES_COMPLETE',flush=True)
