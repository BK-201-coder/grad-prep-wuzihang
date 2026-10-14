import json
import sys
import time
from argparse import Namespace
from patchtst_config import BASE, CODE, config
sys.path.insert(0, str(CODE))
import numpy as np
import torch
import pandas as pd
from data_provider.data_factory import data_provider
from models.PatchTST import Model

torch.set_num_threads(4)
report = {}
for dataset in ('ETTh2', 'ETTm2'):
    raw = pd.read_csv(BASE / 'data' / (dataset + '.csv'))
    assert raw.shape == ((17420 if dataset == 'ETTh2' else 69680), 8), raw.shape
    assert not raw.isna().any().any()
    assert np.isfinite(raw.iloc[:, 1:].values).all()
    dates = pd.to_datetime(raw.date)
    assert dates.is_monotonic_increasing and not dates.duplicated().any()
    expected_delta = pd.Timedelta(hours=1) if dataset == 'ETTh2' else pd.Timedelta(minutes=15)
    assert (dates.diff().iloc[1:] == expected_delta).all()
    report[dataset] = {}
    for horizon in (96, 192, 336, 720):
        args = Namespace(**config(dataset, horizon))
        torch.manual_seed(2021)
        train_data, train_loader = data_provider(args, 'train')
        val_data, val_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')
        model = Model(args).cuda().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        steps = 0
        for i, (x, y, _, _) in enumerate(train_loader):
            if i == 3:
                torch.cuda.synchronize()
                started = time.monotonic()
            x, y = x.float().cuda(), y[:, -horizon:, :].float().cuda()
            optimizer.zero_grad()
            output = model(x)
            assert output.shape == y.shape == (128, horizon, 7)
            loss = (output-y).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            if i >= 3:
                steps += 1
            if i == 12:
                break
        torch.cuda.synchronize()
        train_step = (time.monotonic()-started)/steps
        model.eval()
        started = time.monotonic()
        val_steps = 0
        with torch.no_grad():
            for i, (x, y, _, _) in enumerate(val_loader):
                output = model(x.float().cuda())
                value = (output-y[:, -horizon:, :].float().cuda()).square().mean().item()
                assert np.isfinite(value)
                val_steps += 1
                if i == 4:
                    break
        torch.cuda.synchronize()
        val_step = (time.monotonic()-started)/val_steps
        epoch_seconds = train_step*len(train_loader)+val_step*(len(val_loader)+len(test_loader))
        report[dataset][str(horizon)] = dict(train_samples=len(train_data),
            val_samples=len(val_data),test_samples=len(test_data),
            evaluated_test_samples=len(test_loader)*128, train_steps=len(train_loader),
            train_step_seconds=train_step, val_step_seconds=val_step,
            estimated_epoch_seconds=epoch_seconds,
            estimated_100_epochs_minutes=epoch_seconds*100/60,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        print('BENCHMARK_OK',dataset,horizon,json.dumps(report[dataset][str(horizon)]),flush=True)
        del optimizer,model,train_loader,val_loader,test_loader
        torch.cuda.empty_cache()
    (BASE/'environment/benchmark.json').write_text(json.dumps(report,indent=2))
print('ALL_BENCHMARKS_OK', flush=True)
