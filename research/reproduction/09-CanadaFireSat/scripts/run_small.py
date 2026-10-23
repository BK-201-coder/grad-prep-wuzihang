"""Small reproduction using official architecture, transforms, adaptation and loss."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'cache/huggingface'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'cache/torch'))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
import sys
sys.path.insert(0, str(ROOT / 'official'))
import copy
import csv
import json
import math
import random
import subprocess
import time
from collections import Counter
import numpy as np
import torch
import yaml
import pyarrow.parquet as pq
from datasets import Dataset as HfDataset
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from matplotlib import pyplot as plt
from src.constants import BANDS_10, BANDS_20, BANDS_60, HF_MIX_TAB_COLUMNS, TAB_SOURCE_COLS
from src.data.hf_Canada.hf_mix_dataloader import SatImTabDataset
from src.data.Canada.data_transforms import Canada_segmentation_transform, TabCanada_segmentation_transform
from src.models.resnet import TabResNetConvLSTM
from deepsat.metrics.loss_functions import get_loss
from deepsat.utils.lr_scheduler import build_scheduler_pytorch

OUT = ROOT / 'outputs'
for name in ['figures', 'metrics', 'configs', 'reports', 'checkpoints']:
    (OUT / name).mkdir(parents=True, exist_ok=True)

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str))

class SmallDataset(SatImTabDataset):
    """Reuse official __getitem__/__fastadapt__, read only required Parquet columns.

    Retain fwi explicitly: official __fastadapt__ accesses it although the default
    HF_MIX_TAB_COLUMNS list omits it. No predictive feature is added.
    """
    def __init__(self, files, cfg, training):
        names = pq.ParquetFile(files[0]).schema_arrow.names
        columns = list(dict.fromkeys(HF_MIX_TAB_COLUMNS + ['fwi', 'date']))
        assert set(columns).issubset(names), (set(columns) - set(names))
        self.hf_dataset = HfDataset.from_parquet([str(f) for f in files], columns=columns,
                                                cache_dir=str(ROOT / 'cache/arrow'))
        self.hf_dataset = self.hf_dataset.with_format('numpy', columns=columns)
        self.meta_hf = json.loads((ROOT / 'data/metadata/env_cols.json').read_text())
        self.transform = Canada_segmentation_transform(cfg, ROOT / 'data/metadata/bands_mean.json',
                         ROOT / 'data/metadata/bands_std.json', is_training=training,
                         is_eval=not training, with_doy=True, with_loc=False, img_only=True,
                         eval_sampling='start')
        self.tab_transform = TabCanada_segmentation_transform(cfg, ROOT / 'data/metadata/TAB_ENV',
                                                              with_doy=True, is_training=training)
        self.mix_transform = None
        self.return_paths = False
        self.with_loc = False
        self.bands = BANDS_10 + BANDS_20 + BANDS_60
        self.band_10x = self.band_20x = self.band_60x = True
        self.tab_source_cols = TAB_SOURCE_COLS
        self.nan_value = 'median'
        self.is_spatial = False
        self.one_tile = False

def choose(dataset, split, maximum):
    meta = dataset.hf_dataset.select_columns(['region', 'tile_id', 'file_id', 'date']).with_format(None)
    rng = np.random.default_rng(42)
    records = []
    for i, item in enumerate(meta):
        year = int(str(item['date'])[:4])
        assert (2016 <= year <= 2021) if split == 'train' else year == 2022, (split, item)
        records.append({'row': i, **item, 'year': year})
    # Random subset preserves downloaded subset prevalence, without a 50/50 rebalance.
    indices = sorted(rng.choice(len(records), min(maximum, len(records)), replace=False).tolist())
    selected = [records[i] for i in indices]
    counts = Counter(r['file_id'] for r in selected)
    assert counts['POS'] > 0 and counts['NEG'] > 0, f'Need both classes: {counts}'
    write_json(OUT / f'configs/{split}_samples.json', selected)
    return indices, {'samples': len(indices), 'classes': dict(counts),
                     'years': dict(Counter(r['year'] for r in selected))}

def metrics(labels, scores):
    labels = np.concatenate(labels).ravel().astype(np.uint8)
    scores = np.concatenate(scores).ravel()
    predicted = scores >= 0.5
    assert np.isfinite(scores).all()
    return {'AP': float(average_precision_score(labels, scores)),
            'fire_F1_at_0.5': float(f1_score(labels, predicted, zero_division=0)),
            'fire_precision_at_0.5': float(precision_score(labels, predicted, zero_division=0)),
            'fire_recall_at_0.5': float(recall_score(labels, predicted, zero_division=0)),
            'positive_cell_fraction': float(labels.mean()), 'cells': len(labels)}

def save_figure(dataset, index, probability, target, name):
    raw = dataset.hf_dataset[index]
    rgb = np.asarray(raw['10x'][-1, :3]).transpose(1, 2, 0).astype(float)
    lo, hi = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3))
    axes[0].imshow(rgb)
    axes[0].set_title('Last historical RGB (full tile)')
    axes[1].imshow(target, vmin=0, vmax=1, cmap='gray')
    axes[1].set_title('Target (center crop / 100m)')
    axes[2].imshow(probability, vmin=0, vmax=1, cmap='inferno')
    axes[2].set_title('Predicted fire probability')
    axes[3].imshow(probability >= 0.5, vmin=0, vmax=1, cmap='gray')
    axes[3].set_title('Prediction at 0.5')
    for ax in axes:
        ax.axis('off')
    fig.suptitle(f"{raw['region']} / {raw['tile_id']} / {raw['file_id']} / {raw['date']}")
    fig.tight_layout()
    fig.savefig(OUT / f'figures/{name}.png', dpi=140)
    plt.close(fig)

@torch.no_grad()
def evaluate(model, loader, dataset, indices, tag, save_plots=False):
    model.eval()
    model.sat_model.out_H = model.sat_model.out_W = 26
    targets, predictions = [], []
    position = 0
    illustrated = Counter()
    for image, tab in loader:
        logits = model(image['inputs'].cuda(), tab['tab_inputs'].cuda(), tab['mask'].cuda())
        probability = logits.softmax(1)[:, 1].cpu().numpy()
        target = image['labels'][..., 0].numpy()
        targets.append(target)
        predictions.append(probability)
        if save_plots:
            for k in range(len(target)):
                cls = 'positive' if target[k].any() else 'negative'
                if illustrated[cls] < 3:
                    save_figure(dataset, indices[position + k], probability[k], target[k],
                                f'{tag}_{cls}_{illustrated[cls]}')
                    illustrated[cls] += 1
        position += len(target)
    result = metrics(targets, predictions)
    write_json(OUT / f'metrics/{tag}.json', result)
    return result

def main():
    started = time.time()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    config = yaml.safe_load((ROOT / 'official/configs/ResNet_MULTI.yaml').read_text())
    # PyYAML treats YAML scalars such as 5e-6 as strings, unlike Hydra/OmegaConf.
    for key in ['lr_base', 'lr_min', 'lr_start']:
        config['SOLVER'][key] = float(config['SOLVER'][key])
    train_cfg = copy.deepcopy(config['MODEL'])
    train_cfg['pretrained'] = False
    eval_cfg = copy.deepcopy(train_cfg)
    eval_cfg.update(img_res=260, out_H=26, out_W=26, test_max_seq_len=5)
    train_files = sorted((ROOT / 'data/Alberta').glob('train-*.parquet'))
    val_files = sorted((ROOT / 'data/Alberta').glob('validation-*.parquet'))
    train_dataset = SmallDataset(train_files, train_cfg, True)
    val_dataset = SmallDataset(val_files, eval_cfg, False)
    train_ids, train_info = choose(train_dataset, 'train', 512)
    val_ids, val_info = choose(val_dataset, 'validation', 128)
    protocol = {'train': train_info, 'validation': val_info, 'batch_size': 4, 'epochs': 20,
                'training': 'ImageNet initialized; not fine-tuned from official wildfire checkpoint',
                'evaluation': 'Alberta validation subset; center crop 260 -> 26; latest 5 images; threshold 0.5',
                'loss': 'official combined_dice_ce', 'seed': 42, 'precision': 'float32',
                'normalization': 'author supplied; fitting population not independently verified',
                'metric': 'pixel-pooled sklearn AP and fire F1; no national-table equivalence claimed',
                'loader_adaptation': 'retain fwi and date; read only required columns; reuse official adapters',
                'code_commit': json.loads((ROOT / 'configs/provenance.json').read_text())['official_commit'],
                'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()}
    write_json(OUT / 'configs/protocol.json', protocol)
    print('DATA_AUDIT', json.dumps(protocol), flush=True)
    train_loader = DataLoader(Subset(train_dataset, train_ids), batch_size=4, shuffle=True,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(Subset(val_dataset, val_ids), batch_size=4, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)
    baseline_path = OUT / 'metrics/official_inference.json'
    if os.environ.get('REUSE_OFFICIAL_INFERENCE') == '1' and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        assert baseline['cells'] == len(val_ids) * 26 * 26
        print('STEP1_REUSED_SAME_SEEDED_SUBSET', json.dumps(baseline), flush=True)
    else:
        official = TabResNetConvLSTM(**train_cfg)
        checkpoint_path = next((ROOT / 'checkpoints/official').glob('*.ckpt'))
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state = checkpoint.get('state_dict', checkpoint)
        if all(key.startswith('model.') for key in state):
            state = {key[6:]: value for key, value in state.items()}
        official.load_state_dict(state, strict=True)
        official.cuda()
        baseline = evaluate(official, val_loader, val_dataset, val_ids, 'official_inference', True)
        print('STEP1_COMPLETE', json.dumps(baseline), flush=True)
        del checkpoint, state, official
        torch.cuda.empty_cache()
    # Preserve author's ImageNet initialization for independent small retraining.
    train_cfg['pretrained'] = True
    model = TabResNetConvLSTM(**train_cfg).cuda()
    criterion = get_loss(config, 'cuda', reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
    scheduler = build_scheduler_pytorch(config, optimizer, len(train_loader), 'epoch')
    history = []
    best = -1.0
    for epoch in range(20):
        lr = optimizer.param_groups[0]['lr']
        model.train()
        model.sat_model.out_H = model.sat_model.out_W = 24
        epoch_started = time.time()
        losses = []
        for step, (image, tab) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            logits = model(image['inputs'].cuda(non_blocking=True),
                           tab['tab_inputs'].cuda(non_blocking=True), tab['mask'].cuda(non_blocking=True))
            target = image['labels'][..., 0].long().cuda(non_blocking=True)
            loss = criterion(logits.permute(0, 2, 3, 1).reshape(-1, 2), target.reshape(-1))
            assert torch.isfinite(loss), 'Nonfinite training loss'
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step == 0 or (step + 1) % 25 == 0:
                print(f'TRAIN epoch={epoch+1} step={step+1}/{len(train_loader)} loss={loss.item():.5f}', flush=True)
        validation = evaluate(model, val_loader, val_dataset, val_ids, 'latest_validation')
        row = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'lr': lr,
               'seconds': round(time.time() - epoch_started, 2),
               'peak_gpu_GB': torch.cuda.max_memory_allocated() / 1e9, **validation}
        history.append(row)
        with (OUT / 'metrics/history.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        if validation['AP'] > best:
            best = validation['AP']
            torch.save({'state_dict': model.state_dict(), 'model_config': train_cfg,
                        'epoch': epoch + 1, 'validation': validation}, OUT / 'checkpoints/best.pt')
        torch.save({'state_dict': model.state_dict(), 'model_config': train_cfg,
                    'epoch': epoch + 1, 'validation': validation}, OUT / 'checkpoints/last.pt')
        print('EPOCH_COMPLETE', json.dumps(row), flush=True)
        scheduler.step()
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        axes[0].plot([h['epoch'] for h in history], [h['train_loss'] for h in history])
        axes[0].set_title('Training loss')
        for key in ['AP', 'fire_F1_at_0.5']:
            axes[1].plot([h['epoch'] for h in history], [h[key] for h in history], label=key)
        axes[1].legend()
        axes[1].set_title('Validation (small subset)')
        fig.tight_layout()
        fig.savefig(OUT / 'figures/training_curves.png', dpi=140)
        plt.close(fig)
    best_checkpoint = torch.load(OUT / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(best_checkpoint['state_dict'], strict=True)
    result = evaluate(model, val_loader, val_dataset, val_ids, 'retrained_best', True)
    write_json(OUT / 'metrics/completion.json', {'elapsed_seconds': time.time()-started,
               'best_epoch': best_checkpoint['epoch'], 'official': baseline, 'retrained': result,
               'scope': 'small Alberta validation subset, not national benchmark'})
    print('STEP2_COMPLETE', json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
