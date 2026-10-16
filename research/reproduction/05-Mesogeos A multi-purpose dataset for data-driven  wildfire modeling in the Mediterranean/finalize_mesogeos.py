#!/usr/bin/env python3
"""Verify the completed official Mesogeos LSTM run and save a manifest."""

import ast
import hashlib
import json
import math
import pathlib
import re
import subprocess
import sys

import torch


root = pathlib.Path('/root/autodl-tmp/mesogeos_repro')
results = root / 'results'
sys.path.insert(0, str(root / 'code/ml_tracks/a.fire_danger'))
assert (results / 'exit_code.txt').read_text().strip() == '0'
validation = json.loads((results / 'data_validation_2023.json').read_text())
assert validation['matches_paper_positive_splits_and_negative_pool'] is True
run_text = (results / 'run.log').read_text()
epochs = [int(number) for number in re.findall(r'^\s+epoch\s+:\s+(\d+)$', run_text, re.M)]
assert epochs == list(range(1, 31)), f'Incomplete epoch sequence: {epochs}'
metric_lines = [line for line in run_text.splitlines() if line.startswith("{'loss':")]
assert len(metric_lines) == 1, f'Expected one official test metric line, got {len(metric_lines)}'
metrics = ast.literal_eval(metric_lines[0])
for key in ('loss', 'accuracy', 'precision', 'recall', 'f1_score', 'aucpr'):
    assert key in metrics and math.isfinite(float(metrics[key]))
    if key != 'loss':
        assert 0 <= float(metrics[key]) <= 1

best = pathlib.Path((results / 'model_path.txt').read_text().strip())
assert best.is_file() and best.name == 'model_best.pth'
model_dir = best.parent
checkpoints = sorted(model_dir.glob('checkpoint-epoch*.pth'))
assert len(checkpoints) == 30
checkpoint = torch.load(best, map_location='cpu')
best_epoch = int(checkpoint['epoch'])
assert 1 <= best_epoch <= 30
assert checkpoint['state_dict']
for value in checkpoint['state_dict'].values():
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

config = json.loads((results / 'config_train.json').read_text())
assert config['dataset']['args']['dataset_root'] == str(root / 'data_2023')
assert config['trainer']['epochs'] == 30 and config['seed'] == 12345
test_config = json.loads((results / 'config_test.json').read_text())
assert test_config['model_path'] == str(best)
commit = subprocess.check_output(['git', '-C', str(root / 'code'), 'rev-parse', 'HEAD'], text=True).strip()
paper = {'precision': 0.763, 'recall': 0.812, 'f1_score': 0.786, 'aucpr': 0.853}
manifest = {
    'verified': True,
    'scope': 'Mesogeos 2023 Track A official LSTM, one 30-epoch run',
    'code_commit': commit,
    'seed': config['seed'],
    'best_epoch': best_epoch,
    'epochs_completed': len(epochs),
    'checkpoints_saved': len(checkpoints),
    'test_samples': 4107,
    'test_metrics': metrics,
    'paper_table_1_lstm': paper,
    'difference_reproduction_minus_paper': {key: round(float(metrics[key]) - value, 6) for key, value in paper.items()},
    'best_checkpoint': str(best),
    'best_checkpoint_sha256': sha256(best),
    'config_sha256': sha256(results / 'config_train.json'),
    'original_data': validation['files'],
    'note': 'Official test.py does not seed NumPy before sampling the negative candidate pool; repeated test runs can vary slightly.',
}
(results / 'COMPLETE.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
