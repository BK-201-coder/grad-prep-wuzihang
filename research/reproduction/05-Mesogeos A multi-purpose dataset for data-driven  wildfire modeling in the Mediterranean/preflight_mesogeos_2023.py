#!/usr/bin/env python3
"""Read-only official-loader smoke test for the verified 2023 Track A data."""

import json
import pathlib
import sys

import numpy as np


root = pathlib.Path('/root/autodl-tmp/mesogeos_repro')
sys.path.insert(0, str(root / 'code/ml_tracks/a.fire_danger'))
from datasets.dataset import FireDataset

config = json.loads((root / 'results/config_train.json').read_text())
args = config['dataset']['args']
features = config['features']
np.random.seed(config['seed'])
for split, expected in [('train', 19353), ('val', 2262), ('test', 4107)]:
    dataset = FireDataset(**args, dynamic_features=features['dynamic'], static_features=features['static'], train_val_test=split)
    if len(dataset) != expected:
        raise AssertionError(f'{split}: expected {expected} sampled examples, got {len(dataset)}')
    dynamic, static, burned, label = dataset[0]
    if dynamic.shape != (30, len(features['dynamic'])) or static.shape != (len(features['static']),):
        raise AssertionError(f'{split}: unexpected feature shapes {dynamic.shape}, {static.shape}')
    print(f'PREFLIGHT_OK {split} examples={len(dataset)} dynamic_shape={dynamic.shape} static_shape={static.shape} first_label={label}', flush=True)
