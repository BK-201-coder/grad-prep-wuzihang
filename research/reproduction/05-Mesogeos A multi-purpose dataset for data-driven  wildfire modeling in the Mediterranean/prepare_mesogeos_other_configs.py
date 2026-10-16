#!/usr/bin/env python3
"""Stage official Track A configs for the remaining two models."""

import json
import pathlib

root = pathlib.Path('/root/autodl-tmp/mesogeos_repro')
configs = root / 'code/ml_tracks/a.fire_danger/configs'
validation = json.loads((root / 'results/data_validation_2023.json').read_text())
assert validation['matches_paper_positive_splits_and_negative_pool'] is True

for model in ('transformer', 'gtn'):
    result = root / 'results' / model
    result.mkdir(parents=True, exist_ok=True)
    destination = result / 'config_train.json'
    if destination.exists():
        raise SystemExit(f'Refusing to replace {destination}')
    config = json.loads((configs / f'config_{model}' / 'config_train.json').read_text())
    assert config['model_type'] == model
    assert config['trainer']['epochs'] == 30 and config['seed'] == 12345
    config['name'] = f'mesogeos_trackA_{model}_seed12345'
    config['dataset']['args']['dataset_root'] = str(root / 'data_2023')
    config['trainer']['save_dir'] = str(result)
    if model == 'gtn':
        # Official config selects GPU 1, but this rented instance has only GPU 0.
        config['gpu_id'] = 0
    destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    print(destination)
