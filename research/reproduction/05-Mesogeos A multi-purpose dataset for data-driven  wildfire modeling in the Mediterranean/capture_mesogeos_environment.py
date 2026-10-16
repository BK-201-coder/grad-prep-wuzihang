#!/usr/bin/env python3
"""Save reproducibility details for the completed Mesogeos run."""

import json
import pathlib
import platform
import subprocess
import sys

import numpy
import pandas
import sklearn
import torch


root = pathlib.Path('/root/autodl-tmp/mesogeos_repro')
results = root / 'results'
def command(*args):
    return subprocess.check_output(args, text=True).strip()

environment = {
    'python': sys.version,
    'platform': platform.platform(),
    'torch': torch.__version__,
    'torch_cuda_runtime': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'numpy': numpy.__version__,
    'pandas': pandas.__version__,
    'scikit_learn': sklearn.__version__,
    'git_commit': command('git', '-C', str(root / 'code'), 'rev-parse', 'HEAD'),
    'git_status_porcelain': command('git', '-C', str(root / 'code'), 'status', '--porcelain'),
    'gpu_driver': command('nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'),
}
(results / 'environment.json').write_text(json.dumps(environment, ensure_ascii=False, indent=2) + '\n')
(results / 'pip_freeze.txt').write_text(command(sys.executable, '-m', 'pip', 'freeze') + '\n')
print(json.dumps(environment, ensure_ascii=False, indent=2))
