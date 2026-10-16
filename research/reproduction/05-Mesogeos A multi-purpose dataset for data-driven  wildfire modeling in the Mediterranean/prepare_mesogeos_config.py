#!/usr/bin/env python3
"""Copy the official Track A LSTM settings, changing only file locations/name."""

import json
import pathlib


root = pathlib.Path("/root/autodl-tmp/mesogeos_repro")
official = root / "code/ml_tracks/a.fire_danger/configs/config_lstm/config_train.json"
results = root / "results"
results.mkdir(exist_ok=True)
config = json.loads(official.read_text())
config["name"] = "mesogeos_trackA_lstm_seed12345"
config["dataset"]["args"]["dataset_root"] = str(root / "data_2023")
config["trainer"]["save_dir"] = str(results)
destination = results / "config_train.json"
destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
print(f"Prepared {destination}")
