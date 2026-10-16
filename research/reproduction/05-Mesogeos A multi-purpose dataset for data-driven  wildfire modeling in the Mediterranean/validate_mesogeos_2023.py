#!/usr/bin/env python3
"""Validate the user-supplied 2023 Mesogeos Track A CSVs before training."""

import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd


root = pathlib.Path(sys.argv[1])
expected_sha256 = {
    "positives.csv": "c44554e29f0d676ad13d42950500fb5bc6361c5dc5afe77958c59b7cc3c8ed0b",
    "negatives.csv": "ce5e6fd204d4c26189fe45c2a2d709ccfdead6fa08c0fcbeb5ce5c59d654820b",
}
paper_positive = {"train": 6451, "val": 754, "test": 1369}
paper_negative = {"train": 12902, "val": 1508, "test": 2738}
required_columns = {
    "time", "sample", "burned_area_has", "d2m", "lai", "lst_day", "lst_night", "ndvi", "rh", "smi", "sp", "ssrd", "t2m", "tp", "wind_speed", "dem", "roads_distance", "slope", "lc_agriculture", "lc_forest", "lc_grassland", "lc_settlement", "lc_shrubland", "lc_sparse_vegetation", "lc_water_bodies", "lc_wetland", "population"
}
report = {"source": "Official Mesogeos Google Drive / ml_tracks / a.danger_forecasting", "files": {}}
for filename in ("positives.csv", "negatives.csv"):
    path = root / filename
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256[filename]:
        raise AssertionError(f"{filename}: remote SHA256 differs from local source")
    header = set(pd.read_csv(path, nrows=0).columns)
    if not required_columns <= header:
        raise AssertionError(f"{filename}: missing columns {sorted(required_columns - header)}")
    frame = pd.read_csv(path, usecols=["time", "sample", "burned_area_has"])
    if len(frame) % 30:
        raise AssertionError(f"{filename}: row count is not divisible by 30")
    sample = frame["sample"].to_numpy().reshape(-1, 30)
    if not np.all(sample == sample[:, :1]):
        raise AssertionError(f"{filename}: sample ID changes inside a 30-day block")
    last_year = frame["time"].to_numpy().reshape(-1, 30)[:, -1].astype(str)
    years = np.array([int(value[:4]) for value in last_year])
    counts = {
        "train": int((years <= 2019).sum()),
        "val": int((years == 2020).sum()),
        "test": int(np.isin(years, [2021, 2022]).sum()),
    }
    if sum(counts.values()) != len(sample):
        raise AssertionError(f"{filename}: unexpected years outside paper splits")
    unique_ids = {split: int(len(np.unique(sample[mask, 0]))) for split, mask in {
        "train": years <= 2019,
        "val": years == 2020,
        "test": np.isin(years, [2021, 2022]),
    }.items()}
    entry = {"bytes": path.stat().st_size, "sha256": digest.hexdigest(), "rows": len(frame), "samples": len(sample), "year_split_samples": counts, "unique_sample_ids": unique_ids}
    report["files"][filename] = entry
    if filename == "positives.csv" and counts != paper_positive:
        raise AssertionError(f"Positive split differs from paper: {counts}")
    if filename == "negatives.csv":
        for split, needed in paper_negative.items():
            if counts[split] < needed or unique_ids[split] < needed:
                raise AssertionError(f"Insufficient negative candidates in {split}: {counts[split]} samples, {unique_ids[split]} IDs, need {needed}")
    print(f"VERIFIED {filename} {entry}", flush=True)

report["matches_paper_positive_splits_and_negative_pool"] = True
output = root.parent / "results" / "data_validation_2023.json"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(f"VALIDATION_PASSED {output}", flush=True)
