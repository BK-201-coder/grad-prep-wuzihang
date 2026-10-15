import json
import pathlib
import sys
from datetime import datetime

import numpy as np


run = pathlib.Path(sys.argv[1])
dataset = sys.argv[2]
pred_len = int(sys.argv[3])
metrics_files = list((run / "results").glob("*/metrics.npy"))
checkpoints = list((run / "checkpoints").glob("*/checkpoint.pth"))
if len(metrics_files) != 1 or len(checkpoints) != 1:
    raise RuntimeError("Expected exactly one metrics file and one checkpoint")
results_dir = metrics_files[0].parent
metrics = np.load(metrics_files[0])
pred = np.load(results_dir / "pred.npy")
true = np.load(results_dir / "true.npy")
if pred.shape != true.shape or pred.ndim != 3 or pred.shape[1] != pred_len or pred.shape[2] != 1:
    raise RuntimeError(f"Unexpected forecast shape: {pred.shape}, {true.shape}")
mae = float(np.mean(np.abs(pred - true)))
mse = float(np.mean((pred - true) ** 2))
if not np.isclose(metrics[0], mae, atol=1e-6) or not np.isclose(metrics[1], mse, atol=1e-6):
    raise RuntimeError("Saved metrics disagree with prediction arrays")
start = datetime.fromisoformat((run / "started_at.txt").read_text().strip())
end = datetime.fromisoformat((run / "finished_at.txt").read_text().strip())
summary = {
    "dataset": dataset,
    "pred_len": pred_len,
    "features": "MS",
    "seq_len": 96,
    "seed": 2021,
    "mse": mse,
    "mae": mae,
    "test_shape": list(pred.shape),
    "elapsed_seconds": (end - start).total_seconds(),
    "verified": True,
    "checkpoint": str(checkpoints[0].relative_to(run)),
    "metrics": str(metrics_files[0].relative_to(run)),
}
(run / "COMPLETE.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False))
