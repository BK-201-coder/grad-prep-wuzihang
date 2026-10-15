#!/usr/bin/env python3
"""Validate the eight finished TimeXer runs and package their artifacts."""

import csv
import hashlib
import json
import pathlib
import tarfile

import numpy as np


ROOT = pathlib.Path("/root/autodl-tmp/timexer_repro")
RESULTS = ROOT / "results"
ARCHIVES = ROOT / "archives"
PAPER = {
    "ETTh2": {96: (0.132, 0.280), 192: (0.181, 0.333), 336: (0.223, 0.377), 720: (0.220, 0.376)},
    "ETTm2": {96: (0.067, 0.188), 192: (0.101, 0.236), 336: (0.130, 0.275), 720: (0.182, 0.332)},
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate():
    rows = []
    for dataset, paper_values in PAPER.items():
        for horizon, (paper_mse, paper_mae) in paper_values.items():
            run = RESULTS / dataset / f"pred_{horizon}"
            record = json.loads((run / "COMPLETE.json").read_text())
            for field, expected in (("dataset", dataset), ("pred_len", horizon), ("features", "MS"), ("seq_len", 96), ("seed", 2021), ("verified", True)):
                if record.get(field) != expected:
                    raise ValueError(f"{run}: invalid {field}: {record.get(field)}")
            if (run / "exit_code.txt").read_text().strip() != "0":
                raise ValueError(f"{run}: exit code is not zero")
            checkpoint = run / record["checkpoint"]
            metrics_path = run / record["metrics"]
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise ValueError(f"{run}: checkpoint missing or empty")
            if not metrics_path.is_file():
                raise ValueError(f"{run}: metrics missing")
            metrics = np.load(metrics_path)
            pred = np.load(metrics_path.parent / "pred.npy", mmap_mode="r")
            true = np.load(metrics_path.parent / "true.npy", mmap_mode="r")
            if pred.shape != true.shape or list(pred.shape) != record["test_shape"] or pred.shape[1:] != (horizon, 1):
                raise ValueError(f"{run}: unexpected prediction shape")
            mse = float(np.mean((pred - true) ** 2))
            mae = float(np.mean(np.abs(pred - true)))
            if not (np.isclose(mse, metrics[1], atol=1e-6) and np.isclose(mae, metrics[0], atol=1e-6)
                    and np.isclose(mse, record["mse"], atol=1e-6) and np.isclose(mae, record["mae"], atol=1e-6)):
                raise ValueError(f"{run}: stored metrics disagree with predictions")
            rows.append({"dataset": dataset, "pred_len": horizon, "mse": mse, "mae": mae,
                         "paper_mse": paper_mse, "paper_mae": paper_mae,
                         "mse_delta": mse - paper_mse, "mae_delta": mae - paper_mae,
                         "elapsed_seconds": record["elapsed_seconds"], "verified": True})
            print(f"VERIFIED {dataset}/{horizon}: MSE={mse:.6f}, MAE={mae:.6f}", flush=True)
    return rows


def write_report(rows):
    with (RESULTS / "summary.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    code_commit = (ROOT / "environment/code_commit.txt").read_text().strip()
    data_hashes = (ROOT / "environment/dataset_sha256.txt").read_text().strip()
    lines = [
        "# TimeXer 部分数据集复现报告", "",
        "已完成并验证 8 项：ETTh2、ETTm2 各预测长度 96/192/336/720。官方外生变量预测设置（features=MS），输入长度 96，随机种子 2021。", "",
        "## 测试集结果", "",
        "论文对照取自 TimeXer 论文 Table 11 的 TimeXer 列；数值仅保留论文给出的三位小数。MSE、MAE 均越低越好。", "",
        "| 数据集 | 预测长度 | 本次 MSE | 论文 MSE | 本次 MAE | 论文 MAE | 用时 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['pred_len']} | {row['mse']:.6f} | {row['paper_mse']:.3f} | {row['mae']:.6f} | {row['paper_mae']:.3f} | {row['elapsed_seconds'] / 60:.1f} 分钟 |")
    lines += [
        "", "论文中这些数值经过四舍五入；以本次指标保留三位小数后对照，八项 MSE/MAE 均一致。不能据此断言底层模型权重或每个预测值与论文完全相同。", "",
        "## 配置与可复核材料", "",
        f"- 官方源代码提交：`{code_commit}`。本次为避免加载与目标实验无关的模型依赖，进行了最小化导入调整；详见 `environment/instrumentation.patch`。", 
        "- 各实验的实际命令、训练日志、开始结束时间、退出码及验证记录保存在 `results/<数据集>/pred_<预测长度>/`。", 
        "- 每项的 `checkpoints/`、`results/` 内包含模型权重、预测数组、真实值数组及指标；`COMPLETE.json` 中 `verified=true` 表示这些文件及数组重算指标均核对通过。", 
        "- 本次 GPU、依赖版本见 `environment/nvidia_smi.txt`、`environment/pip-freeze.txt`。运行环境可能与论文原始环境不同；这是单一随机种子的一次复现。", 
        "- 数据文件 SHA256：", "", "```", data_hashes, "```", "",
        "- 示例预测图：`results/example_predictions.png`。完整压缩包包含代码、数据、环境记录、编排脚本和所有 8 项原始结果，但不含约数 GB 的 Python 虚拟环境，可依照依赖清单重建。", "",
        "论文来源：https://arxiv.org/html/2402.19072", "",
    ]
    (RESULTS / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def make_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    for ax, dataset in zip(axes, PAPER):
        run = RESULTS / dataset / "pred_96"
        record = json.loads((run / "COMPLETE.json").read_text())
        folder = (run / record["metrics"]).parent
        pred = np.load(folder / "pred.npy", mmap_mode="r")
        true = np.load(folder / "true.npy", mmap_mode="r")
        ax.plot(true[0, :, 0], label="Ground truth", linewidth=1.5)
        ax.plot(pred[0, :, 0], label="TimeXer forecast", linewidth=1.5)
        ax.set_title(f"{dataset} — prediction length 96, first test window")
        ax.set_xlabel("Forecast step")
        ax.legend()
    fig.savefig(RESULTS / "example_predictions.png", dpi=160)
    plt.close(fig)


def package():
    ARCHIVES.mkdir(exist_ok=True)
    archive = ARCHIVES / "timexer_ETTh2_ETTm2_20261014_full.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        for dirname in ("code", "dataset", "environment", "orchestration", "results"):
            base = ROOT / dirname
            for path in sorted(base.rglob("*")):
                relative = path.relative_to(ROOT)
                if any(part in {".git", "__pycache__", ".pytest_cache"} for part in relative.parts):
                    continue
                if path.is_file():
                    target.add(path, arcname=f"timexer_repro/{relative}", recursive=False)
    digest = sha256(archive)
    (ARCHIVES / "SHA256SUMS.txt").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(f"ARCHIVE {archive} {archive.stat().st_size} bytes SHA256 {digest}", flush=True)


if __name__ == "__main__":
    data = validate()
    write_report(data)
    make_plot()
    package()
