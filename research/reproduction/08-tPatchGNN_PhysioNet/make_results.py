"""Build portable figures and tables from the saved t-PatchGNN console logs."""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "reproduction_logs"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

PAPER = {
    "complete": {"mse": 0.00498, "mse_std": 0.00008, "mae": 0.0372, "mae_std": 0.0003},
    "wo_transformer": {"mse": 0.00510, "mse_std": 0.00012, "mae": 0.0383, "mae_std": 0.0007},
}


def parse_run(path: Path) -> list[dict]:
    rows = []
    current = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"- Epoch (\d+), ExpID (\d+)", line)
        if match:
            current = {"epoch": int(match[1]), "experiment_id": int(match[2])}
            rows.append(current)
            continue
        if current is None:
            continue
        match = re.match(r"Train - Loss \(one batch\): ([0-9.]+)", line)
        if match:
            current["train_batch_loss"] = float(match[1])
        match = re.match(
            r"Val - Loss, MSE, RMSE, MAE, MAPE: ([0-9.]+), ([0-9.]+), ([0-9.]+), ([0-9.]+)", line
        )
        if match:
            current["val_mse"] = float(match[2])
            current["val_mae"] = float(match[4])
        match = re.match(
            r"Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: (\d+), ([0-9.]+), ([0-9.]+), ([0-9.]+), ([0-9.]+)",
            line,
        )
        if match:
            current["best_epoch_so_far"] = int(match[1])
            current["best_test_mse_so_far"] = float(match[3])
            current["best_test_mae_so_far"] = float(match[5])
        match = re.match(r"Time spent: ([0-9.]+)s", line)
        if match:
            current["epoch_seconds"] = float(match[1])
    if not rows or any("epoch_seconds" not in row for row in rows):
        raise ValueError(f"Incomplete training log: {path}")
    return rows


run_paths = {
    "A_smoke": RAW / "A_smoke.console.log",
    "seed1": RAW / "B_physionet_seed1.console.log",
    "seed2": RAW / "seed2.console.log",
    "seed3": RAW / "seed3.console.log",
    "wo_transformer_seed1": RAW / "wo_transformer_seed1.console.log",
}
runs = {name: parse_run(path) for name, path in run_paths.items() if path.exists()}
if "A_smoke" not in runs or "seed1" not in runs:
    raise ValueError("A and B logs are required")

with (ROOT / "epoch_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
    fields = [
        "run", "epoch", "experiment_id", "train_batch_loss", "val_mse", "val_mae",
        "best_epoch_so_far", "best_test_mse_so_far", "best_test_mae_so_far", "epoch_seconds",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for name, rows in runs.items():
        writer.writerows({"run": name, **row} for row in rows)

summaries = {}
for name, rows in runs.items():
    last = rows[-1]
    summaries[name] = {
        "epochs": len(rows),
        "best_epoch_zero_based": last["best_epoch_so_far"],
        "test_mse": last["best_test_mse_so_far"],
        "test_mae": last["best_test_mae_so_far"],
        "sum_epoch_seconds": round(sum(row["epoch_seconds"] for row in rows), 2),
    }

seeds = [summaries[f"seed{i}"] for i in (1, 2, 3) if f"seed{i}" in summaries]
seed_stats = {}
for metric in ("test_mse", "test_mae"):
    values = [item[metric] for item in seeds]
    seed_stats[metric] = {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "n": len(values),
    }
summary = {"runs": summaries, "complete_seed_statistics": seed_stats, "paper": PAPER}
(ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(9, 5))
for name in ("seed1", "seed2", "seed3"):
    if name in runs:
        rows = runs[name]
        ax.plot([r["epoch"] for r in rows], [r["val_mse"] * 1000 for r in rows], label=name)
ax.set(xlabel="Epoch (zero based)", ylabel="Validation MSE (x 1e-3)", title="PhysioNet validation curves")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "validation_curves.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, metric, scale in ((axes[0], "mse", 1000), (axes[1], "mae", 100)):
    paper = PAPER["complete"][metric] * scale
    ours = seed_stats[f"test_{metric}"]["mean"] * scale
    err = seed_stats[f"test_{metric}"]["sample_std"]
    ax.bar(["Paper: 5 seeds", f"Ours: {len(seeds)} seed(s)"], [paper, ours],
           yerr=[PAPER["complete"][metric + "_std"] * scale, (err or 0) * scale],
           capsize=5, color=["#5279a7", "#58a078"])
    ax.set_ylabel(f"{metric.upper()} (x {scale:g})")
    ax.set_ylim(0, max(paper, ours) * 1.25)
    for i, value in enumerate((paper, ours)):
        ax.text(i, value + max(paper, ours) * 0.02, f"{value:.3f}", ha="center")
fig.suptitle("PhysioNet: complete model versus paper")
fig.tight_layout()
fig.savefig(FIGURES / "paper_comparison.png", dpi=180)
plt.close(fig)

if "wo_transformer_seed1" in summaries:
    ab = summaries["wo_transformer_seed1"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, metric, scale in ((axes[0], "mse", 1000), (axes[1], "mae", 100)):
        values = [summaries["seed1"][f"test_{metric}"] * scale, ab[f"test_{metric}"] * scale]
        ax.bar(["Complete seed 1", "No Transformer seed 1"], values,
               color=["#5279a7", "#ca765a"])
        ax.set_ylabel(f"{metric.upper()} (x {scale:g})")
        ax.set_ylim(0, max(values) * 1.25)
        for i, value in enumerate(values):
            ax.text(i, value + max(values) * 0.02, f"{value:.3f}", ha="center")
    fig.suptitle("PhysioNet: single-seed Transformer ablation")
    fig.tight_layout()
    fig.savefig(FIGURES / "ablation_comparison.png", dpi=180)
    plt.close(fig)

with (ROOT / "run_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
    writer = csv.writer(stream)
    writer.writerow(["run", "epochs", "best_epoch_zero_based", "test_mse", "test_mae", "sum_epoch_seconds"])
    for name, item in summaries.items():
        writer.writerow([name, item["epochs"], item["best_epoch_zero_based"],
                         item["test_mse"], item["test_mae"], item["sum_epoch_seconds"]])

lines = [
    "# t-PatchGNN PhysioNet 复现结果",
    "",
    "论文：[ICML 2024 原文](https://proceedings.mlr.press/v235/zhang24bw.html)；代码：[官方仓库](https://github.com/usail-hkust/t-PatchGNN)。",
    "",
    "数据为官方处理后的 PhysioNet，共 12000 条记录（训练/验证/测试为 7200/2400/2400）。使用单张 NVIDIA RTX 4090 D；服务器环境为 Python 3.10.8、PyTorch 2.1.2+cu118，和官方要求文件中的版本略有差异。官方 PhysioNet 参数：history=24、patch_size=stride=8、batch_size=32、lr=0.001、patience=10。",
    "",
    "## A：流程检查",
    "",
    f"训练 1 轮，测试 MSE={summaries['A_smoke']['test_mse']:.5f}，MAE={summaries['A_smoke']['test_mae']:.5f}。此结果只说明训练、验证、测试流程已跑通。",
    "",
    "## B：正式训练和随机种子",
    "",
    "| 运行 | 轮数 | 最佳轮次（从 0 计） | 测试 MSE | 测试 MAE |",
    "|---|---:|---:|---:|---:|",
]
for name in ("seed1", "seed2", "seed3"):
    if name in summaries:
        item = summaries[name]
        lines.append(f"| {name} | {item['epochs']} | {item['best_epoch_zero_based']} | {item['test_mse']:.5f} | {item['test_mae']:.5f} |")
lines += [
    "",
    f"本次 {len(seeds)} 种子均值±样本标准差：MSE={seed_stats['test_mse']['mean']:.5f}±{seed_stats['test_mse']['sample_std'] or 0:.5f}，MAE={seed_stats['test_mae']['mean']:.5f}±{seed_stats['test_mae']['sample_std'] or 0:.5f}。",
    f"论文 5 种子均值±标准差：MSE={PAPER['complete']['mse']:.5f}±{PAPER['complete']['mse_std']:.5f}，MAE={PAPER['complete']['mae']:.5f}±{PAPER['complete']['mae_std']:.5f}。",
    f"本次均值相对论文均值：MSE 高 {(seed_stats['test_mse']['mean'] / PAPER['complete']['mse'] - 1) * 100:.2f}%，MAE 高 {(seed_stats['test_mae']['mean'] / PAPER['complete']['mae'] - 1) * 100:.2f}%；两者种子数不同，作为指标对照而非严格统计检验。",
    "",
    "![验证集逐轮曲线](figures/validation_curves.png)",
    "",
    "![论文指标对照](figures/paper_comparison.png)",
    "",
]
if "wo_transformer_seed1" in summaries:
    ab = summaries["wo_transformer_seed1"]
    lines += [
        "## 消融：移除 Transformer",
        "",
        f"消融（seed=1）：测试 MSE={ab['test_mse']:.5f}，MAE={ab['test_mae']:.5f}。完整模型同种子：MSE={summaries['seed1']['test_mse']:.5f}，MAE={summaries['seed1']['test_mae']:.5f}。",
        f"相对同种子的完整模型，消融 MSE 高 {(ab['test_mse'] / summaries['seed1']['test_mse'] - 1) * 100:.2f}%，MAE 高 {(ab['test_mae'] / summaries['seed1']['test_mae'] - 1) * 100:.2f}%。",
        f"论文 w/o Transformer 的 5 种子均值：MSE={PAPER['wo_transformer']['mse']:.5f}，MAE={PAPER['wo_transformer']['mae']:.5f}。",
        "本次在独立代码副本中跳过 Transformer 及其位置编码，保留后续图网络。论文未提供消融实现脚本，因此此项是按文字定义实现的一次近似复现；单次差异不宜作显著性结论。",
        "",
        "![Transformer 消融对照](figures/ablation_comparison.png)",
        "",
    ]
lines += [
    "## 文件说明",
    "",
    "- `reproduction_logs/`：各次运行的原始输出与退出状态。",
    "- `logs/`：官方训练脚本生成的指标日志。",
    "- `epoch_metrics.csv`：每轮指标；`run_summary.csv`、`summary.json`：汇总数据。",
    "- `figures/`：可随整个文件夹移动的 PNG 图片。",
    "- `reproduction_logs/run_B.sh`、`reproduction_logs/run_extras.sh`：实验命令。",
    "- `source/`：官方代码、处理后数据压缩包和独立的消融代码副本；使用方法见 `source/README_reproduce.md`。",
    "",
    "官方训练脚本没有保存模型权重；本文件夹保存了日志、指标、图表和实验命令。复制时请移动整个文件夹，以保持 Markdown 图片可见。",
    "",
]
print(json.dumps(summary, ensure_ascii=False, indent=2))
