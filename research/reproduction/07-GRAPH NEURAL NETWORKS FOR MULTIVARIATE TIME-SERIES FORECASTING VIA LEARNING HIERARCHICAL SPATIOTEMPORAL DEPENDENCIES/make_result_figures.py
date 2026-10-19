"""Generate PEMSD8 experiment figures from the saved training logs."""

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "figures"
OUT.mkdir(exist_ok=True)

font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "savefig.dpi": 200})

LOGS = {
    "完整模型 · 种子 0": "pemsd8_train.log",
    "完整模型 · 种子 1": "pemsd8_seed1.log",
    "完整模型 · 种子 2": "pemsd8_seed2.log",
    "去动态图 · 种子 0": "pemsd8_wo_dg_seed0.log",
}
PAPER = {"MAE": 13.718, "RMSE": 23.405, "MAPE": 9.077}
METRICS = ("MAE", "RMSE", "MAPE")


def read_log(filename):
    content = (ROOT / "results" / "logs" / filename).read_text(
        encoding="utf-8", errors="replace"
    )
    match = re.search(
        r"Average Horizon, MAE: ([\d.]+), RMSE: ([\d.]+), MAPE: ([\d.]+)%",
        content,
    )
    if match is None:
        raise ValueError(f"No final test result in {filename}")
    metrics = dict(zip(METRICS, map(float, match.groups())))
    val = [
        (int(epoch), float(loss))
        for epoch, loss in re.findall(
            r"Val Epoch (\d+): average Loss: ([\d.]+)", content
        )
    ]
    if not val:
        raise ValueError(f"No validation history in {filename}")
    return metrics, np.asarray(val)


data = {name: read_log(filename) for name, filename in LOGS.items()}
full_names = list(LOGS)[:3]
ablation_name = list(LOGS)[3]
colors = ["#2563a6", "#6199cc", "#9dc2e4"]
orange = "#db713b"

# Figure 1: three-seed test metrics and the paper's full-model mean.
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
for ax, metric in zip(axes, METRICS):
    values = [data[name][0][metric] for name in full_names]
    ax.scatter(range(3), values, s=90, color=colors, zorder=3)
    mean = float(np.mean(values))
    ax.axhline(mean, color="#2563a6", lw=1.6, label="本次三种子均值")
    ax.axhline(PAPER[metric], color="#666666", lw=1.4, ls="--", label="论文十次均值")
    ax.set_xticks(range(3), ["种子 0", "种子 1", "种子 2"])
    ax.set_xlim(-0.35, 2.35)
    ax.set_title(metric + ("（%）" if metric == "MAPE" else ""))
    ax.grid(axis="y", alpha=0.22)
    span = max(max(values + [PAPER[metric]]) - min(values + [PAPER[metric]]), 0.2)
    ax.set_ylim(min(values + [PAPER[metric]]) - span * 0.25,
                max(values + [PAPER[metric]]) + span * 0.35)
    for i, value in enumerate(values):
        ax.annotate(f"{value:.3f}", (i, value), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8)
axes[0].legend(loc="lower left", fontsize=8)
fig.suptitle("PEMSD8：完整模型三个随机种子的测试结果", fontsize=15, weight="bold")
fig.savefig(OUT / "PEMSD8_三种子指标.png", bbox_inches="tight")
plt.close(fig)

# Figure 2: paired same-seed ablation comparison.
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), constrained_layout=True)
for ax, metric in zip(axes, METRICS):
    full = data[full_names[0]][0][metric]
    ablated = data[ablation_name][0][metric]
    ax.plot([full, ablated], [0, 0], color="#a7adb4", lw=2, zorder=1)
    ax.scatter([full], [0], s=135, color=colors[0], zorder=3,
               label="完整模型 · 种子 0")
    ax.scatter([ablated], [0], s=135, color=orange, zorder=3,
               label="去动态图 · 种子 0")
    ax.annotate(f"{full:.3f}", (full, 0), xytext=(0, -24),
                textcoords="offset points", ha="center", fontsize=9)
    ax.annotate(f"{ablated:.3f}", (ablated, 0), xytext=(0, 12),
                textcoords="offset points", ha="center", fontsize=9)
    ax.set_xlim(full - (ablated - full) * 0.3, ablated + (ablated - full) * 0.3)
    ax.set_ylim(-0.55, 0.8)
    ax.set_yticks([])
    ax.set_title(metric + ("（%）" if metric == "MAPE" else ""))
    ax.set_xlabel(f"误差增加 {(ablated / full - 1) * 100:.1f}%")
    ax.spines[["left", "right", "top"]].set_visible(False)
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.00), fontsize=8)
fig.suptitle("PEMSD8：去动态图消融与完整模型对照（相同种子）", fontsize=15,
             weight="bold")
fig.savefig(OUT / "PEMSD8_消融指标对照.png", bbox_inches="tight")
plt.close(fig)

# Figure 3: complete validation histories from the logs.
fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
for i, name in enumerate(full_names):
    val = data[name][1]
    ax.plot(val[:, 0], val[:, 1], color=colors[i], lw=1.8,
            label=name)
val = data[ablation_name][1]
ax.plot(val[:, 0], val[:, 1], color=orange, lw=2.0, label=ablation_name)
ax.set(title="PEMSD8：验证损失随训练轮次变化", xlabel="训练轮次",
       ylabel="验证集 MAE 损失")
ax.grid(alpha=0.22)
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(13.0, 24.5)
fig.savefig(OUT / "PEMSD8_验证损失曲线.png", bbox_inches="tight")
plt.close(fig)

for path in sorted(OUT.glob("*.png")):
    print(path)
