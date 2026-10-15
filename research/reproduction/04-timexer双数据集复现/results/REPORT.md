# TimeXer 部分数据集复现报告

已完成并验证 8 项：ETTh2、ETTm2 各预测长度 96/192/336/720。官方外生变量预测设置（features=MS），输入长度 96，随机种子 2021。

## 测试集结果

论文对照取自 TimeXer 论文 Table 11 的 TimeXer 列；数值仅保留论文给出的三位小数。MSE、MAE 均越低越好。

| 数据集 | 预测长度 | 本次 MSE | 论文 MSE | 本次 MAE | 论文 MAE | 用时 |
|---|---:|---:|---:|---:|---:|---:|
| ETTh2 | 96 | 0.132384 | 0.132 | 0.279582 | 0.280 | 0.9 分钟 |
| ETTh2 | 192 | 0.181474 | 0.181 | 0.332536 | 0.333 | 0.7 分钟 |
| ETTh2 | 336 | 0.223105 | 0.223 | 0.377328 | 0.377 | 2.9 分钟 |
| ETTh2 | 720 | 0.219864 | 0.220 | 0.376133 | 0.376 | 1.1 分钟 |
| ETTm2 | 96 | 0.067294 | 0.067 | 0.188170 | 0.188 | 3.2 分钟 |
| ETTm2 | 192 | 0.100744 | 0.101 | 0.235826 | 0.236 | 11.4 分钟 |
| ETTm2 | 336 | 0.130190 | 0.130 | 0.274544 | 0.275 | 3.7 分钟 |
| ETTm2 | 720 | 0.182155 | 0.182 | 0.332208 | 0.332 | 3.5 分钟 |

论文中这些数值经过四舍五入；以本次指标保留三位小数后对照，八项 MSE/MAE 均一致。不能据此断言底层模型权重或每个预测值与论文完全相同。

## 配置与可复核材料

- 官方源代码提交：`76011909357972bd55a27adba2e1be994d81b327`。本次为避免加载与目标实验无关的模型依赖，进行了最小化导入调整；详见 `environment/instrumentation.patch`。
- 各实验的实际命令、训练日志、开始结束时间、退出码及验证记录保存在 `results/<数据集>/pred_<预测长度>/`。
- 每项的 `checkpoints/`、`results/` 内包含模型权重、预测数组、真实值数组及指标；`COMPLETE.json` 中 `verified=true` 表示这些文件及数组重算指标均核对通过。
- 本次 GPU、依赖版本见 `environment/nvidia_smi.txt`、`environment/pip-freeze.txt`。运行环境可能与论文原始环境不同；这是单一随机种子的一次复现。
- 数据文件 SHA256：

```
a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b  /root/autodl-tmp/timexer_repro/dataset/ETT-small/ETTh2.csv
db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd  /root/autodl-tmp/timexer_repro/dataset/ETT-small/ETTm2.csv
```

- 示例预测图：`results/example_predictions.png`。完整压缩包包含代码、数据、环境记录、编排脚本和所有 8 项原始结果，但不含约数 GB 的 Python 虚拟环境，可依照依赖清单重建。

论文来源：https://arxiv.org/html/2402.19072
