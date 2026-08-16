# 自适应约束系统参数优化算法

本项目用于基于结构化仿真数据，构建面向临碰撞场景的自适应约束系统参数优化算法。

当前代码目标是形成最小可运行闭环：

1. 读取 `data/raw/injury_data.xlsx`。
2. 训练目标特异的结构化损伤风险代理模型，直接预测 `Amax`、`Dmax`、`HIC`、`Nij` 等基础连续损伤响应，并按参考公式派生 `CTI`。
3. 对每个测试样本执行分层逐点参数寻优。
4. 按 `reference/reference material/changan_Injury_Criteria_AIS_Cal_original.py` 中的公式逻辑将 `HIC`、派生 `CTI` 和 `Nij` 映射为 AIS/MAIS，输出 Base 与 Opt 的预测降损对比、置信度指标和典型高速复核 case。

## 目录结构

```text
./
├── run_pipeline.py        # 主运行入口
├── configs/               # 运行配置
├── core/                  # 算法核心源码
├── experiments/           # 代理模型迭代、诊断脚本和可执行 Notebook
├── reports/               # 代理模型实验技术报告
├── data/raw/              # 原始结构化仿真数据
├── outputs/               # 本地运行输出
└── reference/             # 参考材料
```

## 运行方式

建议使用本地 `pytorch` 环境：

```bash
conda activate pytorch
python run_pipeline.py
```

也可以显式指定配置：

```bash
python run_pipeline.py --config configs/default_config.yaml
```

运行后会在 `outputs/` 下生成一次新的结果目录，主要包括：

- `data_check_summary.yaml`
- `surrogate_model_metrics.yaml`，其中包含 `train / val / test / test_high_speed` 四个切片的代理模型指标
- `surrogate_predictions.csv`，其中包含代理模型逐样本真实值、预测值和派生 AIS/MAIS 明细
- `evaluation_results.csv`
- `summary_report.yaml`
- `typical_high_speed_cases.csv`
- `models/surrogate_model_bundle.joblib`

## 代理模型设计

代理模型使用带符号连续比例 `input_overlap_signed` 表达碰撞偏置幅度与主驾侧方向，原始 `input_overlap` 编码只用于数据接口映射和结果审计。四个直接目标分别采用经固定验证集选择的回归器：`Amax` 使用 ExtraTrees 与 RandomForest 加权，`Dmax` 使用 XGBoost，`HIC` 使用 HistGradientBoosting，`Nij` 使用 HistGradientBoosting 与 XGBoost 加权。融合权重由训练集五折折外预测确定。

五轮迭代和瓶颈诊断可通过以下命令复现：

```bash
python -m experiments.run_surrogate_iterations --output_dir outputs/surrogate_model_optimization_20260804
python -m experiments.diagnose_surrogate_limits
```

已执行的实验 Notebook 位于 `experiments/surrogate_model_optimization.ipynb`，技术报告位于 `reports/surrogate_model_optimization/report.html`。

## 结果绘图

### 代理模型结果图绘制
可以基于 `surrogate_predictions.csv` 绘制代理模型评估图，默认输出 `test` 与 `test_high_speed` 两个切片的回归散点图和分类混淆矩阵：

```bash
python plot_surrogate_results.py --pred_csv outputs/<run_dir>/surrogate_predictions.csv
```

脚本会在结果目录下生成 `plots_surrogate_model/`。如需指定切片，可以使用：

```bash
python plot_surrogate_results.py --pred_csv outputs/<run_dir>/surrogate_predictions.csv --data_scope train val test test_high_speed
```
### 寻优评估结果图绘制

可以基于 `evaluation_results.csv` 绘制单 case 的 Base/Opt 对比图：

```bash
python plot_eval_cases.py --eval_csv outputs/<run_dir>/evaluation_results.csv --topn_risk 10 --high_speed_only
```

脚本会在结果目录下生成 `plots_eval_cases/`，每个 case 包含控制参数、连续损伤响应、AIS/MAIS 和严重损伤风险四张图。也可以显式指定 case：

```bash
python plot_eval_cases.py --eval_csv outputs/<run_dir>/evaluation_results.csv --case_ids 181 249
```

## 完成口径

当默认命令能够无报错完成，并生成上述结果文件，且 `summary_report.yaml` 中包含全样本与 `velocity >= 40 km/h` 的预测降损统计时，可认为初版闭环完成。
