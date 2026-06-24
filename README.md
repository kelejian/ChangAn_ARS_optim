# 自适应约束系统参数优化算法

本项目用于基于结构化仿真数据，构建面向临碰撞场景的自适应约束系统参数优化算法。

当前代码目标是形成最小可运行闭环：

1. 读取 `data/raw/injury_data.xlsx`。
2. 训练轻量化结构化损伤风险代理模型，直接预测 `Amax`、`Dmax`、`HIC`、`Nij` 等基础连续损伤响应，并按参考公式派生 `CTI`。
3. 对每个测试样本执行分层逐点参数寻优。
4. 按 `reference_doc/4_Injury_Criteria_AIS_Cal.py` 中的公式逻辑将 `HIC`、派生 `CTI` 和 `Nij` 映射为 AIS/MAIS，输出 Base 与 Opt 的预测降损对比、置信度指标和典型高速复核 case。

## 目录结构

```text
./
├── run_pipeline.py        # 主运行入口
├── configs/               # 运行配置
├── core/                  # 算法核心源码
├── data/raw/              # 原始结构化仿真数据
├── outputs/               # 本地运行输出
└── reference_doc/         # 参考材料
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
- `surrogate_model_metrics.yaml`
- `evaluation_results.csv`
- `summary_report.yaml`
- `typical_high_speed_cases.csv`
- `models/surrogate_model_bundle.joblib`

## 完成口径

当默认命令能够无报错完成，并生成上述结果文件，且 `summary_report.yaml` 中包含全样本与 `velocity >= 40 km/h` 的预测降损统计时，可认为初版闭环完成。
