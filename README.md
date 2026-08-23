# 自适应约束系统参数优化算法

本项目基于结构化仿真数据，构建面向临碰撞场景的自适应约束系统参数优化算法。

程序完成以下工作：

1. 读取 `data/raw/injury_data.xlsx`。
2. 分别训练 `Amax`、`Dmax`、`HIC` 和 `Nij` 的“损伤输出特异”回归模型，并按参考公式计算 `CTI`。
3. 代理模型训练完成后，对完整数据集中的每个工况逐一搜索约束系统参数。
4. 根据 `reference/reference material/changan_Injury_Criteria_AIS_Cal_original.py` 中的公式，由 `HIC`、`CTI` 和 `Nij` 计算 AIS/MAIS，输出原始参数与优化参数的预测损伤对比、预测不确定性以及典型高速工况。

## 目录结构

```text
./
├── run_pipeline.py        # 主运行入口
├── configs/               # 运行配置
├── core/                  # 算法核心源码
├── experiments/           # 代理模型试验、诊断脚本、项目历史记录和可执行 Notebook
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

指定配置文件时运行：

```bash
python run_pipeline.py --config configs/default_config.yaml
```

运行后会在 `outputs/` 下生成一次新的结果目录，主要包括：

- `data_check_summary.yaml`
- `surrogate_model_metrics.yaml`，其中包含训练集、测试集以及测试集中高速工况的代理模型指标
- `surrogate_predictions.csv`，其中包含代理模型逐样本真实值、预测值和派生 AIS/MAIS 明细
- `evaluation_results.csv`
- `summary_report.yaml`
- `typical_high_speed_cases.csv`
- `models/surrogate_model_bundle.joblib`

## 代理模型设计

代理模型使用带符号连续比例 `input_overlap_signed` 表示碰撞偏置大小及主驾侧方向，原始 `input_overlap` 编码仅用于将数据转换为带符号比例，并保留在输出结果中供核对。乘员身高、BMI 和启用状态下的安全带限力值作为连续特征；安全带限力启用状态作为状态特征。乘员质量由身高和 BMI 计算，CTI 所需胸深根据参考体型节点进行双线性插值。数据按 MAIS 分层划分为训练集和测试集，默认比例为 8:2。四个连续损伤指标分别配置回归模型：`Amax` 融合 XGBoost、径向基支持向量回归和随机森林；`Dmax`、`HIC` 同时使用原始目标值和对数变换目标值训练梯度提升模型；`Nij` 组合直方梯度提升与 XGBoost。每个指标最多保留四个回归模型，并通过线性校准调整整体预测尺度与偏移。四项基础损伤响应的预测值均限制为非负数。

截止2026-08-23，当前默认配置在 500 条测试工况上的 Amax、Dmax、HIC、Nij 和 CTI R² 分别为 0.5837、0.6503、0.6827、0.4974 和 0.6526；CTI AIS、HIC AIS、Nij AIS、MAIS 和 AIS3+ 准确率分别为 84.0%、53.6%、80.4%、50.8% 和 80.8%。对应模型和完整评估结果保存在 `outputs/run_20260823_continuous_surrogate_final`。

运行以下命令可复现基础模型比较和误差分析：

```bash
python -m experiments.run_surrogate_iterations
python -m experiments.diagnose_surrogate_limits
```

`experiments/surrogate_model_optimization.ipynb` 和 `reports/surrogate_model_optimization/report.html` 保存了 2026-08-04 的实验结果。当前正式结果见 `outputs/run_20260823_continuous_surrogate_final` 中的 `split_info.yaml`、`surrogate_model_metrics.yaml` 和 `summary_report.yaml`。

## 损伤预测可视化界面

项目根目录下的 `injury_surrogate_gui.py` 提供单工况预测界面。输入包括碰撞速度、角度、重叠率、连续的乘员身高与 BMI、气囊状态、连续的安全带限力值及座椅参数；安全带不启用限力时输入 `inf`。界面初次打开时输入框为空，点击“载入示例”后填入内置示例。输出包括 Amax、Dmax、CTI、HIC、Nij、分项 AIS 和 MAIS。启动命令：

```bash
python injury_surrogate_gui.py
```

程序默认加载 `outputs/` 中最近生成的代理模型。指定模型文件时运行：

```bash
python injury_surrogate_gui.py --model outputs/<run_dir>/models/surrogate_model_bundle.joblib
```

内部测试用训练集工况示例保存在 `experiments/injury_surrogate_input_examples.md`。

## 结果绘图

### 代理模型结果图绘制

使用 `surrogate_predictions.csv` 绘制代理模型评估图。默认输出测试集和高速测试工况的回归散点图与分类混淆矩阵：

```bash
python plot_surrogate_results.py --pred_csv outputs/<run_dir>/surrogate_predictions.csv
```

图像保存在结果目录的 `plots_surrogate_model/` 中。指定数据范围时运行：

```bash
python plot_surrogate_results.py --pred_csv outputs/<run_dir>/surrogate_predictions.csv --data_scope train test test_high_speed
```
### 寻优评估结果图绘制

使用 `evaluation_results.csv` 绘制单个工况在原始参数和优化参数下的结果对比图：

```bash
python plot_eval_cases.py --eval_csv outputs/<run_dir>/evaluation_results.csv --topn_risk 10 --high_speed_only
```

图像保存在结果目录的 `plots_eval_cases/` 中，每个工况包含控制参数、连续损伤指标、AIS/MAIS 和严重损伤风险四张图。指定工况编号时运行：

```bash
python plot_eval_cases.py --eval_csv outputs/<run_dir>/evaluation_results.csv --case_ids 181 249
```

## 运行结果检查

默认命令运行结束后，应生成上述结果文件；`summary_report.yaml` 应同时包含完整数据集和 `velocity >= 40 km/h` 工况的预测降损统计。
