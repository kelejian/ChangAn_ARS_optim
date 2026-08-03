"""端到端运行流程。

本模块负责组织数据检查、数据划分、代理模型训练、参数寻优和结果文件导出，是入口脚本调用的主流水线。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import yaml

from core.data import load_raw_table, make_data_summary, prepare_dataset, split_dataset
from core.optimizer import optimize_cases, summarize_optimization
from core.surrogate_model import (
    build_surrogate_prediction_table,
    evaluate_surrogate_model_bundle,
    train_surrogate_model_bundle,
)


def load_config(path: Path) -> Dict[str, object]:
    """读取 YAML 配置文件。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run_pipeline(config: Dict[str, object], project_dir: Path, run_name: Optional[str] = None) -> Path:
    """执行数据检查、代理模型训练、逐点寻优和结果导出。"""
    project_dir = project_dir.resolve()
    output_dir = _build_output_dir(project_dir, config, run_name=run_name)
    # 不覆盖已有运行结果，避免误删或混淆已经生成的报告文件。
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在: {output_dir}")

    data_cfg = config.get("data", {}) or {}
    split_cfg = config.get("split", {}) or {}
    opt_cfg = config.get("optimization", {}) or {}
    out_cfg = config.get("output", {}) or {}
    data_path = project_dir / str(data_cfg.get("xlsx_path", "data/raw/injury_data.xlsx"))
    case_id_column = str(data_cfg.get("case_id_column", "case_id"))
    high_speed_threshold = float(data_cfg.get("high_speed_threshold", 40.0))

    # 数据读取和字段整理先于输出目录创建执行，避免数据错误时留下半成品输出目录。
    raw_df = load_raw_table(data_path)
    df = prepare_dataset(raw_df, case_id_column=case_id_column)
    data_summary = make_data_summary(df, high_speed_threshold=high_speed_threshold)

    train_df, val_df, test_df, split_info = split_dataset(
        df=df,
        seed=int(config.get("seed", 2026)),
        test_size=float(split_cfg.get("test_size", 0.2)),
        val_size=float(split_cfg.get("val_size", 0.2)),
    )

    surrogate_bundle = train_surrogate_model_bundle(train_df, config)
    neighbor_k = int(opt_cfg.get("neighbor_k", 5))
    high_speed_test_df = test_df[test_df["input_velocity"] >= high_speed_threshold].reset_index(drop=True)
    # train/val/test 指标用于判断代理模型是否具备基本可用性，其中 test 指标写入完成标志。
    surrogate_metrics = {
        "train": evaluate_surrogate_model_bundle(surrogate_bundle, train_df, neighbor_k=neighbor_k),
        "val": evaluate_surrogate_model_bundle(surrogate_bundle, val_df, neighbor_k=neighbor_k),
        "test": evaluate_surrogate_model_bundle(surrogate_bundle, test_df, neighbor_k=neighbor_k),
    }
    if len(high_speed_test_df) > 0:
        surrogate_metrics["test_high_speed"] = evaluate_surrogate_model_bundle(
            surrogate_bundle,
            high_speed_test_df,
            neighbor_k=neighbor_k,
        )
    else:
        surrogate_metrics["test_high_speed"] = {
            "sample_count": 0,
            "available": False,
            "reason": f"test split 中不存在 velocity >= {high_speed_threshold:g} km/h 的样本。",
        }
    surrogate_predictions = build_surrogate_prediction_table(
        surrogate_bundle,
        {
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "test_high_speed": high_speed_test_df,
        },
        neighbor_k=neighbor_k,
    )

    # 优化仅在测试集上执行，输出 Base 与 Opt 的预测对比结果。
    eval_results = optimize_cases(test_df, surrogate_bundle, config)
    summary_report, typical_cases = summarize_optimization(
        eval_results,
        high_speed_threshold=high_speed_threshold,
        typical_case_count=int(out_cfg.get("typical_case_count", 10)),
    )
    summary_report["run_dir"] = str(output_dir)
    summary_report["completion_markers"] = _build_completion_markers(surrogate_metrics, eval_results, summary_report)

    # 所有结果在计算成功后统一写出，便于将一个输出目录视为一次完整运行记录。
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "models").mkdir(parents=True, exist_ok=False)
    _write_yaml(output_dir / "data_check_summary.yaml", data_summary)
    _write_yaml(output_dir / "split_info.yaml", split_info)
    _write_yaml(output_dir / "surrogate_model_metrics.yaml", surrogate_metrics)
    surrogate_predictions.to_csv(output_dir / "surrogate_predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump(surrogate_bundle, output_dir / "models" / "surrogate_model_bundle.joblib")
    eval_results.to_csv(output_dir / "evaluation_results.csv", index=False, encoding="utf-8-sig")
    _write_yaml(output_dir / "summary_report.yaml", summary_report)
    typical_cases.to_csv(output_dir / "typical_high_speed_cases.csv", index=False, encoding="utf-8-sig")
    return output_dir


def _build_output_dir(project_dir: Path, config: Dict[str, object], run_name: Optional[str]) -> Path:
    """根据配置和可选 run_name 构造本次运行的输出目录。"""
    out_cfg = config.get("output", {}) or {}
    root = project_dir / str(out_cfg.get("root_dir", "outputs"))
    prefix = str(out_cfg.get("run_prefix", "run"))
    if run_name:
        return root / run_name
    return root / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _build_completion_markers(
    surrogate_metrics: Dict[str, object],
    eval_results,
    summary_report: Dict[str, object],
) -> Dict[str, object]:
    """记录完成标志对应的机器可读检查结果。"""
    test_mais = surrogate_metrics["test"]["MAIS"]
    return {
        "surrogate_model_beats_constant_baseline": bool(test_mais["beats_constant_baseline_accuracy"]),
        "evaluation_case_count": int(len(eval_results)),
        "all_cases_summary_available": bool(summary_report["all_cases"]["count"] > 0),
        "high_speed_summary_available": bool(summary_report["high_speed_cases"]["count"] > 0),
        "has_recommended_cases": bool(summary_report["recommended_case_count"] > 0),
        "has_cautious_or_recommended_cases": bool(
            summary_report["recommended_case_count"] + summary_report["cautious_case_count"] > 0
        ),
    }


def _write_yaml(path: Path, data: Dict[str, object]) -> None:
    """以 UTF-8 编码写出 YAML 文件，确保中文字段和说明可正常阅读。"""
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)
