"""绘制代理模型逐样本评估结果图。

该脚本读取 pipeline 导出的 surrogate_predictions.csv，并按数据切片生成回归散点图和分类混淆矩阵。绘图脚本只消费既有结果文件，不参与模型训练、推理或参数寻优。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score


CONTINUOUS_METRICS: Sequence[Tuple[str, str]] = (
    ("Amax", "Amax"),
    ("Dmax", "Dmax"),
    ("HIC", "HIC"),
    ("Nij", "Nij"),
    ("CTI", "CTI"),
)

CLASSIFICATION_METRICS: Sequence[Tuple[str, str, List[int]]] = (
    ("hic_AIS", "HIC AIS", list(range(6))),
    ("cti_AIS", "CTI AIS", list(range(6))),
    ("nij_AIS", "Nij AIS", list(range(6))),
    ("MAIS", "MAIS", list(range(6))),
    ("MAIS_GE3", "MAIS >= 3", [0, 1]),
)

PRIMARY_SCOPES = ["test", "test_high_speed"]

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 14,
    }
)


def parse_args() -> argparse.Namespace:
    """解析代理模型绘图命令行参数。"""
    parser = argparse.ArgumentParser(description="Plot surrogate model prediction diagnostics.")
    parser.add_argument(
        "--pred_csv",
        type=str,
        required=True,
        help="pipeline 导出的 surrogate_predictions.csv 路径。",
    )
    parser.add_argument(
        "--data_scope",
        nargs="+",
        default=None,
        help="需要绘图的数据切片；默认绘制 test 和 test_high_speed。",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="图片输出目录；默认写入 surrogate_predictions.csv 同级目录下的 plots_surrogate_model。",
    )
    parser.add_argument("--dpi", type=int, default=180, help="输出图片 DPI。")
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, columns: Iterable[str], scope: str) -> None:
    """检查结果表中是否包含绘图所需字段。"""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{scope} 缺少必要列: {missing}")


def _regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """计算散点图标题中展示的回归指标。"""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return rmse, mae, r2


def _axis_limits(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """根据真实值与预测值共同确定 true-vs-pred 坐标范围。"""
    values = np.concatenate([y_true, y_pred]).astype(float)
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    if np.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 1.0)
    else:
        padding = (upper - lower) * 0.05
    return lower - padding, upper + padding


def _plot_regression_scatter(scope_df: pd.DataFrame, data_scope: str, output_path: Path, dpi: int) -> None:
    """为一个数据切片绘制基础连续损伤响应与派生 CTI 的回归散点图。"""
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.4), squeeze=False, constrained_layout=True)
    color_values = scope_df["true_MAIS"].to_numpy(dtype=int)
    scatter = None

    for axis, (metric, title) in zip(axes.ravel(), CONTINUOUS_METRICS):
        y_true = scope_df[f"true_{metric}"].to_numpy(dtype=float)
        y_pred = scope_df[f"pred_{metric}"].to_numpy(dtype=float)
        lower, upper = _axis_limits(y_true, y_pred)
        rmse, mae, r2 = _regression_scores(y_true, y_pred)

        scatter = axis.scatter(
            y_true,
            y_pred,
            c=color_values,
            cmap="viridis",
            vmin=0,
            vmax=5,
            s=28,
            alpha=0.78,
            edgecolors="none",
        )
        axis.plot([lower, upper], [lower, upper], color="#D62728", linestyle="--", linewidth=1.2)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_title(f"{title}\nRMSE={rmse:.3g}, MAE={mae:.3g}, R2={r2:.3f}")
        axis.set_xlabel("True")
        axis.set_ylabel("Predicted")
        axis.grid(True, linestyle="--", alpha=0.25)

    for axis in axes.ravel()[len(CONTINUOUS_METRICS) :]:
        axis.axis("off")

    if scatter is not None:
        fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.86, label="True MAIS")
    fig.suptitle(f"Surrogate Regression Scatter - {data_scope} (n={len(scope_df)})")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrices(scope_df: pd.DataFrame, data_scope: str, output_path: Path, dpi: int) -> None:
    """为一个数据切片绘制 AIS/MAIS 分类混淆矩阵。"""
    matrices = []
    for metric, title, labels in CLASSIFICATION_METRICS:
        y_true = scope_df[f"true_{metric}"].to_numpy(dtype=int)
        y_pred = scope_df[f"pred_{metric}"].to_numpy(dtype=int)
        cm = confusion_matrix(y_true, y_pred, labels=labels).astype(int)
        acc = float(accuracy_score(y_true, y_pred))
        matrices.append((metric, title, labels, cm, acc))

    vmax = max(int(cm.max()) for _, _, _, cm, _ in matrices)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.1), squeeze=False, constrained_layout=True)
    image = None
    for axis, (_, title, labels, cm, acc) in zip(axes.ravel(), matrices):
        image = axis.imshow(cm, cmap="Blues", vmin=0, vmax=max(vmax, 1))
        axis.set_title(f"{title}\nAcc={acc:.3f}")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_xticks(np.arange(len(labels)))
        axis.set_yticks(np.arange(len(labels)))
        axis.set_xticklabels(labels)
        axis.set_yticklabels(labels)
        threshold = max(vmax, 1) / 2.0
        for row_idx, col_idx in np.ndindex(cm.shape):
            axis.text(
                col_idx,
                row_idx,
                str(int(cm[row_idx, col_idx])),
                ha="center",
                va="center",
                color="white" if cm[row_idx, col_idx] > threshold else "black",
                fontsize=9,
            )

    for axis in axes.ravel()[len(CLASSIFICATION_METRICS) :]:
        axis.axis("off")

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, label="Count")
    fig.suptitle(f"Surrogate Classification Confusion Matrix - {data_scope} (n={len(scope_df)})")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _select_scopes(df: pd.DataFrame, requested: List[str] | None) -> List[str]:
    """确定需要绘图的数据切片顺序。"""
    available = set(df["data_scope"].astype(str).unique())
    scopes = requested if requested is not None else [scope for scope in PRIMARY_SCOPES if scope in available]
    if not scopes:
        scopes = sorted(available)
    missing = [scope for scope in scopes if scope not in available]
    if missing:
        raise ValueError(f"surrogate_predictions.csv 中不存在以下 data_scope: {missing}")
    return scopes


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    pred_csv = Path(args.pred_csv).resolve()
    if not pred_csv.is_file():
        raise FileNotFoundError(f"代理模型预测明细 CSV 不存在: {pred_csv}")

    df = pd.read_csv(pred_csv)
    required = (
        ["data_scope", "true_MAIS"]
        + [f"true_{metric}" for metric, _ in CONTINUOUS_METRICS]
        + [f"pred_{metric}" for metric, _ in CONTINUOUS_METRICS]
        + [f"true_{metric}" for metric, _, _ in CLASSIFICATION_METRICS]
        + [f"pred_{metric}" for metric, _, _ in CLASSIFICATION_METRICS]
    )
    _require_columns(df, required, "surrogate_predictions.csv")

    output_root = Path(args.out_dir).resolve() if args.out_dir else pred_csv.parent / "plots_surrogate_model"
    output_root.mkdir(parents=True, exist_ok=True)

    figure_count = 0
    for data_scope in _select_scopes(df, args.data_scope):
        scope_df = df[df["data_scope"].astype(str) == data_scope].reset_index(drop=True)
        scope_dir = output_root / data_scope
        scope_dir.mkdir(parents=True, exist_ok=True)
        _plot_regression_scatter(
            scope_df,
            data_scope,
            scope_dir / "01_regression_scatter.png",
            int(args.dpi),
        )
        _plot_confusion_matrices(
            scope_df,
            data_scope,
            scope_dir / "02_classification_confusion_matrix.png",
            int(args.dpi),
        )
        figure_count += 2

    print(f"Generated {figure_count} surrogate model figures: {output_root}")


if __name__ == "__main__":
    main()
