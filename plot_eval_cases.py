"""绘制单个 case 的 Base/Opt 结果对比图。

该脚本读取 pipeline 导出的 evaluation_results.csv，并按 case 生成控制参数、连续损伤响应、AIS/MAIS 等级和严重损伤风险对比图。绘图脚本只消费既有结果文件，不参与模型训练或参数寻优。
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONTROL_COLUMNS = [
    "airbag",
    "kneeairbag",
    "ll_level",
    "delta_pos",
    "recline_angle",
]

CONTEXT_COLUMNS = [
    "input_velocity",
    "input_angle",
    "input_overlap",
    "input_overlap_signed",
    "input_swing_angle",
    "input_type_num",
]

CONTINUOUS_METRICS = ["Amax", "Dmax", "CTI", "HIC", "Nij"]
AIS_METRICS = ["cti_AIS", "hic_AIS", "nij_AIS", "MAIS"]
RISK_METRICS = ["P_MAIS_GE3"]

UNIT_MAP = {
    "airbag": "state",
    "kneeairbag": "state",
    "ll_level": "level",
    "delta_pos": "mm",
    "recline_angle": "rad",
    "Amax": "g",
    "Dmax": "mm",
    "CTI": "-",
    "HIC": "-",
    "Nij": "-",
    "cti_AIS": "level",
    "hic_AIS": "level",
    "nij_AIS": "level",
    "MAIS": "level",
    "P_MAIS_GE3": "prob.",
    "Opt_Uncertainty": "-",
    "Opt_OODScore": "-",
}

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
    }
)


def parse_args() -> argparse.Namespace:
    """解析绘图命令行参数。"""
    parser = argparse.ArgumentParser(description="Plot ChangAn ARS evaluation cases.")
    parser.add_argument(
        "--eval_csv",
        type=str,
        required=True,
        help="pipeline 导出的 evaluation_results.csv 路径。",
    )
    parser.add_argument(
        "--case_ids",
        nargs="*",
        default=None,
        help="显式指定需要绘图的 case_id 列表，例如 --case_ids 181 249。",
    )
    parser.add_argument(
        "--topn_risk",
        type=int,
        default=0,
        help="按 Reduction_P_MAIS_GE3 从高到低选择前 N 个 case。",
    )
    parser.add_argument(
        "--topn_mais",
        type=int,
        default=0,
        help="按 Reduction_MAIS 从高到低选择前 N 个 case。",
    )
    parser.add_argument(
        "--high_speed_only",
        action="store_true",
        help="自动选取 topN 时只在高速工况中筛选。",
    )
    parser.add_argument(
        "--high_speed_threshold",
        type=float,
        default=40.0,
        help="高速工况阈值，默认 40 km/h。",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="图片输出目录；默认写入 evaluation_results.csv 同级目录下的 plots_eval_cases。",
    )
    parser.add_argument("--dpi", type=int, default=180, help="输出图片 DPI。")
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, columns: Iterable[str], scope: str) -> None:
    """检查结果表中是否包含绘图所需字段。"""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{scope} 缺少必要列: {missing}")


def _format_float(value: object, digits: int = 4) -> str:
    """以紧凑形式渲染数值标签。"""
    if pd.isna(value):
        return "NaN"
    return f"{float(value):.{digits}g}"


def _format_recommendation_flag(value: object) -> str:
    """将中文推荐标志转换为图片标题中稳定可渲染的英文标签。"""
    mapping = {
        "推荐": "recommended",
        "谨慎推荐": "cautious",
        "不推荐": "not_recommended",
    }
    return mapping.get(str(value), str(value))


def _format_context(row: pd.Series) -> str:
    """将工况信息压缩成标题中的上下文说明。"""
    parts = [
        f"v={_format_float(row['input_velocity'])} km/h",
        f"angle={_format_float(row['input_angle'])} deg",
        f"overlap={int(row['input_overlap'])}",
        f"signed_overlap={_format_float(row['input_overlap_signed'])}",
        f"swing={_format_float(row['input_swing_angle'])} rad",
        f"type={int(row['input_type_num'])}",
        f"flag={_format_recommendation_flag(row['RecommendationFlag'])}",
    ]
    return "\n".join(textwrap.wrap(", ".join(parts), width=110, break_long_words=False))


def _stage_values(row: pd.Series, metric: str) -> List[float]:
    """读取 Base/Opt 两阶段的同名指标。"""
    return [float(row[f"Base_{metric}"]), float(row[f"Opt_{metric}"])]


def _annotate_bars(ax, bars, fmt: str) -> None:
    """在柱状图上方标注数值。"""
    for bar in bars:
        value = bar.get_height()
        if np.isnan(value):
            continue
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _plot_metric_group(
    row: pd.Series,
    metrics: List[str],
    output_path: Path,
    title: str,
    *,
    include_true_mais: bool = False,
    force_level_axis: bool = False,
    dpi: int,
) -> None:
    """绘制一组 Base/Opt 指标对比柱状图。"""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 4.2), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        labels = ["Base", "Opt"]
        values = _stage_values(row, metric)
        colors = ["#4C78A8", "#F58518"]
        if include_true_mais and metric == "MAIS" and "True_MAIS" in row.index:
            labels.append("True")
            values.append(float(row["True_MAIS"]))
            colors.append("#54A24B")

        bars = ax.bar(labels, values, color=colors, width=0.6)
        value_fmt = "{:.0f}" if force_level_axis else "{:.4g}"
        _annotate_bars(ax, bars, value_fmt)
        ax.set_title(f"{metric} ({UNIT_MAP.get(metric, '-')})")
        ax.set_ylabel(UNIT_MAP.get(metric, "-"))
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if force_level_axis:
            ax.set_ylim(0, 5)
            ax.set_yticks(range(6))

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_controls(row: pd.Series, output_path: Path, title: str, dpi: int) -> None:
    """绘制 5 个允许控制变量的 Base/Opt 对比图。"""
    fig, axes = plt.subplots(1, len(CONTROL_COLUMNS), figsize=(4.1 * len(CONTROL_COLUMNS), 4.2), squeeze=False)
    for ax, control in zip(axes[0], CONTROL_COLUMNS):
        values = _stage_values(row, control)
        bars = ax.bar(["Base", "Opt"], values, color=["#4C78A8", "#F58518"], width=0.6)
        _annotate_bars(ax, bars, "{:.4g}")
        ax.set_title(f"{control} ({UNIT_MAP[control]})")
        ax.set_ylabel(UNIT_MAP[control])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_risk(row: pd.Series, output_path: Path, title: str, dpi: int) -> None:
    """绘制联合严重损伤概率及 Opt 可信度辅助指标。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), squeeze=False)

    risk_values = _stage_values(row, "P_MAIS_GE3")
    bars = axes[0][0].bar(["Base", "Opt"], risk_values, color=["#4C78A8", "#F58518"], width=0.6)
    _annotate_bars(axes[0][0], bars, "{:.3f}")
    axes[0][0].set_title("P_MAIS_GE3")
    axes[0][0].set_ylabel("prob.")
    axes[0][0].set_ylim(0, 1)
    axes[0][0].grid(axis="y", linestyle="--", alpha=0.3)

    quality_values = [float(row["Opt_Uncertainty"]), float(row["Opt_OODScore"])]
    bars = axes[0][1].bar(["Uncertainty", "OOD"], quality_values, color=["#72B7B2", "#B279A2"], width=0.55)
    _annotate_bars(axes[0][1], bars, "{:.3f}")
    axes[0][1].set_title("Opt confidence indicators")
    axes[0][1].set_ylabel("score")
    axes[0][1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_one_case(row: pd.Series, output_dir: Path, dpi: int) -> None:
    """为单个 case 生成四张结果图。"""
    case_id = str(row["case_id"])
    safe_case_id = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in case_id).strip("_")
    prefix = f"case_{safe_case_id}"
    title = f"case_id={case_id} | True_MAIS={row['True_MAIS']} | Base_MAIS={row['Base_MAIS']} -> Opt_MAIS={row['Opt_MAIS']}\n{_format_context(row)}"

    _plot_controls(row, output_dir / f"{prefix}_01_controls.png", title, dpi)
    _plot_metric_group(row, CONTINUOUS_METRICS, output_dir / f"{prefix}_02_continuous_injury.png", title, dpi=dpi)
    _plot_metric_group(
        row,
        AIS_METRICS,
        output_dir / f"{prefix}_03_ais_mais.png",
        title,
        include_true_mais=True,
        force_level_axis=True,
        dpi=dpi,
    )
    _plot_risk(row, output_dir / f"{prefix}_04_risk_confidence.png", title, dpi)


def _select_case_ids(df: pd.DataFrame, args: argparse.Namespace) -> List[str]:
    """根据显式 case_id 与 topN 规则确定需要绘图的样本。"""
    selected = [str(case_id) for case_id in (args.case_ids or [])]
    pool = df
    if args.high_speed_only:
        pool = df[df["input_velocity"] >= float(args.high_speed_threshold)]
        if pool.empty:
            raise ValueError(f"不存在 input_velocity >= {args.high_speed_threshold:g} 的高速样本。")

    if int(args.topn_risk) > 0:
        top_risk = pool.sort_values("Reduction_P_MAIS_GE3", ascending=False).head(int(args.topn_risk))
        selected.extend(top_risk["case_id"].astype(str).tolist())

    if int(args.topn_mais) > 0:
        top_mais = pool.sort_values(["Reduction_MAIS", "Reduction_P_MAIS_GE3"], ascending=[False, False]).head(int(args.topn_mais))
        selected.extend(top_mais["case_id"].astype(str).tolist())

    deduped: List[str] = []
    seen = set()
    for case_id in selected:
        if case_id in seen:
            continue
        seen.add(case_id)
        deduped.append(case_id)
    if not deduped:
        raise ValueError("至少需要指定 --case_ids、--topn_risk 或 --topn_mais 之一。")
    return deduped


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    eval_csv = Path(args.eval_csv).resolve()
    if not eval_csv.is_file():
        raise FileNotFoundError(f"评估结果 CSV 不存在: {eval_csv}")

    df = pd.read_csv(eval_csv)
    required = (
        ["case_id", "True_MAIS", "RecommendationFlag", "Opt_Uncertainty", "Opt_OODScore"]
        + CONTEXT_COLUMNS
        + [f"Base_{name}" for name in CONTROL_COLUMNS + CONTINUOUS_METRICS + AIS_METRICS + RISK_METRICS]
        + [f"Opt_{name}" for name in CONTROL_COLUMNS + CONTINUOUS_METRICS + AIS_METRICS + RISK_METRICS]
        + ["Reduction_MAIS", "Reduction_P_MAIS_GE3"]
    )
    _require_columns(df, required, "evaluation_results.csv")

    if df["case_id"].astype(str).duplicated().any():
        duplicated = df.loc[df["case_id"].astype(str).duplicated(), "case_id"].astype(str).unique().tolist()
        raise ValueError(f"结果表中存在重复 case_id，无法唯一绘图: {duplicated}")

    output_dir = Path(args.out_dir).resolve() if args.out_dir else eval_csv.parent / "plots_eval_cases"
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = _select_case_ids(df, args)
    df_indexed = df.assign(_case_id_key=df["case_id"].astype(str)).set_index("_case_id_key", drop=False)
    missing = [case_id for case_id in case_ids if case_id not in df_indexed.index]
    if missing:
        raise ValueError(f"以下 case_id 不存在于结果表: {missing}")

    for case_id in case_ids:
        _plot_one_case(df_indexed.loc[case_id], output_dir, int(args.dpi))
    print(f"Generated {len(case_ids) * 4} figures for {len(case_ids)} cases: {output_dir}")


if __name__ == "__main__":
    main()
