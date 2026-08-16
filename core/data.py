"""数据读取、字段校验与基础派生特征构造。

本模块负责把原始 Excel 表转换为内部统一数据表，并生成数据摘要和 train/val/test 划分信息。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np

from core.injury_mapping import calculate_cti, injury_levels_and_risk
from core.schema import (
    AIS_TARGETS,
    CONTINUOUS_TARGETS,
    FEATURE_COLUMNS,
    OVERLAP_SIGNED_MAP,
    RAW_INPUT_COLUMNS,
    SURROGATE_TARGETS,
)


def load_raw_table(path: Path) -> pd.DataFrame:
    """读取结构化仿真数据表。"""
    if not path.is_file():
        raise FileNotFoundError(f"未找到结构化数据文件: {path}")
    return pd.read_excel(path)


def prepare_dataset(raw_df: pd.DataFrame, case_id_column: str = "case_id") -> pd.DataFrame:
    """构造算法使用的数据表，并将样本编号字段统一为 case_id。"""
    if case_id_column not in raw_df.columns:
        raise KeyError(f"数据表缺少样本编号字段: {case_id_column}")

    df = raw_df.copy()
    if case_id_column != "case_id":
        # 后续流水线统一使用 case_id，配置项只影响外部数据表中的原始列名。
        df = df.rename(columns={case_id_column: "case_id"})

    required = set(["case_id"] + RAW_INPUT_COLUMNS + SURROGATE_TARGETS + ["output_CTI"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"数据表缺少必要字段: {missing}")

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[list(required)].isna().any().any():
        bad_columns = df[list(required)].columns[df[list(required)].isna().any()].tolist()
        raise ValueError(f"必要字段存在无法解析的缺失值或非数值内容: {bad_columns}")

    invalid_overlap = sorted(set(df["input_overlap"].astype(int)) - set(OVERLAP_SIGNED_MAP))
    if invalid_overlap:
        raise ValueError(f"input_overlap 存在未定义编码: {invalid_overlap}")

    df["input_overlap"] = df["input_overlap"].astype(int)
    df["input_airbag"] = df["input_airbag"].astype(int)
    df["input_kneeairbag"] = df["input_kneeairbag"].astype(int)
    df["input_ll_level"] = df["input_ll_level"].astype(int)
    df["input_type_num"] = df["input_type_num"].astype(int)
    # 原始偏置编码只承担数据接口映射；代理模型使用带符号比例统一表达重叠幅度与主驾侧方向。
    df["input_overlap_signed"] = df["input_overlap"].map(OVERLAP_SIGNED_MAP).astype(float)
    calculated_cti = calculate_cti(df["output_Amax"], df["output_Dmax"], df["input_type_num"])
    if not np.allclose(calculated_cti, df["output_CTI"].to_numpy(dtype=float), rtol=0.0, atol=1e-8):
        raise ValueError("output_CTI 与 Amax/Dmax/type_num 按参考公式计算的结果不一致。")
    df["output_CTI"] = calculated_cti
    # AIS/MAIS 由连续损伤指标按项目参考公式统一计算，避免依赖外部表格中可能过期或口径不明的派生列。
    mapped = injury_levels_and_risk(df["output_HIC"], df["output_CTI"], df["output_Nij"])
    df["output_hic_AIS"] = mapped["hic_AIS"].to_numpy(dtype=int)
    df["output_cti_AIS"] = mapped["cti_AIS"].to_numpy(dtype=int)
    df["output_nij_AIS"] = mapped["nij_AIS"].to_numpy(dtype=int)
    df["output_MAIS"] = mapped["MAIS"].to_numpy(dtype=int)
    return df


def make_data_summary(df: pd.DataFrame, high_speed_threshold: float) -> Dict[str, object]:
    """生成数据检查摘要，供运行报告使用。"""
    high_speed_mask = df["input_velocity"] >= high_speed_threshold
    summary: Dict[str, object] = {
        "sample_count": int(len(df)),
        "high_speed_threshold": float(high_speed_threshold),
        "high_speed_count": int(high_speed_mask.sum()),
        "high_speed_ratio": float(high_speed_mask.mean()),
        "feature_columns": list(FEATURE_COLUMNS),
        "continuous_targets": list(CONTINUOUS_TARGETS),
        "ais_targets": list(AIS_TARGETS),
        "input_ranges": {},
        "categorical_counts": {},
        "mais_counts": {
            str(int(k)): int(v)
            for k, v in df["output_MAIS"].value_counts().sort_index().items()
        },
        "high_speed_mais_counts": {
            str(int(k)): int(v)
            for k, v in df.loc[high_speed_mask, "output_MAIS"].value_counts().sort_index().items()
        },
    }
    for column in [
        "input_velocity",
        "input_angle",
        "input_overlap_signed",
        "input_delta_pos",
        "input_recline_angle",
        "input_swing_angle",
    ]:
        summary["input_ranges"][column] = {
            "min": float(df[column].min()),
            "max": float(df[column].max()),
            "mean": float(df[column].mean()),
        }
    for column in [
        "input_overlap",
        "input_airbag",
        "input_kneeairbag",
        "input_ll_level",
        "input_type_num",
    ]:
        summary["categorical_counts"][column] = {
            str(int(k)): int(v)
            for k, v in df[column].value_counts().sort_index().items()
        }
    return summary


def split_dataset(
    df: pd.DataFrame,
    seed: int,
    test_size: float,
    val_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """按 MAIS 分层划分 train/val/test，并记录每个切片的 case_id。"""
    train_val_df, test_df = train_test_split(
        df,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=df["output_MAIS"],
    )
    relative_val = float(val_size) / max(1e-12, 1.0 - float(test_size))
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val,
        random_state=int(seed),
        stratify=train_val_df["output_MAIS"],
    )
    split_info = {
        "seed": int(seed),
        "test_size": float(test_size),
        "val_size": float(val_size),
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "test_count": int(len(test_df)),
        "train_case_ids": [int(v) for v in train_df["case_id"].tolist()],
        "val_case_ids": [int(v) for v in val_df["case_id"].tolist()],
        "test_case_ids": [int(v) for v in test_df["case_id"].tolist()],
    }
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        split_info,
    )
