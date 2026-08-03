"""分层逐点参数寻优与结果汇总。

本模块对每个测试样本单独生成候选约束系统参数，用代理模型评估候选风险，并选择综合评分最低的候选作为推荐方案。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from core.schema import CONTROL_COLUMNS, ControlBounds, FEATURE_COLUMNS
from core.surrogate_model import SurrogateModelBundle


def optimize_cases(
    cases_df: pd.DataFrame,
    surrogate_bundle: SurrogateModelBundle,
    config: Dict[str, object],
) -> pd.DataFrame:
    """对每个 case 执行分层逐点搜索。"""
    opt_cfg = config.get("optimization", {}) or {}
    neighbor_k = int(opt_cfg.get("neighbor_k", 5))
    max_eval_cases = opt_cfg.get("max_eval_cases")
    if max_eval_cases is not None:
        cases_df = cases_df.iloc[: int(max_eval_cases)].copy()
    cases_df = cases_df.reset_index(drop=True)

    bounds = ControlBounds()
    score_weights = opt_cfg.get("score_weights", {}) or {}
    delta_offsets = [float(v) for v in opt_cfg.get("delta_pos_offsets", [-80, -40, 0, 40, 80])]
    recline_offsets = [float(v) for v in opt_cfg.get("recline_angle_offsets", [-0.16, -0.08, 0, 0.08, 0.16])]

    # Base 阶段可一次性批量预测；候选阶段也先整体生成再分块预测，避免逐 case 反复调用代理模型造成明显额外开销。
    base_pred_df = surrogate_bundle.predict(cases_df[FEATURE_COLUMNS], neighbor_k=neighbor_k).reset_index(drop=True)
    candidate_frames: List[pd.DataFrame] = []
    for case_index, (_, case) in enumerate(cases_df.iterrows()):
        candidates = _generate_candidates(case, bounds, delta_offsets, recline_offsets)
        candidates["_case_eval_index"] = int(case_index)
        candidate_frames.append(candidates)
    all_candidates = pd.concat(candidate_frames, ignore_index=True)
    # 候选样本数量约为 case 数量乘以每个 case 的网格规模，分块预测可控制内存峰值。
    candidate_pred = _predict_candidates_in_chunks(
        surrogate_bundle=surrogate_bundle,
        candidates=all_candidates,
        neighbor_k=neighbor_k,
    )
    # 综合评分同时考虑预测损伤、严重损伤概率、不确定性、分布外程度和动作幅度。
    candidate_scores = _score_candidates_batch(
        pred_df=candidate_pred,
        candidates=all_candidates,
        base_cases=cases_df,
        weights=score_weights,
    )
    all_candidates["_score"] = candidate_scores
    # 每个 case 独立选择评分最低的候选，避免不同工况之间互相竞争。
    best_candidate_indices = all_candidates.groupby("_case_eval_index")["_score"].idxmin()

    rows: List[Dict[str, object]] = []
    for case_index, (_, case) in enumerate(cases_df.iterrows()):
        best_idx = int(best_candidate_indices.loc[case_index])
        best_candidate = all_candidates.loc[best_idx]
        best_pred = candidate_pred.loc[best_idx]
        rows.append(
            _build_result_row(
                case=case,
                base_pred=base_pred_df.iloc[case_index],
                best_candidate=best_candidate,
                best_pred=best_pred,
                score=float(all_candidates.loc[best_idx, "_score"]),
            )
        )
    return pd.DataFrame(rows)


def summarize_optimization(
    result_df: pd.DataFrame,
    high_speed_threshold: float,
    typical_case_count: int,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """生成全局与高速子集汇总，并挑选典型复核 case。"""
    all_summary = _subset_summary(result_df, "all")
    high_speed = result_df[result_df["input_velocity"] >= high_speed_threshold].copy()
    high_summary = _subset_summary(high_speed, f"velocity>={high_speed_threshold:g}")
    # 典型 case 优先选择推荐等级高且预测降损明显的高速样本，便于后续仿真复核。
    typical = high_speed.sort_values(
        by=["RecommendationRank", "Reduction_MAIS", "Reduction_P_MAIS_GE3"],
        ascending=[True, False, False],
    ).head(int(typical_case_count))
    report = {
        "all_cases": all_summary,
        "high_speed_cases": high_summary,
        "typical_case_count": int(len(typical)),
        "recommended_case_count": int((result_df["RecommendationFlag"] == "推荐").sum()),
        "cautious_case_count": int((result_df["RecommendationFlag"] == "谨慎推荐").sum()),
        "not_recommended_case_count": int((result_df["RecommendationFlag"] == "不推荐").sum()),
    }
    return report, typical


def _generate_candidates(
    case: pd.Series,
    bounds: ControlBounds,
    delta_offsets: Iterable[float],
    recline_offsets: Iterable[float],
) -> pd.DataFrame:
    """围绕当前 case 构造离散组合与座椅连续变量粗网格候选。"""
    rows: List[Dict[str, object]] = []
    base_delta = float(case["input_delta_pos"])
    base_recline = float(case["input_recline_angle"])
    # 座椅位置和靠背角仅在当前值附近搜索，并裁剪到 schema 中定义的物理边界内。
    delta_values = sorted(
        {
            float(np.clip(base_delta + offset, bounds.delta_pos_min, bounds.delta_pos_max))
            for offset in delta_offsets
        }
    )
    recline_values = sorted(
        {
            float(np.clip(base_recline + offset, bounds.recline_angle_min, bounds.recline_angle_max))
            for offset in recline_offsets
        }
    )
    for airbag, kneeairbag, ll_level in bounds.discrete_combinations():
        for delta_pos in delta_values:
            for recline_angle in recline_values:
                candidate = case[FEATURE_COLUMNS].to_dict()
                candidate.update(
                    {
                        "input_airbag": int(airbag),
                        "input_kneeairbag": int(kneeairbag),
                        "input_ll_level": int(ll_level),
                        "input_delta_pos": float(delta_pos),
                        "input_recline_angle": float(recline_angle),
                    }
                )
                rows.append(candidate)
    return pd.DataFrame(rows)


def _predict_candidates_in_chunks(
    surrogate_bundle: SurrogateModelBundle,
    candidates: pd.DataFrame,
    neighbor_k: int,
    chunk_size: int = 20000,
) -> pd.DataFrame:
    """分块预测候选样本，兼顾运行速度与内存占用。"""
    parts: List[pd.DataFrame] = []
    for start in range(0, len(candidates), int(chunk_size)):
        end = min(start + int(chunk_size), len(candidates))
        pred = surrogate_bundle.predict(candidates.iloc[start:end][FEATURE_COLUMNS], neighbor_k=neighbor_k)
        parts.append(pred)
    return pd.concat(parts, axis=0).sort_index()


def _score_candidates_batch(
    pred_df: pd.DataFrame,
    candidates: pd.DataFrame,
    base_cases: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """批量计算候选控制参数的保守风险评分。

    分数越低表示候选越优；其中 mais 和 p_mais_ge3 分别对应预测 MAIS 等级与 AIS3+ 联合风险，uncertainty 和 ood 表示预测可信度惩罚，action_cost 表示相对原始控制参数的调整幅度。
    """
    action_cost = _action_cost_batch(candidates, base_cases)
    score = (
        float(weights.get("mais", 1.0)) * pred_df["pred_MAIS"].astype(float)
        + float(weights.get("p_mais_ge3", 0.5)) * pred_df["pred_P_MAIS_GE3"]
        + float(weights.get("uncertainty", 0.2)) * pred_df["pred_uncertainty"]
        + float(weights.get("ood", 0.2)) * pred_df["pred_ood_score"]
        + float(weights.get("action_cost", 0.05)) * action_cost
    )
    return score.astype(float)


def _action_cost_batch(candidates: pd.DataFrame, base_cases: pd.DataFrame) -> pd.Series:
    """批量计算候选动作相对各自原始控制参数的归一化调整幅度。"""
    case_indices = candidates["_case_eval_index"].to_numpy(dtype=int)
    base = base_cases.iloc[case_indices].reset_index(drop=True)
    current = candidates.reset_index(drop=True)
    cost = (
        (current["input_airbag"] - base["input_airbag"].astype(int)).abs()
        + (current["input_kneeairbag"] - base["input_kneeairbag"].astype(int)).abs()
        + (current["input_ll_level"] - base["input_ll_level"].astype(int)).abs() / 2.0
        + (current["input_delta_pos"] - base["input_delta_pos"].astype(float)).abs() / 284.0
        + (current["input_recline_angle"] - base["input_recline_angle"].astype(float)).abs() / 0.69814
    )
    return pd.Series((cost / 5.0).to_numpy(dtype=float), index=candidates.index)


def _build_result_row(
    case: pd.Series,
    base_pred: pd.Series,
    best_candidate: pd.Series,
    best_pred: pd.Series,
    score: float,
) -> Dict[str, object]:
    """组装单个 case 的 Base/Opt 对比结果。"""
    row: Dict[str, object] = {
        "case_id": int(case["case_id"]),
        "input_velocity": float(case["input_velocity"]),
        "input_angle": float(case["input_angle"]),
        "input_overlap": int(case["input_overlap"]),
        "input_overlap_signed": float(case["input_overlap_signed"]),
        "input_swing_angle": float(case["input_swing_angle"]),
        "input_type_num": int(case["input_type_num"]),
        "True_MAIS": int(case["output_MAIS"]),
        "Opt_Score": float(score),
    }
    for name in CONTROL_COLUMNS:
        row[f"Base_{name.replace('input_', '')}"] = float(case[name])
        row[f"Opt_{name.replace('input_', '')}"] = float(best_candidate[name])
    for metric in ["Amax", "Dmax", "CTI", "HIC", "Nij"]:
        row[f"Base_{metric}"] = float(base_pred[f"pred_{metric}"])
        row[f"Opt_{metric}"] = float(best_pred[f"pred_{metric}"])
        row[f"Reduction_{metric}"] = row[f"Base_{metric}"] - row[f"Opt_{metric}"]
    for metric in ["cti", "hic", "nij"]:
        row[f"Base_{metric}_AIS"] = int(base_pred[f"pred_{metric}_AIS"])
        row[f"Opt_{metric}_AIS"] = int(best_pred[f"pred_{metric}_AIS"])
    row["Base_MAIS"] = int(base_pred["pred_MAIS"])
    row["Opt_MAIS"] = int(best_pred["pred_MAIS"])
    row["Reduction_MAIS"] = int(row["Base_MAIS"] - row["Opt_MAIS"])
    row["Base_P_MAIS_GE3"] = float(base_pred["pred_P_MAIS_GE3"])
    row["Opt_P_MAIS_GE3"] = float(best_pred["pred_P_MAIS_GE3"])
    row["Reduction_P_MAIS_GE3"] = row["Base_P_MAIS_GE3"] - row["Opt_P_MAIS_GE3"]
    row["Opt_Uncertainty"] = float(best_pred["pred_uncertainty"])
    row["Opt_OODScore"] = float(best_pred["pred_ood_score"])
    row["RecommendationFlag"], row["RecommendationRank"] = _recommendation_flag(row)
    return row


def _recommendation_flag(row: Dict[str, object]) -> Tuple[str, int]:
    """依据预测降损和置信度给出推荐标志。"""
    improves = int(row["Reduction_MAIS"]) >= 1 or float(row["Reduction_P_MAIS_GE3"]) > 0.05
    confident = float(row["Opt_Uncertainty"]) <= 0.75 and float(row["Opt_OODScore"]) <= 1.5
    if improves and confident:
        return "推荐", 0
    if improves:
        return "谨慎推荐", 1
    return "不推荐", 2


def _subset_summary(df: pd.DataFrame, name: str) -> Dict[str, object]:
    """对一个样本子集统计预测降损情况。"""
    if len(df) == 0:
        return {
            "name": name,
            "count": 0,
            "mean_reduction_mais": None,
            "mean_reduction_p_mais_ge3": None,
            "p_reduction_mais_ge1": None,
            "mean_base_mais": None,
            "mean_opt_mais": None,
            "mean_base_p_mais_ge3": None,
            "mean_opt_p_mais_ge3": None,
            "mean_opt_uncertainty": None,
            "mean_opt_ood": None,
        }
    return {
        "name": name,
        "count": int(len(df)),
        "mean_reduction_mais": float(df["Reduction_MAIS"].mean()),
        "mean_reduction_p_mais_ge3": float(df["Reduction_P_MAIS_GE3"].mean()),
        "p_reduction_mais_ge1": float((df["Reduction_MAIS"] >= 1).mean()),
        "mean_base_mais": float(df["Base_MAIS"].mean()),
        "mean_opt_mais": float(df["Opt_MAIS"].mean()),
        "mean_base_p_mais_ge3": float(df["Base_P_MAIS_GE3"].mean()),
        "mean_opt_p_mais_ge3": float(df["Opt_P_MAIS_GE3"].mean()),
        "mean_opt_uncertainty": float(df["Opt_Uncertainty"].mean()),
        "mean_opt_ood": float(df["Opt_OODScore"].mean()),
    }
