"""连续损伤指标与 AIS/MAIS 的经验公式映射。

本模块依据 reference_doc/changan_Injury_Criteria_AIS_Cal_original.py 中的计算逻辑实现。代理模型直接预测 Amax、Dmax、HIC 和 Nij，CTI 由 Amax、Dmax 与乘员体型参数按参考公式派生，AIS 与 MAIS 再由 HIC、CTI、Nij 映射得到。
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


MASS_BY_TYPE = {
    1: 53.76,
    2: 60.16,
    3: 66.56,
    4: 72.96,
    5: 60.69,
    6: 67.915,
    7: 75.14,
    8: 82.365,
    9: 68.04,
    10: 76.14,
    11: 84.24,
    12: 92.34,
}

CHEST_DEPTH_BY_TYPE = {
    1: 210.981,
    2: 224.122,
    3: 241.008,
    4: 252.976,
    5: 217.745,
    6: 232.555,
    7: 247.132,
    8: 262.192,
    9: 222.327,
    10: 238.309,
    11: 255.163,
    12: 268.576,
}


def calculate_cti(
    amax: Iterable[float],
    dmax: Iterable[float],
    type_num: Iterable[int],
) -> np.ndarray:
    """按参考脚本公式由 Amax、Dmax 和乘员体型编号计算 CTI。"""
    amax_value = np.atleast_1d(np.asarray(amax, dtype=float))
    dmax_value = np.atleast_1d(np.asarray(dmax, dtype=float))
    type_value = np.atleast_1d(np.asarray(type_num, dtype=int))
    mass = _lookup_type_values(type_value, MASS_BY_TYPE, "mass")
    chest_depth = _lookup_type_values(type_value, CHEST_DEPTH_BY_TYPE, "chest_depth")
    a_int = 90.0 * mass / 77.0
    d_int = chest_depth / 255.594 * 103.0
    return amax_value / a_int + dmax_value / d_int


def injury_levels_and_risk(hic: Iterable[float], cti: Iterable[float], nij: Iterable[float]) -> pd.DataFrame:
    """根据项目参考脚本的经验公式计算分项 AIS、MAIS 和 AIS3+ 联合风险。"""
    hic_probs = _head_hic_probabilities(hic)
    cti_probs = _chest_cti_probabilities(cti)
    nij_probs = _neck_nij_probabilities(nij)

    result = pd.DataFrame(
        {
            "hic_AIS": _ais_from_ordered_probabilities(hic_probs, first_level=1),
            "cti_AIS": _ais_from_ordered_probabilities(cti_probs, first_level=2),
            "nij_AIS": _ais_from_ordered_probabilities(nij_probs, first_level=2),
            "P_hic_AIS_GE3": hic_probs[:, 2],
            "P_cti_AIS_GE3": cti_probs[:, 1],
            "P_nij_AIS_GE3": nij_probs[:, 1],
        }
    )
    result["MAIS"] = np.maximum.reduce(
        [
            result["hic_AIS"].to_numpy(dtype=int),
            result["cti_AIS"].to_numpy(dtype=int),
            result["nij_AIS"].to_numpy(dtype=int),
        ]
    )
    # 将头/胸/颈 AIS3+ 概率合成为单一严重损伤风险项，用于优化评分中的高等级损伤惩罚。
    result["P_MAIS_GE3"] = 1.0 - (
        (1.0 - result["P_hic_AIS_GE3"])
        * (1.0 - result["P_cti_AIS_GE3"])
        * (1.0 - result["P_nij_AIS_GE3"])
    )
    return result


def append_ais_from_injury_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """根据预测或派生的 HIC/CTI/Nij 计算对应 AIS、MAIS 和严重损伤风险。"""
    result = pred_df.copy()
    mapped = injury_levels_and_risk(result["pred_HIC"], result["pred_CTI"], result["pred_Nij"])
    result["pred_hic_AIS"] = mapped["hic_AIS"].to_numpy(dtype=int)
    result["pred_cti_AIS"] = mapped["cti_AIS"].to_numpy(dtype=int)
    result["pred_nij_AIS"] = mapped["nij_AIS"].to_numpy(dtype=int)
    result["pred_MAIS"] = mapped["MAIS"].to_numpy(dtype=int)
    result["pred_P_hic_AIS_GE3"] = mapped["P_hic_AIS_GE3"].to_numpy(dtype=float)
    result["pred_P_cti_AIS_GE3"] = mapped["P_cti_AIS_GE3"].to_numpy(dtype=float)
    result["pred_P_nij_AIS_GE3"] = mapped["P_nij_AIS_GE3"].to_numpy(dtype=float)
    result["pred_P_MAIS_GE3"] = mapped["P_MAIS_GE3"].to_numpy(dtype=float)
    return result


def _head_hic_probabilities(hic: Iterable[float]) -> np.ndarray:
    """计算 HIC 对应的 AIS1+ 至 AIS5+ 概率。"""
    value = np.clip(np.atleast_1d(np.asarray(hic, dtype=float)), 1e-6, None)
    exponents = (
        1.54 + 200.0 / value - 0.0065 * value,
        2.49 + 200.0 / value - 0.00483 * value,
        3.39 + 200.0 / value - 0.00372 * value,
        4.90 + 200.0 / value - 0.00351 * value,
        7.82 + 200.0 / value - 0.00429 * value,
    )
    return _stack_logistic_probabilities(exponents)


def _chest_cti_probabilities(cti: Iterable[float]) -> np.ndarray:
    """计算 CTI 对应的 AIS2+ 至 AIS5+ 概率。"""
    value = np.atleast_1d(np.asarray(cti, dtype=float))
    exponents = (
        4.870 - 6.036 * value,
        8.224 - 7.125 * value,
        9.872 - 7.125 * value,
        14.242 - 6.589 * value,
    )
    return _stack_logistic_probabilities(exponents)


def _neck_nij_probabilities(nij: Iterable[float]) -> np.ndarray:
    """计算 Nij 对应的 AIS2+ 至 AIS5+ 概率。"""
    value = np.atleast_1d(np.asarray(nij, dtype=float))
    exponents = (
        2.054 - 1.195 * value,
        3.227 - 1.969 * value,
        2.693 - 1.195 * value,
        3.817 - 1.195 * value,
    )
    return _stack_logistic_probabilities(exponents)


def _stack_logistic_probabilities(exponents: Tuple[np.ndarray, ...]) -> np.ndarray:
    """将参考代码中的 1 / (1 + exp(exponent)) 公式批量转换为概率矩阵。"""
    stacked = np.stack([np.asarray(item, dtype=float) for item in exponents], axis=1)
    return 1.0 / (1.0 + np.exp(np.clip(stacked, -700.0, 700.0)))


def _ais_from_ordered_probabilities(probabilities: np.ndarray, first_level: int) -> np.ndarray:
    """按参考代码逻辑，从最高等级向低等级回退，选择首个概率不小于 0.5 的 AIS 等级。"""
    ais = np.zeros(probabilities.shape[0], dtype=int)
    for col_idx in range(probabilities.shape[1] - 1, -1, -1):
        level = int(first_level + col_idx)
        ais = np.where((ais == 0) & (probabilities[:, col_idx] >= 0.5), level, ais)
    return ais


def _lookup_type_values(type_num: np.ndarray, table: dict[int, float], name: str) -> np.ndarray:
    """按乘员体型编号查询参考脚本中的体型参数。"""
    values = []
    for item in type_num:
        key = int(item)
        if key not in table:
            raise ValueError(f"input_type_num 存在未定义编号，无法查询 {name}: {key}")
        values.append(table[key])
    return np.asarray(values, dtype=float)
