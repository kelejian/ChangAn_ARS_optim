"""字段约定与参数边界。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


RAW_INPUT_COLUMNS = [
    "input_velocity",
    "input_angle",
    "input_overlap",
    "input_airbag",
    "input_kneeairbag",
    "input_ll_level",
    "input_delta_pos",
    "input_recline_angle",
    "input_swing_angle",
    "input_type_num",
]

# 优化器只允许改变这些约束系统参数，其余输入视为给定工况或乘员状态。
CONTROL_COLUMNS = [
    "input_airbag",
    "input_kneeairbag",
    "input_ll_level",
    "input_delta_pos",
    "input_recline_angle",
]

# FEATURE_COLUMNS 是代理模型实际接收的完整输入，其中包含工况变量、乘员状态变量和可调控制变量。
FEATURE_COLUMNS = [
    "input_velocity",
    "input_angle",
    "input_overlap_signed",
    "input_swing_angle",
    "input_height",
    "input_bmi",
    "input_airbag",
    "input_kneeairbag",
    "input_ll_force",
    "input_ll_enabled",
    "input_delta_pos",
    "input_recline_angle",
]

# 连续变量进入 StandardScaler，类别变量进入 OneHotEncoder。
CONTINUOUS_FEATURES = [
    "input_velocity",
    "input_angle",
    "input_overlap_signed",
    "input_swing_angle",
    "input_height",
    "input_bmi",
    "input_ll_force_effective",
    "input_delta_pos",
    "input_recline_angle",
]

CATEGORICAL_FEATURES = [
    "input_airbag",
    "input_kneeairbag",
    "input_ll_enabled",
]

LL_FORCE_BY_LEVEL: Dict[int, float] = {1: float("inf"), 2: 3.7, 3: 5.0}

SURROGATE_TARGETS = [
    "output_Amax",
    "output_Dmax",
    "output_HIC",
    "output_Nij",
]

DERIVED_CONTINUOUS_TARGETS = [
    "output_CTI",
]

# CONTINUOUS_TARGETS 包含直接回归目标和公式派生指标，用于数据检查、评估和报告。
CONTINUOUS_TARGETS = SURROGATE_TARGETS + DERIVED_CONTINUOUS_TARGETS

# AIS_TARGETS 是由 HIC/CTI/Nij 经项目参考公式派生的等级列，用于评估和报告，不作为代理模型回归目标。
AIS_TARGETS = [
    "output_cti_AIS",
    "output_hic_AIS",
    "output_nij_AIS",
]

OVERLAP_SIGNED_MAP: Dict[int, float] = {
    1: 1.00,
    2: 0.75,
    4: 0.50,
    6: 0.25,
    3: -0.75,
    5: -0.50,
    7: -0.25,
}


@dataclass(frozen=True)
class ControlBounds:
    """可调控制变量的取值边界。

    离散变量通过全组合枚举生成候选，座椅位置和靠背角在原始值附近生成粗网格候选后再裁剪到这里定义的范围内。
    """

    airbag_values: Tuple[int, ...] = (0, 1)
    kneeairbag_values: Tuple[int, ...] = (0, 1)
    ll_level_values: Tuple[int, ...] = (1, 2, 3)
    delta_pos_min: float = -142.0
    delta_pos_max: float = 142.0
    recline_angle_min: float = -0.34907
    recline_angle_max: float = 0.34907

    def discrete_combinations(self) -> List[Tuple[int, int, int]]:
        """返回气囊、膝部气囊和安全带限力的离散组合。"""
        return [
            (airbag, kneeairbag, ll_level)
            for airbag in self.airbag_values
            for kneeairbag in self.kneeairbag_values
            for ll_level in self.ll_level_values
        ]
