"""轻量化损伤风险代理模型训练与推理。

本模块使用结构化表格代理模型替代耗时仿真，在给定工况和约束系统参数后快速预测基础连续损伤响应。CTI 由 Amax、Dmax 和乘员体型参数按参考公式派生，AIS 与 MAIS 在推理后由经验公式映射得到。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.base import RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from core.injury_mapping import append_ais_from_injury_metrics, calculate_chest_depth, calculate_cti_from_physical
from core.schema import (
    AIS_TARGETS,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    CONTINUOUS_TARGETS,
    FEATURE_COLUMNS,
    SURROGATE_TARGETS,
)


COMPACT_DERIVED_FEATURES = [
    "input_overlap_abs",
    "impact_angle_abs",
    "velocity_sq",
    "velocity_overlap_exposure",
    "angle_overlap_coupling",
    "velocity_airbag_coupling",
    "velocity_ll_coupling",
    "occupant_mass",
    "occupant_chest_depth",
]

EXPANDED_DERIVED_FEATURES = COMPACT_DERIVED_FEATURES + [
    "swing_angle_abs",
    "delta_pos_sq",
    "recline_angle_sq",
    "velocity_angle_exposure",
    "velocity_swing_exposure",
    "velocity_kneeairbag_coupling",
    "velocity_recline_coupling",
    "delta_recline_coupling",
]


def _make_preprocessor(feature_engineering: str) -> ColumnTransformer:
    """构造结构化特征预处理器。"""
    derived_features = _derived_feature_names(feature_engineering)
    return ColumnTransformer(
        transformers=[
            # 连续变量标准化后进入树模型，便于近邻距离指标具有统一尺度。
            ("continuous", StandardScaler(), CONTINUOUS_FEATURES + derived_features),
            # 类别变量使用独热编码；handle_unknown=ignore 允许后续推理时出现训练集中未覆盖的类别。
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


@dataclass
class SurrogateModelBundle:
    """代理模型包，包含目标特异回归器、不确定性参考和训练分布近邻索引。

    该对象是后续寻优阶段的统一推理入口，优化器只需要调用 predict 即可得到连续损伤预测、公式映射 AIS/MAIS、不确定性和分布外程度。
    """

    preprocessor: ColumnTransformer
    target_regressors: Dict[str, List[RegressorMixin]]
    target_weights: Dict[str, np.ndarray]
    prediction_scale: np.ndarray
    prediction_offset: np.ndarray
    uncertainty_regressor: ExtraTreesRegressor
    neighbor_index: NearestNeighbors
    ood_scale: float
    target_scale: np.ndarray
    feature_columns: List[str]
    feature_engineering: str

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """将输入表转换为代理模型特征矩阵。"""
        feature_frame = _engineer_features(df[self.feature_columns], self.feature_engineering)
        return self.preprocessor.transform(feature_frame)

    def predict(self, df: pd.DataFrame, neighbor_k: int = 5) -> pd.DataFrame:
        """对一批候选样本输出连续损伤、公式映射 AIS/MAIS 和置信度指标。"""
        x = self.transform(df)
        continuous_pred = np.column_stack([
            np.column_stack([model.predict(x) for model in self.target_regressors[target]])
            @ self.target_weights[target]
            for target in SURROGATE_TARGETS
        ])
        # 目标级仿射校准用于调整集成预测的整体尺度和偏移，CTI 与 AIS 仍由校准后的基础响应按参考公式派生。
        continuous_pred = continuous_pred * self.prediction_scale + self.prediction_offset
        # 四项基础损伤响应的物理定义均为非负量。
        continuous_pred = np.maximum(continuous_pred, 0.0)
        tree_pred = np.stack([tree.predict(x) for tree in self.uncertainty_regressor.estimators_], axis=0)
        # 独立的随机树集合用于统一估计四个目标的局部预测离散程度。
        uncertainty = (tree_pred.std(axis=0) / self.target_scale).mean(axis=1)

        # OOD 分数衡量候选样本与训练分布的距离，数值越大表示代理模型外推风险越高。
        effective_neighbor_k = max(1, min(int(neighbor_k), int(self.neighbor_index.n_samples_fit_)))
        distances, _ = self.neighbor_index.kneighbors(x, n_neighbors=effective_neighbor_k)
        ood_score = distances.mean(axis=1) / max(self.ood_scale, 1e-9)

        output = pd.DataFrame(
            continuous_pred,
            columns=[name.replace("output_", "pred_") for name in SURROGATE_TARGETS],
            index=df.index,
        )
        output["pred_CTI"] = calculate_cti_from_physical(
            output["pred_Amax"], output["pred_Dmax"], df["input_height"], df["input_bmi"]
        )
        output = append_ais_from_injury_metrics(output)
        output["pred_uncertainty"] = uncertainty
        output["pred_ood_score"] = ood_score
        return output


def train_surrogate_model_bundle(
    train_df: pd.DataFrame,
    config: Dict[str, object],
) -> SurrogateModelBundle:
    """训练目标特异代理模型、不确定性参考和训练分布近邻索引。"""
    surrogate_cfg = config.get("surrogate_model", {}) or {}
    target_model_cfg = surrogate_cfg.get("target_models")
    if not isinstance(target_model_cfg, Mapping):
        raise ValueError("surrogate_model.target_models 必须为按目标字段组织的模型配置。")
    missing_targets = [target for target in SURROGATE_TARGETS if target not in target_model_cfg]
    if missing_targets:
        raise ValueError(f"代理模型配置缺少目标字段: {missing_targets}")

    seed = int(config.get("seed", 2026))
    n_jobs = int(surrogate_cfg.get("n_jobs", -1))

    feature_engineering = str(surrogate_cfg.get("feature_engineering", "base"))
    preprocessor = _make_preprocessor(feature_engineering)
    train_features = _engineer_features(train_df[FEATURE_COLUMNS], feature_engineering)
    x_train = preprocessor.fit_transform(train_features)

    # 四个基础损伤响应的量纲和可预测性差异较大，分别使用实验对比后确定的回归器。
    target_regressors: Dict[str, List[RegressorMixin]] = {}
    target_weights: Dict[str, np.ndarray] = {}
    for target in SURROGATE_TARGETS:
        component_configs = target_model_cfg[target]
        if not isinstance(component_configs, list) or not component_configs:
            raise ValueError(f"{target} 的模型配置必须为非空列表。")
        regressors: List[RegressorMixin] = []
        weights: List[float] = []
        for model_cfg in component_configs:
            if not isinstance(model_cfg, Mapping):
                raise ValueError(f"{target} 的每个模型分量必须为映射。")
            regressor = _build_target_regressor(model_cfg, seed=seed, n_jobs=n_jobs)
            regressor.fit(x_train, train_df[target].to_numpy(dtype=float))
            regressors.append(regressor)
            weights.append(float(model_cfg.get("weight", 1.0)))
        weight_array = np.asarray(weights, dtype=float)
        if np.any(weight_array < 0.0) or weight_array.sum() <= 0.0:
            raise ValueError(f"{target} 的模型权重必须为非负数且总和大于零。")
        target_regressors[target] = regressors
        target_weights[target] = weight_array / weight_array.sum()

    calibration_cfg = surrogate_cfg.get("prediction_calibration", {}) or {}
    prediction_scale = np.asarray(
        [float((calibration_cfg.get(target, {}) or {}).get("scale", 1.0)) for target in SURROGATE_TARGETS],
        dtype=float,
    )
    prediction_offset = np.asarray(
        [float((calibration_cfg.get(target, {}) or {}).get("offset", 0.0)) for target in SURROGATE_TARGETS],
        dtype=float,
    )

    uncertainty_cfg = surrogate_cfg.get("uncertainty_model", {}) or {}
    uncertainty_regressor = ExtraTreesRegressor(
        n_estimators=int(uncertainty_cfg.get("n_estimators", 160)),
        min_samples_leaf=int(uncertainty_cfg.get("min_samples_leaf", 2)),
        max_features=uncertainty_cfg.get("max_features", "sqrt"),
        random_state=seed,
        n_jobs=n_jobs,
    )
    uncertainty_regressor.fit(x_train, train_df[SURROGATE_TARGETS])
    target_scale = train_df[SURROGATE_TARGETS].to_numpy(dtype=float).std(axis=0)
    target_scale = np.where(target_scale < 1e-9, 1.0, target_scale)

    neighbor_k = int((config.get("optimization", {}) or {}).get("neighbor_k", 5))
    neighbor_k = max(1, min(neighbor_k, len(train_df)))
    neighbor_index = NearestNeighbors(n_neighbors=neighbor_k)
    neighbor_index.fit(x_train)
    distances, _ = neighbor_index.kneighbors(x_train, n_neighbors=neighbor_k)
    # 以训练集内部近邻平均距离的中位数作为尺度，使 OOD 分数具有相对可比性。
    ood_scale = float(np.median(distances.mean(axis=1)))
    return SurrogateModelBundle(
        preprocessor=preprocessor,
        target_regressors=target_regressors,
        target_weights=target_weights,
        prediction_scale=prediction_scale,
        prediction_offset=prediction_offset,
        uncertainty_regressor=uncertainty_regressor,
        neighbor_index=neighbor_index,
        ood_scale=max(ood_scale, 1e-9),
        target_scale=target_scale,
        feature_columns=list(FEATURE_COLUMNS),
        feature_engineering=feature_engineering,
    )


def _derived_feature_names(feature_engineering: str) -> List[str]:
    """返回指定内部特征工程方案对应的派生连续字段。"""
    if feature_engineering == "base":
        return []
    if feature_engineering == "compact_interactions":
        return list(COMPACT_DERIVED_FEATURES)
    if feature_engineering == "expanded_interactions":
        return list(EXPANDED_DERIVED_FEATURES)
    raise ValueError(f"不支持的内部特征工程方案: {feature_engineering!r}")


def _engineer_features(df: pd.DataFrame, feature_engineering: str) -> pd.DataFrame:
    """由既有输入字段计算内部派生特征，保持外部数据接口不变。"""
    result = df.copy()
    # 未启用限力时以有限区间中点填充数值特征，启用状态由独立类别特征表达，避免无穷值进入预处理器。
    result["input_ll_force_effective"] = np.where(
        result["input_ll_enabled"].astype(int) == 1,
        result["input_ll_force"].astype(float),
        4.35,
    )
    derived_features = _derived_feature_names(feature_engineering)
    if not derived_features:
        return result

    velocity = result["input_velocity"].astype(float)
    angle = result["input_angle"].astype(float)
    overlap = result["input_overlap_signed"].astype(float)
    swing = result["input_swing_angle"].astype(float)
    delta_pos = result["input_delta_pos"].astype(float)
    recline = result["input_recline_angle"].astype(float)
    result["input_overlap_abs"] = overlap.abs()
    result["impact_angle_abs"] = angle.abs()
    result["velocity_sq"] = velocity.pow(2)
    result["velocity_overlap_exposure"] = velocity * overlap.abs()
    result["angle_overlap_coupling"] = angle * overlap
    result["velocity_airbag_coupling"] = velocity * result["input_airbag"].astype(float)
    result["velocity_ll_coupling"] = (
        velocity * result["input_ll_force_effective"] * result["input_ll_enabled"].astype(float)
    )
    result["occupant_mass"] = result["input_height"].astype(float).pow(2) * result["input_bmi"].astype(float)
    result["occupant_chest_depth"] = calculate_chest_depth(result["input_height"], result["input_bmi"])

    if feature_engineering == "expanded_interactions":
        result["swing_angle_abs"] = swing.abs()
        result["delta_pos_sq"] = delta_pos.pow(2)
        result["recline_angle_sq"] = recline.pow(2)
        result["velocity_angle_exposure"] = velocity * angle.abs()
        result["velocity_swing_exposure"] = velocity * swing.abs()
        result["velocity_kneeairbag_coupling"] = velocity * result["input_kneeairbag"].astype(float)
        result["velocity_recline_coupling"] = velocity * recline
        result["delta_recline_coupling"] = delta_pos * recline
    return result


def _build_target_regressor(
    model_cfg: Mapping[str, object],
    seed: int,
    n_jobs: int,
) -> RegressorMixin:
    """根据配置构造一个基础连续损伤指标的回归器。"""
    model_type = str(model_cfg.get("type", ""))
    if model_type.endswith("_log"):
        base_config = dict(model_cfg)
        base_config["type"] = model_type.removesuffix("_log")
        base_regressor = _build_target_regressor(base_config, seed=seed, n_jobs=n_jobs)
        return TransformedTargetRegressor(
            regressor=base_regressor,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    if model_type == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(model_cfg.get("n_estimators", 600)),
            min_samples_leaf=int(model_cfg.get("min_samples_leaf", 2)),
            max_features=model_cfg.get("max_features", 0.7),
            random_state=seed,
            n_jobs=n_jobs,
        )
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(model_cfg.get("n_estimators", 600)),
            min_samples_leaf=int(model_cfg.get("min_samples_leaf", 2)),
            max_features=model_cfg.get("max_features", 0.7),
            random_state=seed,
            n_jobs=n_jobs,
        )
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=float(model_cfg.get("learning_rate", 0.05)),
            max_iter=int(model_cfg.get("max_iter", 500)),
            max_leaf_nodes=int(model_cfg.get("max_leaf_nodes", 31)),
            min_samples_leaf=int(model_cfg.get("min_samples_leaf", 12)),
            l2_regularization=float(model_cfg.get("l2_regularization", 0.5)),
            random_state=seed,
        )
    if model_type == "svr":
        return TransformedTargetRegressor(
            regressor=SVR(
                C=float(model_cfg.get("C", 3.0)),
                gamma=model_cfg.get("gamma", "scale"),
                epsilon=float(model_cfg.get("epsilon", 0.05)),
            ),
            transformer=StandardScaler(),
        )
    if model_type == "xgboost":
        return XGBRegressor(
            n_estimators=int(model_cfg.get("n_estimators", 650)),
            max_depth=int(model_cfg.get("max_depth", 5)),
            learning_rate=float(model_cfg.get("learning_rate", 0.02)),
            min_child_weight=float(model_cfg.get("min_child_weight", 8.0)),
            subsample=float(model_cfg.get("subsample", 0.85)),
            colsample_bytree=float(model_cfg.get("colsample_bytree", 0.85)),
            reg_alpha=float(model_cfg.get("reg_alpha", 0.05)),
            reg_lambda=float(model_cfg.get("reg_lambda", 5.0)),
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=n_jobs,
        )
    raise ValueError(f"不支持的代理模型类型: {model_type!r}")


def evaluate_surrogate_model_bundle(
    bundle: SurrogateModelBundle,
    df: pd.DataFrame,
    neighbor_k: int = 5,
) -> Dict[str, object]:
    """在固定数据切片上评估代理模型。"""
    if len(df) == 0:
        raise ValueError("评估切片为空，无法计算代理模型指标。")
    pred = bundle.predict(df, neighbor_k=neighbor_k)
    metrics: Dict[str, object] = {
        "sample_count": int(len(df)),
    }
    for target in CONTINUOUS_TARGETS:
        pred_col = target.replace("output_", "pred_")
        y_true = df[target].to_numpy(dtype=float)
        y_pred = pred[pred_col].to_numpy(dtype=float)
        metrics[target] = {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    y_true_mais = df["output_MAIS"].to_numpy(dtype=int)
    y_pred_mais = pred["pred_MAIS"].to_numpy(dtype=int)
    # 常数基线表示始终预测当前切片中最常见 MAIS 等级，用于判断代理模型等级判别是否优于简单规则。
    constant = int(pd.Series(y_true_mais).mode().iloc[0])
    constant_pred = np.full_like(y_true_mais, constant)
    mais_metrics = _classification_metrics(y_true_mais, y_pred_mais, labels=list(range(6)))
    mais_metrics.update(
        {
            "constant_baseline_class": int(constant),
            "constant_baseline_accuracy": float(accuracy_score(y_true_mais, constant_pred)),
            "beats_constant_baseline_accuracy": bool(
                accuracy_score(y_true_mais, y_pred_mais)
                > accuracy_score(y_true_mais, constant_pred)
            ),
        }
    )
    metrics["MAIS"] = mais_metrics
    metrics["MAIS_GE3"] = _classification_metrics(
        (y_true_mais >= 3).astype(int),
        (y_pred_mais >= 3).astype(int),
        labels=[0, 1],
    )
    for target in AIS_TARGETS:
        short = target.replace("output_", "").replace("_AIS", "")
        pred_col = f"pred_{short}_AIS"
        y_true = df[target].to_numpy(dtype=int)
        y_pred = pred[pred_col].to_numpy(dtype=int)
        metrics[target] = _classification_metrics(y_true, y_pred, labels=list(range(6)))
    return metrics


def build_surrogate_prediction_table(
    bundle: SurrogateModelBundle,
    data_slices: Mapping[str, pd.DataFrame],
    neighbor_k: int = 5,
) -> pd.DataFrame:
    """导出代理模型逐样本预测明细，供后续绘制散点图和混淆矩阵使用。"""
    tables: List[pd.DataFrame] = []
    context_columns = [
        "case_id",
        "input_velocity",
        "input_angle",
        "input_overlap",
        "input_overlap_signed",
        "input_swing_angle",
        "input_type_num",
        "input_height",
        "input_bmi",
    ]
    for data_scope, df in data_slices.items():
        if len(df) == 0:
            continue
        current = df.reset_index(drop=True)
        pred = bundle.predict(current, neighbor_k=neighbor_k).reset_index(drop=True)
        table = current[context_columns].copy()
        table.insert(0, "data_scope", str(data_scope))

        for target in CONTINUOUS_TARGETS:
            short_name = target.replace("output_", "")
            table[f"true_{short_name}"] = current[target].to_numpy(dtype=float)
            table[f"pred_{short_name}"] = pred[f"pred_{short_name}"].to_numpy(dtype=float)

        for target in AIS_TARGETS:
            short_name = target.replace("output_", "")
            table[f"true_{short_name}"] = current[target].to_numpy(dtype=int)
            table[f"pred_{short_name}"] = pred[f"pred_{short_name}"].to_numpy(dtype=int)

        table["true_MAIS"] = current["output_MAIS"].to_numpy(dtype=int)
        table["pred_MAIS"] = pred["pred_MAIS"].to_numpy(dtype=int)
        table["true_MAIS_GE3"] = (table["true_MAIS"].to_numpy(dtype=int) >= 3).astype(int)
        table["pred_MAIS_GE3"] = (table["pred_MAIS"].to_numpy(dtype=int) >= 3).astype(int)
        table["pred_P_MAIS_GE3"] = pred["pred_P_MAIS_GE3"].to_numpy(dtype=float)
        table["pred_uncertainty"] = pred["pred_uncertainty"].to_numpy(dtype=float)
        table["pred_ood_score"] = pred["pred_ood_score"].to_numpy(dtype=float)
        tables.append(table)

    return pd.concat(tables, ignore_index=True)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> Dict[str, object]:
    """计算离散损伤等级分类指标，不使用连续误差指标评价等级标签。"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }
