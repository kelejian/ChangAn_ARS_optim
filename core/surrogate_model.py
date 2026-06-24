"""轻量化损伤风险代理模型训练与推理。

本模块使用结构化表格代理模型替代耗时仿真，在给定工况和约束系统参数后快速预测基础连续损伤响应。CTI 由 Amax、Dmax 和乘员体型参数按参考公式派生，AIS 与 MAIS 在推理后由经验公式映射得到。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
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

from core.injury_mapping import append_ais_from_injury_metrics, calculate_cti
from core.schema import (
    AIS_TARGETS,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    CONTINUOUS_TARGETS,
    FEATURE_COLUMNS,
    SURROGATE_TARGETS,
)


def _make_preprocessor() -> ColumnTransformer:
    """构造结构化特征预处理器。"""
    return ColumnTransformer(
        transformers=[
            # 连续变量标准化后进入树模型，便于近邻距离指标具有统一尺度。
            ("continuous", StandardScaler(), CONTINUOUS_FEATURES),
            # 类别变量使用独热编码；handle_unknown=ignore 允许后续推理时出现训练集中未覆盖的类别。
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


@dataclass
class SurrogateModelBundle:
    """代理模型包，包含连续损伤回归器和训练分布近邻参考。

    该对象是后续寻优阶段的统一推理入口，优化器只需要调用 predict 即可得到连续损伤预测、公式映射 AIS/MAIS、不确定性和分布外程度。
    """

    preprocessor: ColumnTransformer
    continuous_regressor: ExtraTreesRegressor
    neighbor_index: NearestNeighbors
    ood_scale: float
    target_scale: np.ndarray
    feature_columns: List[str]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """将输入表转换为代理模型特征矩阵。"""
        return self.preprocessor.transform(df[self.feature_columns])

    def predict(self, df: pd.DataFrame, neighbor_k: int = 5) -> pd.DataFrame:
        """对一批候选样本输出连续损伤、公式映射 AIS/MAIS 和置信度指标。"""
        x = self.transform(df)
        continuous_pred = self.continuous_regressor.predict(x)
        tree_pred = np.stack([tree.predict(x) for tree in self.continuous_regressor.estimators_], axis=0)
        # 树间离散程度用于近似代理模型不确定性；按训练集目标标准差归一化后，各损伤指标可合并比较。
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
        output["pred_CTI"] = calculate_cti(output["pred_Amax"], output["pred_Dmax"], df["input_type_num"])
        output = append_ais_from_injury_metrics(output)
        output["pred_uncertainty"] = uncertainty
        output["pred_ood_score"] = ood_score
        return output


def train_surrogate_model_bundle(
    train_df: pd.DataFrame,
    config: Dict[str, object],
) -> SurrogateModelBundle:
    """训练结构化代理模型与训练分布近邻参考。"""
    surrogate_cfg = config.get("surrogate_model", {}) or {}
    base_kwargs = {
        "n_estimators": int(surrogate_cfg.get("n_estimators", 160)),
        "min_samples_leaf": int(surrogate_cfg.get("min_samples_leaf", 2)),
        "max_features": surrogate_cfg.get("max_features", "sqrt"),
        "random_state": int(config.get("seed", 2026)),
        "n_jobs": int(surrogate_cfg.get("n_jobs", -1)),
    }

    preprocessor = _make_preprocessor()
    x_train = preprocessor.fit_transform(train_df[FEATURE_COLUMNS])

    # 代理模型只回归基础连续损伤响应；CTI 在推理阶段由 Amax、Dmax 和乘员体型编号派生。
    continuous_regressor = ExtraTreesRegressor(**base_kwargs)
    continuous_regressor.fit(x_train, train_df[SURROGATE_TARGETS])
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
        continuous_regressor=continuous_regressor,
        neighbor_index=neighbor_index,
        ood_scale=max(ood_scale, 1e-9),
        target_scale=target_scale,
        feature_columns=list(FEATURE_COLUMNS),
    )


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
