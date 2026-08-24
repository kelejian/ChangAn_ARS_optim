"""乘员损伤预测代理模型的本地可视化界面。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, Mapping

import joblib
import pandas as pd
AIRBAG_STATES = {"启用": 1, "未启用": 0}
DEFAULT_EXAMPLE = {
    "velocity": "30.0",
    "impact_angle": "-10.0",
    "overlap_percent": "100",
    "swing_angle_deg": "3.0",
    "height": "1.8",
    "bmi": "23.5",
    "airbag": "启用",
    "knee_airbag": "未启用",
    "load_limiter": "5.0",
    "seat_position": "20.0",
    "recline_angle_deg": "12.0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the injury surrogate prediction GUI.")
    parser.add_argument("--model", type=str, default=None, help="可选的代理模型 joblib 文件路径。")
    parser.add_argument("--check", action="store_true", help="加载模型并运行内置样例后退出。")
    return parser.parse_args()


def resolve_model_path(project_dir: Path, requested_path: str | None) -> Path:
    """解析模型路径；未指定时使用 outputs 下最近生成的代理模型。"""
    if requested_path:
        path = Path(requested_path)
        if not path.is_absolute():
            path = project_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"代理模型文件不存在: {path}")
        return path.resolve()

    candidates = list(project_dir.glob("outputs/*/models/surrogate_model_bundle.joblib"))
    if not candidates:
        raise FileNotFoundError("未找到代理模型，请先运行 run_pipeline.py 或使用 --model 指定 joblib 文件。")
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def parse_load_limiter(value: object) -> float:
    """解析安全带限力值，inf 表示不限力。"""
    text = str(value).strip().lower()
    if text in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    try:
        force = float(text)
    except ValueError as error:
        raise ValueError("安全带限力请输入 3.7–5.0 kN 范围内的数值，未启用限力时输入 inf。") from error
    if not 3.7 <= force <= 5.0:
        raise ValueError("安全带限力应在 3.7–5.0 kN 范围内，未启用限力时输入 inf。")
    return force


def build_model_input(values: Mapping[str, object]) -> pd.DataFrame:
    """校验工程物理量并生成单条代理模型输入。"""
    velocity = float(values["velocity"])
    impact_angle = float(values["impact_angle"])
    overlap_percent = float(values["overlap_percent"])
    swing_angle_deg = float(values["swing_angle_deg"])
    height = float(values["height"])
    bmi = float(values["bmi"])
    seat_position = float(values["seat_position"])
    recline_angle_deg = float(values["recline_angle_deg"])

    checks = (
        (25.0 <= velocity <= 65.0, "碰撞速度应在 25–65 km/h 范围内。"),
        (-30.0 <= impact_angle <= 30.0, "碰撞角度应在 -30°–30° 范围内。"),
        (-100.0 <= overlap_percent <= 100.0 and overlap_percent != 0.0, "碰撞重叠率应在 -100%–100% 范围内且不能为 0。"),
        (-8.60 <= swing_angle_deg <= 8.60, "乘员左右摆角应在约 -8.60°–8.60° 范围内。"),
        (1.60 <= height <= 1.80, "乘员身高应在 1.60–1.80 m 范围内。"),
        (21.0 <= bmi <= 28.5, "乘员 BMI 应在 21.0–28.5 kg/m² 范围内。"),
        (-142.0 <= seat_position <= 142.0, "座椅纵向位置参数应在 -142–142 mm 范围内。"),
        (-20.0 <= recline_angle_deg <= 20.0, "座椅前后仰角应在约 -20°–20° 范围内。"),
    )
    for condition, message in checks:
        if not condition:
            raise ValueError(message)

    try:
        airbag = AIRBAG_STATES[str(values["airbag"])]
        knee_airbag = AIRBAG_STATES[str(values["knee_airbag"])]
    except KeyError as error:
        raise ValueError(f"未识别的约束系统状态: {error.args[0]}") from error

    limiter_force = parse_load_limiter(values["load_limiter"])
    return pd.DataFrame(
        [{
            "input_velocity": velocity,
            "input_angle": impact_angle,
            "input_overlap_signed": overlap_percent / 100.0,
            "input_swing_angle": math.radians(swing_angle_deg),
            "input_height": height,
            "input_bmi": bmi,
            "input_airbag": airbag,
            "input_kneeairbag": knee_airbag,
            "input_ll_force": limiter_force,
            "input_delta_pos": seat_position,
            "input_recline_angle": math.radians(recline_angle_deg),
        }]
    )


def predict_case(bundle: object, values: Mapping[str, object]) -> Dict[str, float | int]:
    """计算连续损伤指标和损伤等级。"""
    predictions = bundle.predict(build_model_input(values)).iloc[0]
    return {
        "Amax": float(predictions["pred_Amax"]),
        "Dmax": float(predictions["pred_Dmax"]),
        "CTI": float(predictions["pred_CTI"]),
        "HIC": float(predictions["pred_HIC"]),
        "Nij": float(predictions["pred_Nij"]),
        "cti_AIS": int(predictions["pred_cti_AIS"]),
        "hic_AIS": int(predictions["pred_hic_AIS"]),
        "nij_AIS": int(predictions["pred_nij_AIS"]),
        "MAIS": int(predictions["pred_MAIS"]),
    }


class InjurySurrogateApp:
    """组织输入表单、代理模型预测和损伤等级图形展示。"""

    def __init__(self, root: tk.Tk, bundle: object, model_path: Path) -> None:
        self.root = root
        self.bundle = bundle
        self.model_path = model_path
        self.entries: Dict[str, tk.StringVar] = {}
        self.scalar_values: Dict[str, tk.StringVar] = {}
        self.ais_values: Dict[str, tk.StringVar] = {}
        self.last_levels: Dict[str, int] = {}
        self._configure_window()
        self._build_layout()

    def _configure_window(self) -> None:
        self.root.title("乘员损伤预测代理模型")
        self.root.geometry("1180x760")
        self.root.minsize(1000, 700)
        self.root.option_add("*Font", ("Microsoft YaHei UI", 10))
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 20, "bold"), foreground="#17324d")
        style.configure("Subtitle.TLabel", font=("Microsoft YaHei UI", 10), foreground="#52606d")
        style.configure("Section.TLabelframe.Label", font=("Microsoft YaHei UI", 11, "bold"), foreground="#17324d")
        style.configure("MetricName.TLabel", font=("Microsoft YaHei UI", 9), foreground="#263746")
        style.configure("MetricCode.TLabel", font=("Microsoft YaHei UI", 10, "bold"), foreground="#17324d")
        style.configure("Metric.TLabel", font=("Microsoft YaHei UI", 16, "bold"), foreground="#0b6e4f")
        style.configure("Predict.TButton", font=("Microsoft YaHei UI", 11, "bold"), padding=(18, 8))

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=18)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=5)
        container.columnconfigure(1, weight=6)
        container.rowconfigure(1, weight=1)

        header = ttk.Frame(container)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        # ttk.Label(header, text="乘员损伤预测代理模型", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="输入单一碰撞工况及约束系统参数，输出连续损伤指标、分项 AIS 和 MAIS。",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 0))
        ttk.Label(
            header,
            text=f"当前模型：{self.model_path}",
            style="Subtitle.TLabel",
            wraplength=1040,
        ).pack(anchor="w", pady=(3, 0))

        input_panel = ttk.LabelFrame(container, text="工况与约束系统输入", style="Section.TLabelframe", padding=14)
        input_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        input_panel.columnconfigure(1, weight=1)
        self._add_entry(input_panel, 0, "velocity", "碰撞速度（km/h）", "25–65，例如 34")
        self._add_entry(input_panel, 1, "impact_angle", "碰撞角度（°）", "逆时针为正，-30–30")
        self._add_entry(input_panel, 2, "overlap_percent", "碰撞重叠率（%）", "含主驾侧为正，例如 -55")
        self._add_entry(input_panel, 3, "swing_angle_deg", "乘员左右摆角（°）", "约 -8.60–8.60")
        self._add_entry(input_panel, 4, "height", "乘员身高（m）", "连续输入，1.60–1.80")
        self._add_entry(input_panel, 5, "bmi", "乘员 BMI（kg/m²）", "连续输入，21.0–28.5")
        self._add_combo(input_panel, 6, "airbag", "正面气囊状态", list(AIRBAG_STATES))
        self._add_combo(input_panel, 7, "knee_airbag", "膝部气囊状态", list(AIRBAG_STATES))
        self._add_entry(input_panel, 8, "load_limiter", "安全带限力（kN）", "连续输入 3.7–5.0；未启用输入 inf")
        self._add_entry(input_panel, 9, "seat_position", "座椅纵向位置参数（mm）", "-142–142，正负方向沿用仿真坐标")
        self._add_entry(input_panel, 10, "recline_angle_deg", "座椅前后仰角（°）", "约 -20–20，正负方向沿用仿真坐标")

        button_row = ttk.Frame(input_panel)
        button_row.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        ttk.Button(button_row, text="载入示例", command=self.fill_example).pack(side="left")
        ttk.Button(button_row, text="开始预测", command=self.run_prediction, style="Predict.TButton").pack(side="right")

        result_panel = ttk.Frame(container)
        result_panel.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        result_panel.columnconfigure(0, weight=1)
        result_panel.rowconfigure(2, weight=1)
        self._build_scalar_results(result_panel)
        self._build_ais_results(result_panel)
        self._build_ais_chart(result_panel)

        ttk.Label(
            container,
            text="2026-08",
            style="Subtitle.TLabel",
            wraplength=1040,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.root.bind("<Return>", lambda _event: self.run_prediction())

    def _add_entry(self, parent: ttk.LabelFrame, row: int, key: str, label: str, hint: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        variable = tk.StringVar()
        ttk.Entry(parent, textvariable=variable, width=18).grid(row=row, column=1, sticky="ew", padx=(10, 8), pady=5)
        ttk.Label(parent, text=hint, style="Subtitle.TLabel").grid(row=row, column=2, sticky="w", pady=5)
        self.entries[key] = variable

    def _add_combo(self, parent: ttk.LabelFrame, row: int, key: str, label: str, options: list[str]) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        variable = tk.StringVar()
        ttk.Combobox(parent, textvariable=variable, values=options, state="readonly", width=16).grid(
            row=row, column=1, sticky="ew", padx=(10, 8), pady=5
        )
        self.entries[key] = variable

    def _build_scalar_results(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="连续损伤指标", style="Section.TLabelframe", padding=12)
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        metrics = (
            ("Amax", "最大胸部加速度", "Amax", "g"),
            ("Dmax", "最大胸部压缩量", "Dmax", "mm"),
            ("CTI", "综合胸部损伤指标", "CTI", "—"),
            ("HIC", "头部损伤指标", "HIC", "—"),
            ("Nij", "颈部损伤指标", "Nij", "—"),
        )
        for column, (key, name, code, unit) in enumerate(metrics):
            frame.columnconfigure(column, weight=1, uniform="scalar_metric", minsize=96)
            cell = ttk.Frame(frame)
            cell.grid(row=0, column=column, sticky="nsew", padx=4)
            cell.columnconfigure(0, weight=1)
            ttk.Label(cell, text=name, style="MetricName.TLabel", anchor="center").grid(row=0, column=0, sticky="ew")
            ttk.Label(cell, text=code, style="MetricCode.TLabel", anchor="center").grid(
                row=1, column=0, sticky="ew", pady=(1, 0)
            )
            variable = tk.StringVar(value="--")
            ttk.Label(cell, textvariable=variable, style="Metric.TLabel", anchor="center").grid(
                row=2, column=0, sticky="ew", pady=(7, 0)
            )
            ttk.Label(cell, text=unit, style="Subtitle.TLabel", anchor="center").grid(row=3, column=0, sticky="ew")
            self.scalar_values[key] = variable

    def _build_ais_results(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="损伤等级", style="Section.TLabelframe", padding=12)
        frame.grid(row=1, column=0, sticky="ew", pady=8)
        labels = (("cti_AIS", "胸部 AIS"), ("hic_AIS", "头部 AIS"), ("nij_AIS", "颈部 AIS"), ("MAIS", "MAIS"))
        for column, (key, label) in enumerate(labels):
            frame.columnconfigure(column, weight=1)
            ttk.Label(frame, text=label, anchor="center").grid(row=0, column=column, sticky="ew")
            variable = tk.StringVar(value="--")
            ttk.Label(frame, textvariable=variable, style="Metric.TLabel", anchor="center").grid(
                row=1, column=column, sticky="ew", pady=(5, 0)
            )
            self.ais_values[key] = variable

    def _build_ais_chart(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="AIS 等级对比", style="Section.TLabelframe", padding=10)
        frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.chart = tk.Canvas(frame, background="#f7f9fb", highlightthickness=0, height=250)
        self.chart.grid(row=0, column=0, sticky="nsew")
        self.chart.bind("<Configure>", lambda _event: self._draw_chart(self.last_levels))

    def fill_example(self) -> None:
        for key, value in DEFAULT_EXAMPLE.items():
            self.entries[key].set(value)

    def run_prediction(self) -> None:
        values = {key: variable.get().strip() for key, variable in self.entries.items()}
        try:
            result = predict_case(self.bundle, values)
        except (TypeError, ValueError, KeyError) as error:
            messagebox.showerror("输入参数有误", str(error), parent=self.root)
            return
        except Exception as error:
            messagebox.showerror("预测失败", f"代理模型执行失败：{error}", parent=self.root)
            return

        formats = {"Amax": ".2f", "Dmax": ".2f", "CTI": ".3f", "HIC": ".1f", "Nij": ".3f"}
        for key, number_format in formats.items():
            self.scalar_values[key].set(format(float(result[key]), number_format))
        for key in self.ais_values:
            self.ais_values[key].set(str(int(result[key])))
        self.last_levels = {key: int(result[key]) for key in self.ais_values}
        self._draw_chart(self.last_levels)

    def _draw_chart(self, levels: Mapping[str, int]) -> None:
        self.chart.delete("all")
        width = max(self.chart.winfo_width(), 420)
        height = max(self.chart.winfo_height(), 220)
        left, right, top, bottom = 115, 28, 24, 30
        chart_width = width - left - right
        names = (("cti_AIS", "胸部 AIS"), ("hic_AIS", "头部 AIS"), ("nij_AIS", "颈部 AIS"), ("MAIS", "MAIS"))
        for level in range(6):
            x = left + chart_width * level / 5
            self.chart.create_line(x, top, x, height - bottom, fill="#d9e2ec")
            self.chart.create_text(x, height - 12, text=str(level), fill="#52606d")
        colors = ("#6ab04c", "#badc58", "#f9ca24", "#f0932b", "#eb4d4b", "#8e2c2c")
        row_height = (height - top - bottom) / len(names)
        for index, (key, label) in enumerate(names):
            y = top + row_height * (index + 0.5)
            self.chart.create_text(left - 12, y, text=label, anchor="e", fill="#17324d")
            if key not in levels:
                continue
            level = max(0, min(5, int(levels[key])))
            bar_end = left + chart_width * level / 5 if level > 0 else left + 4
            self.chart.create_rectangle(left, y - 13, bar_end, y + 13, fill=colors[level], outline="")
            self.chart.create_text(min(bar_end + 18, width - 12), y, text=str(level), fill="#17324d", font=("Microsoft YaHei UI", 10, "bold"))


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    model_path = resolve_model_path(project_dir, args.model)
    bundle = joblib.load(model_path)
    if args.check:
        print(json.dumps({"model_path": str(model_path), "prediction": predict_case(bundle, DEFAULT_EXAMPLE)}, ensure_ascii=True, indent=2))
        return
    root = tk.Tk()
    InjurySurrogateApp(root, bundle, model_path)
    root.mainloop()


if __name__ == "__main__":
    main()
