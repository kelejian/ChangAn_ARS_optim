"""自适应约束系统参数优化闭环的命令行入口。

该脚本只负责解析运行参数、定位项目根目录并调用流水线函数，具体的数据处理、代理模型训练、参数寻优和结果导出逻辑均位于 core 包内。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.pipeline import load_config, run_pipeline


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回配置路径和可选输出目录名。"""
    parser = argparse.ArgumentParser(description="Run adaptive restraint optimization pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="配置文件路径，默认使用 configs/default_config.yaml。",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="可选输出目录名；不提供时自动使用时间戳目录。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 所有相对路径均相对于入口脚本所在的项目根目录解析，避免从不同工作目录启动脚本时路径不一致。
    project_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_dir / config_path
    config = load_config(config_path)
    # run_name 为空时由流水线自动生成时间戳目录；指定 run_name 时可用于复现实验命名。
    output_dir = run_pipeline(config=config, project_dir=project_dir, run_name=args.run_name)
    # 仅输出 ASCII，避免 Windows 下 `conda run` 转印中文 stdout 时触发编码异常。
    print(f"ARS optimization finished. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
