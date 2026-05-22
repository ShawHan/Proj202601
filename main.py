import argparse
import importlib
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

DEFAULT_THREAD_COUNT = "1"
os.environ.setdefault("OMP_NUM_THREADS", DEFAULT_THREAD_COUNT)
os.environ.setdefault("MKL_NUM_THREADS", DEFAULT_THREAD_COUNT)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", DEFAULT_THREAD_COUNT)
os.environ.setdefault("NUMEXPR_NUM_THREADS", DEFAULT_THREAD_COUNT)

import numpy as np
from tqdm import tqdm

from utils import loader as data_utils
from utils import tools as run_tools


METHODS = {
    "XGBBaseline": ("scripts.XGBBaseline", "XGBBaseline"),
    "PINN20260414": ("scripts.PINN20260414", "PINN20260414"),
    "PINN20260511": ("scripts.PINN20260511", "PINN20260511"),
    "PINN20260512-HS": ("scripts.PINN20260512_HS", "PINN20260512-HS"),
    "PINN20260522-KimLayered": ("scripts.PINN20260522_KimLayered", "PINN20260522-KimLayered"),
}


def resolve_methods(raw_method):
    selected = []
    for raw_name in raw_method.split(","):
        method_name = Path(raw_name.strip()).stem
        if not method_name:
            continue
        if method_name not in METHODS:
            raise SystemExit(f"method not found: {method_name}")
        selected.append(METHODS[method_name])
    if not selected:
        raise SystemExit("method not found")
    return selected


def build_common_parser():
    parser = argparse.ArgumentParser(
        description="Proj202601"
    )
    parser.add_argument(
        "--method",
        default="XGBBaseline",
        help="方法名: XGBBaseline、PINN20260414、PINN20260511、PINN20260512-HS，多个方法用逗号分隔",
    )
    parser.add_argument("--data", default="datasets/PE_TEOS.csv", help="CSV 数据路径")
    parser.add_argument("--runs", type=int, default=1, help="重复运行次数")
    parser.add_argument("--seed", type=int, default=42, help="全局随机数种子")
    parser.add_argument("--output", default="output", help="输出根目录")
    parser.add_argument("--torch-threads", type=int, default=1, help="PyTorch/OpenMP CPU 线程数")
    return parser


def parse_model_args(module, unknown_args):
    parser = argparse.ArgumentParser(add_help=False)
    if hasattr(module, "add_model_args"):
        module.add_model_args(parser)
    args, unused = parser.parse_known_args(unknown_args)
    return vars(args), unused


def build_method_state(module_path, script_name, unknown_args, torch_threads):
    module = importlib.import_module(module_path)
    configure_torch_threads(module, torch_threads)
    model_args, unused = parse_model_args(module, unknown_args)
    if unused:
        print(f"警告: {script_name} 忽略未识别参数: {' '.join(unused)}")

    return {
        "module": module,
        "script_name": script_name,
        "model_args": model_args,
        "result_rows": [],
        "run_summaries": [],
        "history_frames": [],
    }


def configure_torch_threads(module, torch_threads):
    thread_count = max(1, int(torch_threads))
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    os.environ["MKL_NUM_THREADS"] = str(thread_count)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(thread_count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(thread_count)

    torch_module = getattr(module, "torch", None)
    if torch_module is None:
        return
    torch_module.set_num_threads(thread_count)
    try:
        torch_module.set_num_interop_threads(1)
    except RuntimeError:
        pass


def run_one_method(state, common_args, run_index, run_seed, split_obj):
    script_name = state["script_name"]
    args = SimpleNamespace(
        data=common_args.data,
        seed=run_seed,
        **state["model_args"],
    )

    started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    result = state["module"].train(args, split_obj=split_obj)
    elapsed = time.time() - t0
    finished_at = datetime.now().isoformat(timespec="seconds")

    metrics = result["metrics"]
    runtime_seconds = float(result.get("runtime_seconds", elapsed))
    metrics["run_started_at"] = started_at
    metrics["run_finished_at"] = finished_at
    state["result_rows"].extend(
        run_tools.flatten_run_metrics(
            script_name,
            run_index,
            run_seed,
            runtime_seconds,
            metrics,
        )
    )
    state["run_summaries"].append(
        {
            "method": script_name,
            "run": run_index,
            "seed": run_seed,
            "runtime_seconds": runtime_seconds,
            "run_started_at": started_at,
            "run_finished_at": finished_at,
            "counts": metrics.get("counts", {}),
            "split_ratios": split_obj.get("split_ratios"),
            "training": metrics.get("training", {}),
        }
    )
    train_history = result.get("loss_history")
    if train_history is not None and not train_history.empty:
        train_history = train_history.copy()
        train_history.insert(0, "run", run_index)
        train_history.insert(0, "method", script_name)
        state["history_frames"].append(train_history)


def write_method_outputs(state, output_root):
    script_name = state["script_name"]
    method_dir = Path(output_root) / script_name
    result_path, summary_csv_path, summary_path = run_tools.write_result_files(
        method_dir,
        state["result_rows"],
        state["run_summaries"],
    )
    train_history_path = method_dir / "trainHistory.csv"

    train_history_df = run_tools.build_train_history_frame(state["history_frames"])
    train_history_df.to_csv(train_history_path, index=False)

    print(f"\n[{script_name}] 已写入:")
    print(f"- {result_path}")
    print(f"- {summary_csv_path}")
    print(f"- {summary_path}")
    print(f"- {train_history_path}")


def main():
    common_parser = build_common_parser()
    common_args, unknown_args = common_parser.parse_known_args()
    if common_args.runs < 1:
        raise SystemExit("--runs 必须大于等于 1")

    random.seed(common_args.seed)
    np.random.seed(common_args.seed)

    selected_methods = resolve_methods(common_args.method)
    method_states = [
        build_method_state(module_path, script_name, unknown_args, common_args.torch_threads)
        for module_path, script_name in selected_methods
    ]
    splits_by_run = {}
    method_names = ", ".join(state["script_name"] for state in method_states)

    print("运行配置:")
    print(f"- methods: {method_names}")
    print(f"- runs: {common_args.runs}")
    print(f"- seed: {common_args.seed}")
    print(f"- data: {common_args.data}")
    print(f"- output: {common_args.output}")

    total_tasks = common_args.runs * len(method_states)
    with tqdm(total=total_tasks, desc="Runs", unit="task") as progress:
        for run_index in range(1, common_args.runs + 1):
            run_seed = int(common_args.seed + run_index - 1)
            random.seed(run_seed)
            np.random.seed(run_seed)
            split_obj = data_utils.create_split(common_args.data, seed=run_seed)
            splits_by_run[str(run_index)] = split_obj

            for state in method_states:
                random.seed(run_seed)
                np.random.seed(run_seed)
                progress.set_postfix(run=run_index, method=state["script_name"])
                run_one_method(state, common_args, run_index, run_seed, split_obj)
                progress.update(1)

    for state in method_states:
        write_method_outputs(state, common_args.output)

    output_root = Path(common_args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(splits_by_run, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
