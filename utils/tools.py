import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


METRIC_KEY_MAP = {
    "Thickness": "thickness",
    "MAX": "max_thickness",
    "MIN": "min_thickness",
    "Deposition_Rate": "deposition_rate",
    "Uniformity_Range": "uniformity_range",
    "Uniformity_1sigma": "uniformity_1sigma",
    "RI": "ri",
    "Stress": "stress",
}


def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def format_seconds(seconds):
    seconds = int(round(float(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_prediction_frame(prepared, predictions_by_split):
    feature_cols = prepared["feature_cols"]
    target_cols = prepared["target_cols"]
    feat_order = prepared["feat_order"]
    target_order = prepared["target_order"]
    derived_feat_order = prepared["derived_feat_order"]
    target_keys = prepared["target_keys"]

    rename_map = {v: k for k, v in feature_cols.items()}
    rename_map.update({k: k.replace("drv_", "Derived_") for k in derived_feat_order if k in feat_order})
    for key in target_keys:
        rename_map[target_cols[key]] = f"true_{key}"

    pred_col_names = [f"pred_{key}" for key in target_keys]
    pred_frames = []
    for split_name in ["train", "val", "test"]:
        if f"{split_name}_df" not in prepared or split_name not in predictions_by_split:
            continue
        part_df = prepared[f"{split_name}_df"]
        pred = predictions_by_split[split_name]
        out_df = part_df[feat_order + target_order + ["recipe_id"]].copy().rename(columns=rename_map)
        for i, col in enumerate(pred_col_names):
            out_df[col] = pred[:, i]

        for key, pred_col in zip(target_keys, pred_col_names):
            true_col = f"true_{key}"
            out_df[f"abs_err_{key}"] = np.abs(out_df[true_col] - out_df[pred_col])

        out_df.insert(0, "split", split_name)
        pred_frames.append(out_df)

    return pd.concat(pred_frames, axis=0, ignore_index=True)


def metrics_by_target(target_keys, y_by_split, pred_by_split):
    metrics = {}
    for i, target_key in enumerate(target_keys):
        metrics[METRIC_KEY_MAP[target_key]] = {
            split_name: calc_metrics(y[:, i], pred_by_split[split_name][:, i])
            for split_name, y in y_by_split.items()
        }
    return metrics


def flatten_run_metrics(method, run_index, seed, runtime_seconds, metrics):
    rows = []
    run_started_at = metrics.get("run_started_at")
    for target_name, split_metrics in metrics.items():
        if target_name in {"counts", "training", "split_seed", "run_started_at", "run_finished_at"}:
            continue
        if not isinstance(split_metrics, dict):
            continue
        for split_name, values in split_metrics.items():
            if split_name != "test":
                continue
            if not isinstance(values, dict) or "rmse" not in values:
                continue
            rows.append(
                {
                    "method": method,
                    "run": int(run_index),
                    "run_started_at": run_started_at,
                    "runtime_seconds": round(float(runtime_seconds), 6),
                    "target": target_name,
                    "mae": values.get("mae"),
                    "rmse": values.get("rmse"),
                    "r2": values.get("r2"),
                }
            )
    return rows


def summarize_results(result_df, run_summaries):
    summary_df = build_summary_frame(result_df)

    return {
        "runs": run_summaries,
        "test_averages": summary_df.to_dict(orient="records"),
    }


def build_summary_frame(result_df):
    grouped = (
        result_df.groupby(["target"], dropna=False)[["mae", "rmse", "r2"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    rows = []
    for _, row in grouped.iterrows():
        target = row[("target", "")]
        out_row = {"target": target}
        for metric, label in [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R2")]:
            mean = float(row[(metric, "mean")])
            std = row[(metric, "std")]
            std = 0.0 if pd.isna(std) else float(std)
            digits = 4 if target == "ri" and metric in {"mae", "rmse"} else 2
            out_row[label] = f"{mean:.{digits}f} ± {std:.{digits}f}"
        rows.append(out_row)

    return pd.DataFrame(rows, columns=["target", "MAE", "RMSE", "R2"])


def _sample_history_frame(history_df, max_rows):
    if len(history_df) <= max_rows:
        return history_df
    positions = np.linspace(0, len(history_df) - 1, max_rows).round().astype(int)
    positions = np.unique(positions)
    return history_df.iloc[positions]


def build_train_history_frame(history_frames, max_rows_per_run=200):
    if not history_frames:
        return pd.DataFrame()
    history_df = pd.concat(history_frames, axis=0, ignore_index=True)
    history_df = history_df.drop(columns=["seed"], errors="ignore")

    if "run" not in history_df.columns:
        return _sample_history_frame(history_df, max_rows_per_run).reset_index(drop=True)

    sampled_frames = [
        _sample_history_frame(group, max_rows_per_run)
        for _, group in history_df.groupby("run", sort=False)
    ]
    return pd.concat(sampled_frames, axis=0, ignore_index=True)


def write_result_files(output_dir, result_rows, run_summaries):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(result_rows)
    result_path = out_dir / "result.csv"
    summary_csv_path = out_dir / "summary.csv"
    summary_path = out_dir / "summary.json"

    result_df.to_csv(result_path, index=False)
    build_summary_frame(result_df).to_csv(summary_csv_path, index=False)
    summary = summarize_results(result_df, run_summaries)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return result_path, summary_csv_path, summary_path
