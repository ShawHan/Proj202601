import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from utils import loader as data_utils
from utils import tools as run_tools

try:
    from xgboost import XGBRegressor
except Exception as exc:
    raise SystemExit(
        "无法导入 xgboost。请先安装: pip install xgboost"
    ) from exc


def _normalize_col(col: str) -> str:
    # Normalize punctuations and spaces to make bilingual/variant headers matchable.
    s = str(col).strip().lower()
    s = (
        s.replace("（", "(")
        .replace("）", ")")
        .replace("℃", "c")
        .replace("°c", "c")
        .replace(" ", "")
    )
    s = re.sub(r"[\u3000\t\n\r]", "", s)
    return s


def _build_column_map(columns):
    return {_normalize_col(c): c for c in columns}


def _pick_col(col_map, aliases, name):
    for alias in aliases:
        key = _normalize_col(alias)
        if key in col_map:
            return col_map[key]
    raise KeyError(f"未找到{name}列，候选别名: {aliases}")


def resolve_columns(df: pd.DataFrame):
    col_map = _build_column_map(df.columns)

    feature_aliases = {
        "Pressure": ["压力pressure(torr)", "pressure(torr)", "pressure"],
        "O2": ["氧气o2(sccm)", "o2(sccm)", "o2"],
        "TEOS": ["teos(mgm)", "teos"],
        "He": ["氦气he(sccm)", "he(sccm)", "he"],
        "Time": ["time(s)", "time"],
        "Space": ["距离space(mm)", "space(mm)", "space"],
        "Temperature": ["温度temperature(c)", "temperature(c)", "temperature"],
        "Power": ["射频功率power(w)", "power(w)", "power"],
    }

    target_aliases = {
        "Thickness": ["厚度thickness(a)", "thickness(a)", "thickness"],
        "MAX": ["max", "最大厚度max"],
        "MIN": ["min", "最小厚度min"],
        "Uniformity_Range": ["均匀性uniformity/range", "uniformity/range", "uniformity_range"],
        "Uniformity_1sigma": [
            "均匀性uniformity/1sigma",
            "uniformity/1sigma",
            "uniformity_1sigma",
            "uniformity/1 sigma",
        ],
        "RI": ["折射率refractiveindex", "refractiveindex", "ri"],
        "Deposition_Rate": [
            "沉积速率depositionrate(a/min)",
            "depositionrate(a/min)",
            "depositionrate",
        ],
        "Stress": ["应力stress(mpa)", "stress(mpa)", "stress"],
    }

    feature_cols = {
        "Pressure": _pick_col(col_map, feature_aliases["Pressure"], "压力 Pressure(Torr)"),
        "O2": _pick_col(col_map, feature_aliases["O2"], "氧气 O2(sccm)"),
        "TEOS": _pick_col(col_map, feature_aliases["TEOS"], "TEOS(mgm)"),
        "He": _pick_col(col_map, feature_aliases["He"], "氦气 HE(sccm)"),
        "Time": _pick_col(col_map, feature_aliases["Time"], "Time(S)"),
        "Space": _pick_col(col_map, feature_aliases["Space"], "距离 Space(mm)"),
        "Temperature": _pick_col(col_map, feature_aliases["Temperature"], "温度 Temperature(℃)"),
        "Power": _pick_col(col_map, feature_aliases["Power"], "射频功率 Power(W)"),
    }
    target_cols = {
        "Thickness": _pick_col(col_map, target_aliases["Thickness"], "厚度 Thickness(A)"),
        "MAX": _pick_col(col_map, target_aliases["MAX"], "MAX"),
        "MIN": _pick_col(col_map, target_aliases["MIN"], "MIN"),
        "Deposition_Rate": _pick_col(
            col_map,
            target_aliases["Deposition_Rate"],
            "沉积速率 Deposition Rate(A/min)",
        ),
        "Uniformity_Range": _pick_col(col_map, target_aliases["Uniformity_Range"], "均匀性 Uniformity/Range"),
        "Uniformity_1sigma": _pick_col(
            col_map,
            target_aliases["Uniformity_1sigma"],
            "均匀性 Uniformity/1 sigma",
        ),
        "RI": _pick_col(col_map, target_aliases["RI"], "折射率 Refractive Index"),
        "Stress": _pick_col(col_map, target_aliases["Stress"], "应力 Stress(Mpa)"),
    }

    return feature_cols, target_cols


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def report_metrics(y_true, y_pred, name="set"):
    metrics = calculate_metrics(y_true, y_pred)
    return metrics


def _row_to_key(row, cols):
    return tuple(round(float(row[c]), 6) for c in cols)


def load_split(split_path: Path):
    with split_path.open("r", encoding="utf-8") as f:
        split_obj = json.load(f)
    return split_obj


def resolve_output_path(raw_arg: str, split_path: Path, default_name: str, model_tag: str) -> Path:
    split_tag = split_path.stem
    out_dir = Path("output") / split_tag / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if raw_arg == default_name:
        return out_dir / default_name

    out_path = Path(raw_arg)
    if not out_path.is_absolute():
        out_path = out_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    return out_path


def add_model_args(parser):
    parser.add_argument("--xgb-n-estimators", type=int, default=1200)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.03)
    parser.add_argument("--xgb-subsample", type=float, default=0.85)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--xgb-min-child-weight", type=float, default=2.0)
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.05)
    parser.add_argument("--xgb-reg-lambda", type=float, default=2.0)
    return parser


def build_arg_parser():
    parser = argparse.ArgumentParser(description="XGBoost 回归：按固定划分预测薄膜 8 项质量指标")
    parser.add_argument("--data", default="datasets/PE_TEOS.csv", help="CSV 数据路径")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pred-out", default="test_predictions_phase1.csv", help="预测结果 CSV")
    parser.add_argument("--metrics-out", default="metrics.json", help="指标 JSON")
    parser.add_argument("--loss-out", default="loss_history.csv", help="训练过程损失 CSV")
    add_model_args(parser)
    return parser


def train(args, split_obj=None):
    import time

    t0 = time.time()
    data_path = Path(args.data)
    if split_obj is None:
        split_obj = data_utils.create_split(data_path, seed=args.seed)

    prepared = data_utils.prepare_model_data(
        data_path,
        split_obj,
        disable_derived_features=True,
    )

    feature_cols = prepared["feature_cols"]
    target_cols = prepared["target_cols"]
    target_order = prepared["target_keys"]
    work_df = prepared["work_df"]
    train_mask = prepared["train_mask"]
    test_mask = prepared["test_mask"]

    X = work_df[list(feature_cols.values())]
    y = work_df[[target_cols[k] for k in target_order]]

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    models = {}
    pred_train = {}
    pred_test = {}
    loss_frames = []
    per_target_metrics = {}

    metric_key_map = {
        "Thickness": "thickness",
        "MAX": "max_thickness",
        "MIN": "min_thickness",
        "Deposition_Rate": "deposition_rate",
        "Uniformity_Range": "uniformity_range",
        "Uniformity_1sigma": "uniformity_1sigma",
        "RI": "ri",
        "Stress": "stress",
    }

    for idx, target_key in enumerate(tqdm(target_order, desc="XGB targets", leave=False)):
        y_train_col = y_train.iloc[:, idx]
        y_test_col = y_test.iloc[:, idx]

        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            min_child_weight=args.xgb_min_child_weight,
            reg_alpha=args.xgb_reg_alpha,
            reg_lambda=args.xgb_reg_lambda,
            random_state=args.seed + idx,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train_col,
            eval_set=[(X_train, y_train_col)],
            verbose=False,
        )

        models[target_key] = model

        evals_result = model.evals_result()
        train_rmse_curve = evals_result.get("validation_0", {}).get("rmse", [])

        loss_frames.append(
            pd.DataFrame(
                {
                    "target": target_key,
                    "iteration": np.arange(1, len(train_rmse_curve) + 1),
                    "train_rmse": train_rmse_curve,
                }
            )
        )

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        pred_train[target_key] = train_pred
        pred_test[target_key] = test_pred

        train_m = report_metrics(y_train_col, train_pred, "Train")
        test_m = report_metrics(y_test_col, test_pred, "Test")

        per_target_metrics[metric_key_map[target_key]] = {
            "train": train_m,
            "test": test_m,
        }

    loss_df = pd.concat(loss_frames, axis=0, ignore_index=True)

    # Backward-compatible keys for existing multi-split aggregation (thickness as default task).
    metrics = {
        "train": per_target_metrics["thickness"]["train"],
        "test": per_target_metrics["thickness"]["test"],
        **per_target_metrics,
    }
    metrics["split_seed"] = split_obj.get("split_seed")
    metrics["counts"] = {
        "train": int(len(X_train)),
        "test": int(len(X_test)),
    }

    training_by_target = {}
    for target_key in target_order:
        model = models[target_key]
        evals_result = model.evals_result()
        train_rmse_curve = evals_result.get("validation_0", {}).get("rmse", [])
        if len(train_rmse_curve) == 0:
            continue
        best_idx = int(np.argmin(train_rmse_curve))
        training_by_target[metric_key_map[target_key]] = {
            "loss_name": "rmse",
            "objective": "reg:squarederror",
            "best_train_rmse": float(train_rmse_curve[best_idx]),
            "best_iteration": best_idx + 1,
            "final_train_rmse": float(train_rmse_curve[-1]),
        }
    metrics["training_loss"] = training_by_target

    pred_df = run_tools.build_prediction_frame(
        prepared,
        {
            "train": np.column_stack([pred_train[k] for k in target_order]),
            "test": np.column_stack([pred_test[k] for k in target_order]),
        },
    )

    runtime_seconds = time.time() - t0
    metrics.setdefault("training", {})
    metrics["training"]["seconds"] = float(runtime_seconds)

    if getattr(args, "pred_out", None):
        pred_out_path = Path(args.pred_out)
        pred_out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_out_path, index=False)
    if getattr(args, "metrics_out", None):
        metrics_out_path = Path(args.metrics_out)
        metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    if getattr(args, "loss_out", None):
        loss_out_path = Path(args.loss_out)
        loss_out_path.parent.mkdir(parents=True, exist_ok=True)
        loss_df.to_csv(loss_out_path, index=False)

    return {
        "metrics": metrics,
        "predictions": pred_df,
        "loss_history": loss_df,
        "runtime_seconds": runtime_seconds,
    }


def main():
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
