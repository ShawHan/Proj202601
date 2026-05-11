import argparse
import copy
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from utils import loader as data_utils
from utils import tools as run_tools

try:
    import torch
    import torch.nn as nn
except Exception as exc:
    raise SystemExit("无法导入 torch。请先安装: pip install torch") from exc


def _normalize_col(col: str) -> str:
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
        "Uniformity_Range": _pick_col(col_map, target_aliases["Uniformity_Range"], "均匀性 Uniformity/Range"),
        "Uniformity_1sigma": _pick_col(
            col_map,
            target_aliases["Uniformity_1sigma"],
            "均匀性 Uniformity/1 sigma",
        ),
        "RI": _pick_col(col_map, target_aliases["RI"], "折射率 Refractive Index"),
        "Deposition_Rate": _pick_col(
            col_map,
            target_aliases["Deposition_Rate"],
            "沉积速率 Deposition Rate(A/min)",
        ),
        "Stress": _pick_col(col_map, target_aliases["Stress"], "应力 Stress(Mpa)"),
    }
    return feature_cols, target_cols


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-6):
    return a / np.clip(b, eps, None)


def add_derived_features(df: pd.DataFrame, feature_cols):
    p = df[feature_cols["Pressure"]].to_numpy(dtype=np.float32)
    o2 = df[feature_cols["O2"]].to_numpy(dtype=np.float32)
    teos = df[feature_cols["TEOS"]].to_numpy(dtype=np.float32)
    he = df[feature_cols["He"]].to_numpy(dtype=np.float32)
    t = df[feature_cols["Time"]].to_numpy(dtype=np.float32)
    temp = df[feature_cols["Temperature"]].to_numpy(dtype=np.float32)
    power = df[feature_cols["Power"]].to_numpy(dtype=np.float32)

    total_flow = o2 + teos + he
    df["drv_Total_Flow"] = total_flow
    df["drv_O2_TEOS_Ratio"] = _safe_div(o2, teos)
    df["drv_HE_TEOS_Ratio"] = _safe_div(he, teos)
    df["drv_Power_Time"] = power * t
    df["drv_Temp_Time"] = temp * t
    df["drv_Power_Pressure_Ratio"] = _safe_div(power, p)
    df["drv_TEOS_Partial_Fraction"] = _safe_div(teos, total_flow)
    df["drv_O2_Partial_Fraction"] = _safe_div(o2, total_flow)
    return df


def load_split(split_path: Path):
    with split_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _row_to_key(row, cols):
    return tuple(round(float(row[c]), 6) for c in cols)


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


def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


class StandardScalerNumpy:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray):
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        return self

    def transform(self, x: np.ndarray):
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray):
        return x * self.std_ + self.mean_


class MultiTaskHybridPINN(nn.Module):
    def __init__(self, in_dim=8, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 64),
            nn.Tanh(),
        )
        self.heads = nn.ModuleDict(
            {
                "Thickness": nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)),
                "MAX": nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)),
                "MIN": nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)),
                "Deposition_Rate": nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)),
                "Uniformity_Range": nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)),
                "Uniformity_1sigma": nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)),
                "RI": nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)),
                "Stress": nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)),
            }
        )

    def forward(self, x):
        z = self.encoder(x)
        out = [self.heads[k](z) for k in [
            "Thickness",
            "MAX",
            "MIN",
            "Deposition_Rate",
            "Uniformity_Range",
            "Uniformity_1sigma",
            "RI",
            "Stress",
        ]]
        return torch.cat(out, dim=1)


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


def add_model_args(parser):
    parser.add_argument("--epochs-pretrain", type=int, default=300)
    parser.add_argument("--epochs-finetune", type=int, default=900)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--disable-derived-features", action="store_true", help="关闭派生特征")
    parser.add_argument("--lambda-thickness", type=float, default=1.0, help="Thickness/MAX/MIN 数据项权重")
    parser.add_argument("--lambda-rate", type=float, default=0.8, help="沉积速率数据项权重")
    parser.add_argument("--lambda-uniformity", type=float, default=0.6, help="均匀性数据项权重")
    parser.add_argument("--lambda-ri", type=float, default=0.4, help="折射率数据项权重")
    parser.add_argument("--lambda-stress", type=float, default=0.5, help="应力数据项权重")
    parser.add_argument("--lambda-ht", type=float, default=0.5)
    parser.add_argument("--lambda-maxmin", type=float, default=0.2)
    parser.add_argument("--lambda-uniformity-def", type=float, default=0.3)
    parser.add_argument("--lambda-mono-t", type=float, default=0.1)
    parser.add_argument("--lambda-nonneg-h", type=float, default=0.05)
    parser.add_argument("--lambda-nonneg-r", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=120)
    return parser


def build_arg_parser():
    parser = argparse.ArgumentParser(description="多任务混合 PINN: 薄膜质量 8 输出预测")
    parser.add_argument("--data", default="datasets/PE_TEOS.csv", help="CSV 数据路径")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pred-out", default="old_predictions.csv", help="预测结果 CSV")
    parser.add_argument("--metrics-out", default="old_metrics.json", help="指标 JSON")
    parser.add_argument("--loss-out", default="old_loss_history.csv", help="训练过程损失 CSV")
    add_model_args(parser)
    return parser


def train(args, split_obj=None):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = Path(args.data)
    if split_obj is None:
        split_obj = data_utils.create_split(data_path, seed=args.seed)

    prepared = data_utils.prepare_model_data(
        data_path,
        split_obj,
        disable_derived_features=args.disable_derived_features,
    )
    feature_cols = prepared["feature_cols"]
    target_cols = prepared["target_cols"]
    feat_order = prepared["feat_order"]
    target_order = prepared["target_order"]
    derived_feat_order = prepared["derived_feat_order"]
    train_df = prepared["train_df"]
    val_df = prepared["val_df"]
    test_df = prepared["test_df"]

    time_idx = feat_order.index(feature_cols["Time"])
    idx = {
        "Thickness": 0,
        "MAX": 1,
        "MIN": 2,
        "Deposition_Rate": 3,
        "Uniformity_Range": 4,
        "Uniformity_1sigma": 5,
        "RI": 6,
        "Stress": 7,
    }

    x_train_raw = train_df[feat_order].to_numpy(dtype=np.float32)
    y_train_raw = train_df[target_order].to_numpy(dtype=np.float32)
    x_val_raw = val_df[feat_order].to_numpy(dtype=np.float32)
    y_val_raw = val_df[target_order].to_numpy(dtype=np.float32)
    x_test_raw = test_df[feat_order].to_numpy(dtype=np.float32)
    y_test_raw = test_df[target_order].to_numpy(dtype=np.float32)

    x_scaler = StandardScalerNumpy().fit(x_train_raw)
    y_scaler = StandardScalerNumpy().fit(y_train_raw)

    x_train = x_scaler.transform(x_train_raw).astype(np.float32)
    x_val = x_scaler.transform(x_val_raw).astype(np.float32)
    x_test = x_scaler.transform(x_test_raw).astype(np.float32)
    y_train = y_scaler.transform(y_train_raw).astype(np.float32)
    y_val = y_scaler.transform(y_val_raw).astype(np.float32)
    y_test = y_scaler.transform(y_test_raw).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskHybridPINN(in_dim=len(feat_order), hidden=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    x_train_t = to_tensor(x_train, device)
    y_train_t = to_tensor(y_train, device)
    x_val_t = to_tensor(x_val, device)

    train_time_raw_t = to_tensor(x_train_raw[:, [time_idx]], device)

    y_mean_t = to_tensor(y_scaler.mean_.reshape(1, -1), device)
    y_std_t = to_tensor(y_scaler.std_.reshape(1, -1), device)
    x_std_time_t = torch.tensor([[x_scaler.std_[time_idx]]], dtype=torch.float32, device=device)

    mse = nn.MSELoss()

    total_epochs = args.epochs_pretrain + args.epochs_finetune
    history = []
    best_state = None
    best_val_rmse = float("inf")
    best_epoch = 0
    bad_epochs = 0

    t0 = time.time()
    epoch_iter = tqdm(
        range(1, total_epochs + 1),
        desc=f"PINN20260414 seed={args.seed}",
        leave=False,
    )
    for epoch in epoch_iter:
        model.train()
        optimizer.zero_grad()

        x_in = x_train_t.detach().clone().requires_grad_(True)
        pred_n = model(x_in)
        h_pred_n = pred_n[:, [idx["Thickness"]]]
        max_pred_n = pred_n[:, [idx["MAX"]]]
        min_pred_n = pred_n[:, [idx["MIN"]]]
        r_pred_n = pred_n[:, [idx["Deposition_Rate"]]]
        u_range_pred_n = pred_n[:, [idx["Uniformity_Range"]]]
        u_sigma_pred_n = pred_n[:, [idx["Uniformity_1sigma"]]]
        ri_pred_n = pred_n[:, [idx["RI"]]]
        stress_pred_n = pred_n[:, [idx["Stress"]]]

        l_data_thk = (
            mse(h_pred_n, y_train_t[:, [idx["Thickness"]]])
            + mse(max_pred_n, y_train_t[:, [idx["MAX"]]])
            + mse(min_pred_n, y_train_t[:, [idx["MIN"]]])
        )
        l_data_rate = mse(r_pred_n, y_train_t[:, [idx["Deposition_Rate"]]])
        l_data_uni = (
            mse(u_range_pred_n, y_train_t[:, [idx["Uniformity_Range"]]])
            + mse(u_sigma_pred_n, y_train_t[:, [idx["Uniformity_1sigma"]]])
        )
        l_data_ri = mse(ri_pred_n, y_train_t[:, [idx["RI"]]])
        l_data_stress = mse(stress_pred_n, y_train_t[:, [idx["Stress"]]])
        l_data = (
            args.lambda_thickness * l_data_thk
            + args.lambda_rate * l_data_rate
            + args.lambda_uniformity * l_data_uni
            + args.lambda_ri * l_data_ri
            + args.lambda_stress * l_data_stress
        )

        pred_raw = pred_n * y_std_t + y_mean_t
        h_pred_raw = pred_raw[:, [idx["Thickness"]]]
        max_pred_raw = pred_raw[:, [idx["MAX"]]]
        min_pred_raw = pred_raw[:, [idx["MIN"]]]
        r_pred_raw = pred_raw[:, [idx["Deposition_Rate"]]]
        u_range_pred_raw = pred_raw[:, [idx["Uniformity_Range"]]]
        u_sigma_pred_raw = pred_raw[:, [idx["Uniformity_1sigma"]]]

        l_ht = torch.mean((h_pred_raw - r_pred_raw * (train_time_raw_t / 60.0)) ** 2)

        eps = 1e-6
        u_range_def = (max_pred_raw - min_pred_raw) * 100.0 / (2.0 * (h_pred_raw + eps))
        u_sigma_def = (max_pred_raw - min_pred_raw) * 100.0 / (6.0 * (h_pred_raw + eps))
        l_uniformity_def = torch.mean((u_range_pred_raw - u_range_def) ** 2) + torch.mean(
            (u_sigma_pred_raw - u_sigma_def) ** 2
        )

        l_maxmin = (
            torch.mean(torch.relu(h_pred_raw - max_pred_raw) ** 2)
            + torch.mean(torch.relu(min_pred_raw - h_pred_raw) ** 2)
            + torch.mean(torch.relu(min_pred_raw - max_pred_raw) ** 2)
        )

        dh_dxnorm = torch.autograd.grad(
            h_pred_raw,
            x_in,
            grad_outputs=torch.ones_like(h_pred_raw),
            create_graph=True,
        )[0][:, [time_idx]]
        dh_dt = dh_dxnorm / x_std_time_t

        l_mono_t = torch.mean(torch.relu(-dh_dt) ** 2)
        l_nonneg_h = (
            torch.mean(torch.relu(-h_pred_raw) ** 2)
            + torch.mean(torch.relu(-max_pred_raw) ** 2)
            + torch.mean(torch.relu(-min_pred_raw) ** 2)
        )
        l_nonneg_r = (
            torch.mean(torch.relu(-r_pred_raw) ** 2)
            + torch.mean(torch.relu(-u_range_pred_raw) ** 2)
            + torch.mean(torch.relu(-u_sigma_pred_raw) ** 2)
        )

        if epoch <= args.epochs_pretrain:
            total_loss = l_data
            stage = "pretrain"
        else:
            total_loss = (
                l_data
                + args.lambda_ht * l_ht
                + args.lambda_maxmin * l_maxmin
                + args.lambda_uniformity_def * l_uniformity_def
                + args.lambda_mono_t * l_mono_t
                + args.lambda_nonneg_h * l_nonneg_h
                + args.lambda_nonneg_r * l_nonneg_r
            )
            stage = "physics"

        total_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_pred_raw = model(x_train_t) * y_std_t + y_mean_t
            val_pred_raw = model(x_val_t) * y_std_t + y_mean_t

            train_thk_rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_train_raw[:, idx["Thickness"]],
                        train_pred_raw[:, idx["Thickness"]].cpu().numpy(),
                    )
                )
            )
            val_thk_rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_val_raw[:, idx["Thickness"]],
                        val_pred_raw[:, idx["Thickness"]].cpu().numpy(),
                    )
                )
            )

        history.append(
            {
                "epoch": epoch,
                "stage": stage,
                "total_loss": float(total_loss.item()),
                "data_thickness_group_loss": float(l_data_thk.item()),
                "data_rate_loss": float(l_data_rate.item()),
                "data_uniformity_loss": float(l_data_uni.item()),
                "data_ri_loss": float(l_data_ri.item()),
                "data_stress_loss": float(l_data_stress.item()),
                "phys_ht_loss": float(l_ht.item()),
                "phys_maxmin_loss": float(l_maxmin.item()),
                "phys_uniformity_def_loss": float(l_uniformity_def.item()),
                "phys_mono_t_loss": float(l_mono_t.item()),
                "phys_nonneg_h_loss": float(l_nonneg_h.item()),
                "phys_nonneg_r_loss": float(l_nonneg_r.item()),
                "train_thk_rmse": train_thk_rmse,
                "val_thk_rmse": val_thk_rmse,
            }
        )
        epoch_iter.set_postfix(
            stage=stage,
            train_rmse=f"{train_thk_rmse:.2f}",
            val_rmse=f"{val_thk_rmse:.2f}",
        )

        if val_thk_rmse < best_val_rmse:
            best_val_rmse = val_thk_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience and epoch > args.epochs_pretrain:
            break

    train_seconds = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    def predict_raw(x_np):
        model.eval()
        x_t = to_tensor(x_np, device)
        with torch.no_grad():
            pred_n = model(x_t).cpu().numpy()
        return y_scaler.inverse_transform(pred_n)

    train_pred = predict_raw(x_train)
    val_pred = predict_raw(x_val)
    test_pred = predict_raw(x_test)

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
    target_internal_order = [
        "Thickness",
        "MAX",
        "MIN",
        "Deposition_Rate",
        "Uniformity_Range",
        "Uniformity_1sigma",
        "RI",
        "Stress",
    ]

    per_target_metrics = {}
    for i, name in enumerate(target_internal_order):
        per_target_metrics[metric_key_map[name]] = {
            "train": calc_metrics(y_train_raw[:, i], train_pred[:, i]),
            "val": calc_metrics(y_val_raw[:, i], val_pred[:, i]),
            "test": calc_metrics(y_test_raw[:, i], test_pred[:, i]),
        }

    metrics = {
        "split_seed": split_obj.get("split_seed"),
        "counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        **per_target_metrics,
        "training": {
            "best_val_thk_rmse": float(best_val_rmse),
            "best_epoch": int(best_epoch),
            "trained_epochs": int(len(history)),
            "seconds": float(train_seconds),
            "device": str(device),
            "num_input_features": int(len(feat_order)),
            "num_outputs": int(len(target_order)),
            "use_derived_features": bool(not args.disable_derived_features),
            "loss_weights": {
                "lambda_thickness": args.lambda_thickness,
                "lambda_rate": args.lambda_rate,
                "lambda_uniformity": args.lambda_uniformity,
                "lambda_ri": args.lambda_ri,
                "lambda_stress": args.lambda_stress,
                "lambda_ht": args.lambda_ht,
                "lambda_maxmin": args.lambda_maxmin,
                "lambda_uniformity_def": args.lambda_uniformity_def,
                "lambda_mono_t": args.lambda_mono_t,
                "lambda_nonneg_h": args.lambda_nonneg_h,
                "lambda_nonneg_r": args.lambda_nonneg_r,
            },
        },
    }

    hist_df = pd.DataFrame(history)
    pred_df = run_tools.build_prediction_frame(
        prepared,
        {"train": train_pred, "val": val_pred, "test": test_pred},
    )

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
        hist_df.to_csv(loss_out_path, index=False)

    return {
        "metrics": metrics,
        "predictions": pred_df,
        "loss_history": hist_df,
        "runtime_seconds": train_seconds,
    }


def main():
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
