import re
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_KEYS = [
    "Pressure",
    "O2",
    "TEOS",
    "He",
    "Time",
    "Space",
    "Temperature",
    "Power",
]

TARGET_KEYS = [
    "Thickness",
    "MAX",
    "MIN",
    "Deposition_Rate",
    "Uniformity_Range",
    "Uniformity_1sigma",
    "RI",
    "Stress",
]

DERIVED_FEATURES = [
    "drv_Total_Flow",
    "drv_O2_TEOS_Ratio",
    "drv_HE_TEOS_Ratio",
    "drv_Power_Time",
    "drv_Temp_Time",
    "drv_Power_Pressure_Ratio",
    "drv_TEOS_Partial_Fraction",
    "drv_O2_Partial_Fraction",
]


def normalize_col(col: str) -> str:
    s = str(col).strip().lower().lstrip("\ufeff")
    s = (
        s.replace("（", "(")
        .replace("）", ")")
        .replace("℃", "c")
        .replace("°c", "c")
        .replace(" ", "")
        .replace("_", "")
    )
    return re.sub(r"[\u3000\t\n\r/]", "", s)


def _pick_col(col_map, aliases, name):
    for alias in aliases:
        key = normalize_col(alias)
        if key in col_map:
            return col_map[key]
    raise KeyError(f"未找到{name}列，候选别名: {aliases}")


def resolve_columns(df: pd.DataFrame):
    col_map = {normalize_col(c): c for c in df.columns}

    feature_aliases = {
        "Pressure": ["压力 pressure(torr)", "pressure(torr)", "pressure"],
        "O2": ["氧气 o2(sccm)", "o2(sccm)", "o2"],
        "TEOS": ["teos(mgm)", "teos"],
        "He": ["氦气 he(sccm)", "he(sccm)", "he"],
        "Time": ["time(s)", "time"],
        "Space": ["距离 space(mm)", "space(mm)", "space"],
        "Temperature": ["温度 temperature(c)", "temperature(c)", "temperature"],
        "Power": ["射频功率 power(w)", "power(w)", "power"],
    }

    target_aliases = {
        "Thickness": ["厚度 thickness(a)", "thickness(a)", "thickness"],
        "MAX": ["max", "最大厚度 max"],
        "MIN": ["min", "最小厚度 min"],
        "Uniformity_Range": ["均匀性 uniformity/range", "uniformity/range", "uniformity_range"],
        "Uniformity_1sigma": [
            "均匀性 uniformity/1sigma",
            "uniformity/1sigma",
            "uniformity_1sigma",
            "uniformity/1 sigma",
        ],
        "RI": ["折射率 refractive index", "refractive index", "refractiveindex", "ri"],
        "Deposition_Rate": [
            "沉积速率 deposition rate(a/min)",
            "deposition rate(a/min)",
            "deposition rate",
            "depositionrate(a/min)",
            "depositionrate",
        ],
        "Stress": ["应力 stress(mpa)", "stress(mpa)", "stress"],
    }

    feature_cols = {
        key: _pick_col(col_map, feature_aliases[key], key) for key in FEATURE_KEYS
    }
    target_cols = {
        key: _pick_col(col_map, target_aliases[key], key) for key in TARGET_KEYS
    }
    return feature_cols, target_cols


def load_raw_dataset(data_path):
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到 CSV 数据文件: {path}")
    return pd.read_csv(path, encoding="utf-8-sig").dropna(how="all").copy()


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


def row_to_key(row, cols):
    return tuple(round(float(row[c]), 6) for c in cols)


def create_split(data_path, seed, ratios=(0.6, 0.2, 0.2), recipe_columns=None):
    df = load_raw_dataset(data_path)
    feature_cols, target_cols = resolve_columns(df)
    recipe_columns = recipe_columns or FEATURE_KEYS

    selected_cols = [feature_cols[k] for k in FEATURE_KEYS] + [target_cols[k] for k in TARGET_KEYS]
    work_df = df[selected_cols].dropna().copy()
    if len(work_df) < 5:
        raise ValueError(f"有效样本数过少: {len(work_df)}，请检查数据文件")

    recipe_df_cols = [feature_cols[k] for k in recipe_columns]
    recipe_keys = work_df[recipe_df_cols].drop_duplicates().reset_index(drop=True)
    recipe_ids = np.arange(len(recipe_keys))

    rng = np.random.default_rng(seed)
    rng.shuffle(recipe_ids)

    n_recipes = len(recipe_ids)
    n_train = max(1, int(round(n_recipes * ratios[0])))
    n_val = max(1, int(round(n_recipes * ratios[1])))
    if n_train + n_val >= n_recipes:
        n_train = max(1, n_recipes - 2)
        n_val = 1

    train_ids = sorted(int(i) for i in recipe_ids[:n_train])
    val_ids = sorted(int(i) for i in recipe_ids[n_train : n_train + n_val])
    test_ids = sorted(int(i) for i in recipe_ids[n_train + n_val :])

    recipe_key_map = {
        str(i): [float(v) for v in recipe_keys.iloc[i].to_list()]
        for i in range(len(recipe_keys))
    }

    return {
        "split_seed": int(seed),
        "split_ratios": list(ratios),
        "recipe_columns": recipe_columns,
        "train_recipe_ids": train_ids,
        "val_recipe_ids": val_ids,
        "test_recipe_ids": test_ids,
        "recipe_key_map": recipe_key_map,
    }


def prepare_model_data(data_path, split_obj, disable_derived_features=False):
    df = load_raw_dataset(data_path)
    feature_cols, target_cols = resolve_columns(df)

    if not disable_derived_features:
        df = add_derived_features(df, feature_cols)

    base_feat_order = [feature_cols[k] for k in FEATURE_KEYS]
    feat_order = base_feat_order + ([] if disable_derived_features else DERIVED_FEATURES)
    target_order = [target_cols[k] for k in TARGET_KEYS]
    work_df = df[feat_order + target_order].dropna().copy()

    recipe_cols = split_obj["recipe_columns"]
    recipe_key_map = {
        tuple(round(float(v), 6) for v in vals): int(k)
        for k, vals in split_obj["recipe_key_map"].items()
    }
    recipe_df_cols = [feature_cols[c] for c in recipe_cols]
    work_df["recipe_id"] = [
        recipe_key_map.get(row_to_key(row, recipe_df_cols), -1)
        for _, row in work_df.iterrows()
    ]

    unknown_count = int((work_df["recipe_id"] < 0).sum())
    if unknown_count > 0:
        raise ValueError(f"有 {unknown_count} 行无法匹配当前 split 的 recipe_key_map")

    train_ids = set(split_obj["train_recipe_ids"])
    val_ids = set(split_obj.get("val_recipe_ids", []))
    test_ids = set(split_obj["test_recipe_ids"])
    train_mask = work_df["recipe_id"].isin(train_ids)
    val_mask = work_df["recipe_id"].isin(val_ids)
    test_mask = work_df["recipe_id"].isin(test_ids)
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("划分后 train/val/test 至少有一个为空，请检查数据集和 seed")

    return {
        "df": df,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "feat_order": feat_order,
        "target_order": target_order,
        "target_keys": TARGET_KEYS,
        "derived_feat_order": DERIVED_FEATURES,
        "work_df": work_df,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "train_df": work_df.loc[train_mask].copy(),
        "val_df": work_df.loc[val_mask].copy(),
        "test_df": work_df.loc[test_mask].copy(),
        "split_obj": split_obj,
    }
