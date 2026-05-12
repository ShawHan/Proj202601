import argparse
from dataclasses import dataclass

import numpy as np
import torch

from utils import loader as data_utils


TARGET_INDEX = {name: idx for idx, name in enumerate(data_utils.TARGET_KEYS)}


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("boolean argument must be true/false")


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


def safe_output_scales(y_train_raw, floor=1.0):
    scales = y_train_raw.std(axis=0).astype(np.float32)
    scales[scales < floor] = floor
    return scales.reshape(1, -1)


def scaled_mse(pred, target, scale):
    return torch.mean(((pred - target) / scale) ** 2)


def scaled_huber(pred, target, scale, beta=1.0, weight=None):
    err = (pred - target) / scale
    abs_err = torch.abs(err)
    quad = torch.minimum(abs_err, torch.tensor(float(beta), dtype=err.dtype, device=err.device))
    loss = 0.5 * quad**2 / float(beta) + (abs_err - quad)
    if weight is not None:
        loss = loss * weight
        return loss.sum() / torch.clamp(weight.sum(), min=1.0)
    return torch.mean(loss)


def log_huber(pred, target, beta=0.08, weight=None):
    err = torch.log1p(torch.clamp(pred, min=0.0)) - torch.log1p(torch.clamp(target, min=0.0))
    abs_err = torch.abs(err)
    quad = torch.minimum(abs_err, torch.tensor(float(beta), dtype=err.dtype, device=err.device))
    loss = 0.5 * quad**2 / float(beta) + (abs_err - quad)
    if weight is not None:
        loss = loss * weight
        return loss.sum() / torch.clamp(weight.sum(), min=1.0)
    return torch.mean(loss)


def inverse_softplus(value):
    value = np.asarray(value, dtype=np.float32)
    value = np.maximum(value, 1e-6)
    if value.ndim == 0:
        if float(value) > 20.0:
            return np.float32(value)
        return np.float32(np.log(np.expm1(value)))
    stable = value.copy()
    small_mask = value <= 20.0
    stable[small_mask] = np.log(np.expm1(value[small_mask]))
    return stable.astype(np.float32)


def feature_indices(feat_order, feature_cols):
    return {name: feat_order.index(feature_cols[name]) for name in data_utils.FEATURE_KEYS}


@dataclass(frozen=True)
class FeatureReference:
    pressure: float
    o2: float
    teos: float
    he: float
    time: float
    space: float
    temperature: float
    power: float
    total_flow: float


def build_feature_reference(x_raw, indices):
    values = {
        "pressure": x_raw[:, indices["Pressure"]],
        "o2": x_raw[:, indices["O2"]],
        "teos": x_raw[:, indices["TEOS"]],
        "he": x_raw[:, indices["He"]],
        "time": x_raw[:, indices["Time"]],
        "space": x_raw[:, indices["Space"]],
        "temperature": x_raw[:, indices["Temperature"]],
        "power": x_raw[:, indices["Power"]],
    }
    values["total_flow"] = values["o2"] + values["teos"] + values["he"]
    refs = {
        key: float(np.median(np.asarray(val, dtype=np.float32)))
        for key, val in values.items()
    }
    for key, value in refs.items():
        if abs(value) < 1e-6:
            refs[key] = 1.0
    return FeatureReference(**refs)
