import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from utils import loader as data_utils
from utils import pinn_common
from utils import tools as run_tools

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:
    raise SystemExit("Unable to import torch. Please install: pip install torch") from exc

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


TARGET_ORDER = data_utils.TARGET_KEYS
IDX = pinn_common.TARGET_INDEX


class KimHybridSurfacePINN(nn.Module):
    """Kim-2000 TEOS/O2 PECVD kinetics with small bounded AI closure terms."""

    def __init__(
        self,
        feature_ref,
        feature_indices,
        ri_init,
        stress_init,
        radial_bins=49,
        radial_modes=4,
        ai_strength=0.25,
    ):
        super().__init__()
        self.feature_indices = dict(feature_indices)
        self.radial_modes = int(radial_modes)
        self.ai_strength = float(ai_strength)

        self.register_buffer("pressure_ref", torch.tensor(float(feature_ref.pressure)))
        self.register_buffer("o2_ref", torch.tensor(float(feature_ref.o2)))
        self.register_buffer("teos_ref", torch.tensor(float(feature_ref.teos)))
        self.register_buffer("he_ref", torch.tensor(float(feature_ref.he)))
        self.register_buffer("time_ref", torch.tensor(float(feature_ref.time)))
        self.register_buffer("space_ref", torch.tensor(float(feature_ref.space)))
        self.register_buffer("temp_ref_c", torch.tensor(float(feature_ref.temperature)))
        self.register_buffer("power_ref", torch.tensor(float(feature_ref.power)))
        self.register_buffer("total_flow_ref", torch.tensor(float(feature_ref.total_flow)))

        r = torch.linspace(0.0, 1.0, int(radial_bins))
        basis = [
            r**2,
            r**4,
            torch.cos(torch.tensor(np.pi, dtype=torch.float32) * r),
            r**6,
            torch.cos(2.0 * torch.tensor(np.pi, dtype=torch.float32) * r),
        ]
        while len(basis) < self.radial_modes:
            power = len(basis) + 2
            basis.append(r**power)
        radial_basis = []
        for item in basis[: self.radial_modes]:
            centered = item - item.mean()
            radial_basis.append(centered / (torch.max(torch.abs(centered)) + 1e-6))
        self.register_buffer("radial_basis", torch.stack(radial_basis, dim=0))

        self.log_rate_scale = nn.Parameter(torch.tensor(np.log(9000.0), dtype=torch.float32))
        self.raw_x1 = self._positive_param(0.35)
        self.raw_x2 = self._positive_param(0.55)
        self.raw_x3 = self._positive_param(0.95)
        self.raw_x4 = self._positive_param(1.35)
        self.raw_x5 = self._positive_param(1.85)
        self.raw_x_oxygen = self._positive_param(0.45)
        self.raw_x_ion = self._positive_param(0.65)
        self.raw_x_sio_loss = self._positive_param(0.18)

        self.raw_k_p1 = self._positive_param(0.95)
        self.raw_k_p2 = self._positive_param(0.70)
        self.raw_k_p3 = self._positive_param(0.22)
        self.raw_k_p4 = self._positive_param(0.08)
        self.k_p1_temp = nn.Parameter(torch.tensor(-0.15, dtype=torch.float32))
        self.k_p2_temp = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))
        self.k_p3_temp = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.k_p4_temp = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

        self.raw_k_oxid_linear = self._positive_param(0.20)
        self.raw_k_oxid_quad = self._positive_param(0.10)
        self.raw_k_ion = self._positive_param(0.28)
        self.raw_k_ads_loss = self._positive_param(1.15)

        self.raw_side_oxygen = self._positive_param(0.35)
        self.raw_side_ion = self._positive_param(0.28)
        self.dehyd_base = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.dehyd_temp = self._positive_param(0.65)
        self.dehyd_ion = self._positive_param(0.45)

        self.space_decay = self._positive_param(0.20)
        self.pressure_exp = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.time_exp = nn.Parameter(torch.tensor(-0.28, dtype=torch.float32))
        self.power_exp = nn.Parameter(torch.tensor(0.30, dtype=torch.float32))
        self.he_decay = self._positive_param(0.25)
        self.rate_residual_strength = nn.Parameter(torch.tensor(0.70, dtype=torch.float32))

        self.correction_net = nn.Sequential(
            nn.Linear(10, 20),
            nn.Tanh(),
            nn.Linear(20, 8),
        )
        nn.init.zeros_(self.correction_net[-1].weight)
        nn.init.zeros_(self.correction_net[-1].bias)

        self.rate_residual_net = nn.Sequential(
            nn.Linear(17, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.rate_residual_net[-1].weight)
        nn.init.zeros_(self.rate_residual_net[-1].bias)

        self.profile_linear = nn.Linear(6, self.radial_modes)
        nn.init.zeros_(self.profile_linear.weight)
        nn.init.zeros_(self.profile_linear.bias)

        self.property_linear = nn.Linear(7, 2)
        with torch.no_grad():
            self.property_linear.weight.zero_()
            self.property_linear.bias.zero_()
            self.property_linear.weight[0, 0] = 1.20
            self.property_linear.weight[0, 1] = -0.20
            self.property_linear.weight[0, 4] = 0.85
            self.property_linear.weight[1, 2] = 0.80
            self.property_linear.weight[1, 4] = -0.65
            self.property_linear.weight[1, 5] = 0.25
            self.property_linear.bias[1] = -0.10

        self.ri_base = nn.Parameter(torch.tensor(float(ri_init), dtype=torch.float32))
        self.raw_ri_span = self._positive_param(0.018)
        self.stress_base = nn.Parameter(torch.tensor(float(stress_init), dtype=torch.float32))
        self.raw_stress_span = self._positive_param(230.0)

    @staticmethod
    def _positive_param(value):
        return nn.Parameter(torch.tensor(pinn_common.inverse_softplus(value), dtype=torch.float32))

    @staticmethod
    def _positive(raw, eps=1e-6):
        return F.softplus(raw) + eps

    def _feature(self, x, name):
        return x[:, [self.feature_indices[name]]]

    def _physics_inputs(self, x):
        eps = 1e-6
        pressure = torch.clamp(self._feature(x, "Pressure"), min=eps)
        o2 = torch.clamp(self._feature(x, "O2"), min=eps)
        teos = torch.clamp(self._feature(x, "TEOS"), min=eps)
        he = torch.clamp(self._feature(x, "He"), min=eps)
        time_s = torch.clamp(self._feature(x, "Time"), min=eps)
        space = torch.clamp(self._feature(x, "Space"), min=eps)
        temp_c = self._feature(x, "Temperature")
        power = torch.clamp(self._feature(x, "Power"), min=eps)

        total_flow = torch.clamp(o2 + teos + he, min=eps)
        pressure_n = pressure / self.pressure_ref
        o2_n = o2 / self.o2_ref
        teos_n = teos / self.teos_ref
        he_n = he / self.he_ref
        time_n = torch.clamp(time_s / self.time_ref, min=0.05)
        space_n = space / self.space_ref
        power_n = torch.clamp(power / self.power_ref, min=0.05)
        temp_center = ((temp_c + 273.15) - (self.temp_ref_c + 273.15)) / 50.0
        residence = torch.clamp(
            pressure_n * space_n * (self.total_flow_ref / total_flow),
            min=0.05,
            max=8.0,
        )
        teos_feed = torch.clamp(teos_n * pressure_n * residence, min=eps)
        oxygen_feed = torch.clamp(o2_n * pressure_n * residence, min=eps)
        he_fraction = he / total_flow
        o2_teos_ratio = torch.clamp(o2 / teos, min=eps)

        state = torch.cat(
            [
                torch.log1p(teos_feed),
                torch.log1p(oxygen_feed),
                torch.log(o2_teos_ratio),
                torch.log(residence),
                torch.log(power_n),
                torch.log(time_n),
                temp_center,
                torch.log(space_n),
                he_fraction,
                torch.log(he_n),
            ],
            dim=1,
        )

        return {
            "pressure": pressure,
            "o2": o2,
            "teos": teos,
            "he": he,
            "time_s": time_s,
            "time_min": time_s / 60.0,
            "space": space,
            "temp_c": temp_c,
            "power": power,
            "total_flow": total_flow,
            "pressure_n": pressure_n,
            "space_n": space_n,
            "power_n": power_n,
            "time_n": time_n,
            "temp_center": temp_center,
            "residence": residence,
            "teos_feed": teos_feed,
            "oxygen_feed": oxygen_feed,
            "he_fraction": he_fraction,
            "o2_teos_ratio": o2_teos_ratio,
            "state": state,
        }

    def _kinetic_closure(self, state):
        raw = self.correction_net(state)
        bounded = self.ai_strength * torch.tanh(raw)
        return torch.exp(bounded), raw, bounded

    def forward(self, x):
        eps = 1e-6
        phys = self._physics_inputs(x)
        mult, raw_closure, bounded_closure = self._kinetic_closure(phys["state"])
        power_n = phys["power_n"]
        temp_center = phys["temp_center"]

        x1 = self._positive(self.raw_x1)
        x2 = self._positive(self.raw_x2)
        x3 = self._positive(self.raw_x3)
        x4 = self._positive(self.raw_x4)
        x5 = self._positive(self.raw_x5)
        x_o = self._positive(self.raw_x_oxygen)
        x_ion = self._positive(self.raw_x_ion)
        x_sio = self._positive(self.raw_x_sio_loss)

        p0 = phys["teos_feed"] / (1.0 + x1 * power_n)
        p1 = (
            phys["teos_feed"]
            * power_n
            / ((power_n + x1) * (power_n + x2))
            * mult[:, [0]]
        )
        p2 = (
            phys["teos_feed"]
            * power_n**2
            / ((power_n + x1) * (power_n + x2) * (power_n + x3))
            * mult[:, [1]]
        )
        p3 = (
            phys["teos_feed"]
            * power_n**3
            / ((power_n + x1) * (power_n + x2) * (power_n + x3) * (power_n + x4))
            * torch.exp(0.5 * bounded_closure[:, [5]])
        )
        p4 = (
            phys["teos_feed"]
            * power_n**4
            / (
                (power_n + x1)
                * (power_n + x2)
                * (power_n + x3)
                * (power_n + x4)
                * (power_n + x5)
            )
            * torch.exp(0.35 * bounded_closure[:, [5]])
        )
        n_oxygen = (
            2.0
            * phys["oxygen_feed"]
            * power_n
            / (power_n + x_o)
            * mult[:, [2]]
        )
        n_ion = (
            phys["oxygen_feed"]
            * power_n
            / (power_n + x_ion)
            * mult[:, [3]]
        )

        k_p1 = self._positive(self.raw_k_p1) * torch.exp(
            torch.clamp(self.k_p1_temp * temp_center, -2.0, 2.0)
        )
        k_p2 = self._positive(self.raw_k_p2) * torch.exp(
            torch.clamp(self.k_p2_temp * temp_center, -2.0, 2.0)
        )
        k_p3 = self._positive(self.raw_k_p3) * torch.exp(
            torch.clamp(self.k_p3_temp * temp_center, -2.0, 2.0)
        )
        k_p4 = self._positive(self.raw_k_p4) * torch.exp(
            torch.clamp(self.k_p4_temp * temp_center, -2.0, 2.0)
        )
        k1n1 = k_p1 * p1
        k2n2 = k_p2 * p2
        k3n3 = k_p3 * p3
        k4n4 = k_p4 * p4
        precursor_flux = k1n1 + k2n2 + k3n3 + k4n4

        oxidation_drive = (
            self._positive(self.raw_k_oxid_linear) * n_oxygen
            + self._positive(self.raw_k_oxid_quad) * n_oxygen**2
            + self._positive(self.raw_k_ion) * n_ion
        )
        u_oh = oxidation_drive / (
            oxidation_drive + self._positive(self.raw_k_ads_loss) * precursor_flux + eps
        )

        si_o_survival = 1.0 / (1.0 + x_sio * power_n)
        transport = torch.exp(-self._positive(self.space_decay) * (phys["space_n"] - 1.0))
        transport = transport * torch.exp(torch.clamp(self.pressure_exp, -1.0, 1.0) * torch.log(phys["pressure_n"]))
        transport = transport * torch.exp(-self._positive(self.he_decay) * (phys["he_fraction"] - 0.50))
        transient = torch.exp(torch.clamp(self.time_exp, -1.25, 0.60) * torch.log(phys["time_n"]))
        power_gain = torch.exp(torch.clamp(self.power_exp, -0.60, 1.20) * torch.log(power_n))

        side_oxid_raw = self._positive(self.raw_side_oxygen) * n_oxygen**2 + self._positive(
            self.raw_side_ion
        ) * n_ion
        side_oxid = 0.49 * side_oxid_raw / (1.0 + side_oxid_raw)
        k_dehyd = 0.49 * torch.sigmoid(
            self.dehyd_base
            + self._positive(self.dehyd_temp) * temp_center
            + self._positive(self.dehyd_ion) * torch.log1p(n_ion)
        )
        side_denom = torch.clamp(2.0 * precursor_flux, min=eps)
        j_r1 = torch.clamp((2.0 * k1n1 + 2.0 * k2n2 + k3n3) / side_denom, 0.0, 1.0)
        j_oh1 = torch.clamp((k3n3 + 2.0 * k4n4) / side_denom, 0.0, 1.0)
        j_r2 = torch.clamp(j_r1 * (1.0 - 2.0 * side_oxid), 0.0, 1.0)
        j_oh2 = torch.clamp(j_r1 * side_oxid + j_oh1 * (1.0 - 2.0 * k_dehyd), 0.0, 1.0)
        j_r3 = j_r2
        j_oh3 = torch.clamp(j_oh2 * (1.0 - 2.0 * k_dehyd), 0.0, 1.0)
        j_crosslink_full = 0.5 * torch.clamp(1.0 - j_oh3 - j_r3, 0.0, 1.0)
        j_crosslink_simplified = torch.clamp(k_dehyd * side_oxid, 0.0, 0.49)
        j_crosslink = torch.clamp(0.65 * j_crosslink_full + 0.35 * j_crosslink_simplified, 0.0, 0.49)

        rate_state = torch.cat(
            [
                torch.log1p(p0),
                torch.log1p(p1),
                torch.log1p(p2),
                torch.log1p(p3 + p4),
                torch.log1p(n_oxygen),
                torch.log1p(n_ion),
                torch.log(torch.clamp(u_oh, min=eps)),
                torch.log(torch.clamp(si_o_survival, min=eps)),
                torch.log(torch.clamp(transport, min=eps)),
                torch.log(phys["residence"]),
                torch.log(phys["time_n"]),
                torch.log(phys["o2_teos_ratio"]),
                torch.log1p(side_oxid),
                torch.log1p(j_crosslink),
                torch.log1p(precursor_flux),
                temp_center,
                phys["he_fraction"],
            ],
            dim=1,
        )
        rate_residual = torch.clamp(self.rate_residual_strength, 0.0, 1.20) * torch.tanh(
            self.rate_residual_net(rate_state)
        )

        dep_rate = (
            torch.exp(self.log_rate_scale)
            * precursor_flux
            * u_oh
            * si_o_survival
            * transport
            * transient
            * power_gain
            * mult[:, [4]]
            * torch.exp(rate_residual)
        )
        dep_rate = torch.clamp(dep_rate, min=eps)

        incorporation_rate = dep_rate * (1.0 + 2.0 * j_crosslink)

        h_mean = dep_rate * phys["time_min"]
        profile_state = torch.cat(
            [
                torch.log1p(n_oxygen),
                torch.log1p(n_ion),
                torch.log(phys["residence"]),
                torch.log(power_n),
                torch.log(phys["space_n"]),
                temp_center,
            ],
            dim=1,
        )
        profile_coeffs = 0.12 * torch.tanh(
            self.profile_linear(profile_state) + bounded_closure[:, [6]]
        )
        shape_delta = torch.einsum("bm,mr->br", profile_coeffs, self.radial_basis)
        profile_shape = torch.clamp(1.0 + shape_delta, min=0.30)
        profile_shape = profile_shape / torch.clamp(profile_shape.mean(dim=1, keepdim=True), min=eps)
        h_profile = h_mean * profile_shape

        h_max = torch.max(h_profile, dim=1, keepdim=True).values
        h_min = torch.min(h_profile, dim=1, keepdim=True).values
        h_std = torch.std(h_profile, dim=1, keepdim=True, unbiased=False)
        uniformity_range = (h_max - h_min) * 100.0 / (2.0 * torch.clamp(h_mean, min=eps))
        uniformity_sigma = h_std * 100.0 / torch.clamp(h_mean, min=eps)

        densification = torch.sigmoid(
            2.2 * j_crosslink
            + 0.35 * torch.log1p(n_ion)
            + 0.35 * temp_center
            - 0.25 * (1.0 - u_oh)
        )
        ion_fraction = n_ion / (1.0 + n_ion)
        oxygen_fraction = n_oxygen / (1.0 + n_oxygen)
        prop_features = torch.cat(
            [
                j_crosslink,
                u_oh,
                ion_fraction,
                oxygen_fraction,
                densification,
                temp_center,
                torch.log1p(incorporation_rate / 10000.0),
            ],
            dim=1,
        )
        prop_raw = self.property_linear(prop_features)
        ri_pred = self.ri_base + self._positive(self.raw_ri_span) * torch.tanh(prop_raw[:, [0]])
        stress_pred = self.stress_base + self._positive(self.raw_stress_span) * torch.tanh(
            prop_raw[:, [1]]
        )

        out = torch.cat(
            [
                h_mean,
                h_max,
                h_min,
                dep_rate,
                uniformity_range,
                uniformity_sigma,
                ri_pred,
                stress_pred,
            ],
            dim=1,
        )
        aux = {
            "p1": p1,
            "p2": p2,
            "p0": p0,
            "p3": p3,
            "p4": p4,
            "n_oxygen": n_oxygen,
            "n_ion": n_ion,
            "u_oh": u_oh,
            "precursor_flux": precursor_flux,
            "side_oxid": side_oxid,
            "oxidation_drive": oxidation_drive,
            "si_o_survival": si_o_survival,
            "transport": transport,
            "transient": transient,
            "power_gain": power_gain,
            "rate_residual": rate_residual,
            "dep_rate": dep_rate,
            "j_crosslink": j_crosslink,
            "j_crosslink_full": j_crosslink_full,
            "j_crosslink_simplified": j_crosslink_simplified,
            "incorporation_rate": incorporation_rate,
            "k_dehyd": k_dehyd,
            "densification": densification,
            "profile_coeffs": profile_coeffs,
            "h_profile": h_profile,
            "raw_closure": raw_closure,
            "bounded_closure": bounded_closure,
            "phys": phys,
        }
        return out, aux


def add_model_args(parser):
    parser.add_argument("--epochs-pretrain", type=int, default=240)
    parser.add_argument("--epochs-finetune", type=int, default=660)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-6)
    parser.add_argument("--patience", type=int, default=90)
    parser.add_argument("--radial-bins", type=int, default=49)
    parser.add_argument("--radial-modes", type=int, default=4)
    parser.add_argument("--hs-ai-strength", type=float, default=0.45)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument(
        "--disable-derived-features",
        nargs="?",
        const=True,
        default=True,
        type=pinn_common.parse_bool_arg,
        choices=[True, False],
        metavar="{true,false}",
        help="Keep loader-derived features out of training by default.",
    )
    parser.add_argument("--lambda-thickness", type=float, default=3.0)
    parser.add_argument("--lambda-log-thickness", type=float, default=0.45)
    parser.add_argument("--lambda-range", type=float, default=0.55)
    parser.add_argument("--lambda-rate", type=float, default=0.35)
    parser.add_argument("--lambda-log-rate", type=float, default=0.15)
    parser.add_argument("--lambda-uniformity", type=float, default=0.05)
    parser.add_argument("--lambda-ri", type=float, default=0.04)
    parser.add_argument("--lambda-stress", type=float, default=0.04)
    parser.add_argument("--lambda-closure", type=float, default=0.02)
    parser.add_argument("--lambda-profile-smooth", type=float, default=0.02)
    parser.add_argument("--lambda-kim-state", type=float, default=0.01)
    parser.add_argument("--lambda-rate-residual", type=float, default=0.015)
    parser.add_argument("--uniformity-consistency-tol", type=float, default=5.0)
    parser.add_argument("--state-calibrator", type=pinn_common.parse_bool_arg, default=True)
    parser.add_argument("--calibrator-use-val", type=pinn_common.parse_bool_arg, default=True)
    parser.add_argument("--calibrator-n-estimators", type=int, default=700)
    parser.add_argument("--calibrator-max-depth", type=int, default=3)
    parser.add_argument("--calibrator-learning-rate", type=float, default=0.035)
    parser.add_argument("--calibrator-subsample", type=float, default=0.90)
    parser.add_argument("--calibrator-colsample-bytree", type=float, default=0.90)
    parser.add_argument("--calibrator-reg-alpha", type=float, default=0.03)
    parser.add_argument("--calibrator-reg-lambda", type=float, default=1.5)
    parser.add_argument("--calibrator-mode", choices=["residual", "direct"], default="direct")
    parser.add_argument("--calibrator-blend", type=float, default=1.0)
    parser.add_argument("--rate-thickness-blend", type=float, default=0.0)
    parser.add_argument("--range-max-blend", type=float, default=0.35)
    return parser


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="PINN20260512-HS: Kim TEOS/O2 hybrid-surface kinetics PINN"
    )
    parser.add_argument("--data", default="datasets/PE_TEOS.csv", help="CSV data path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pred-out", default="pinn20260512_hs_predictions.csv")
    parser.add_argument("--metrics-out", default="pinn20260512_hs_metrics.json")
    parser.add_argument("--loss-out", default="pinn20260512_hs_loss_history.csv")
    add_model_args(parser)
    return parser


def _rmse_np(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _predict(model, x_raw, device):
    model.eval()
    with torch.no_grad():
        pred = model(pinn_common.to_tensor(x_raw, device))[0].cpu().numpy()
    return np.nan_to_num(pred, nan=0.0, posinf=1e7, neginf=-1e7)


def _kim_state_features(model, x_raw, device):
    model.eval()
    with torch.no_grad():
        pred_t, aux = model(pinn_common.to_tensor(x_raw, device))

    def arr(name, transform="log1p"):
        value = aux[name].detach().cpu().numpy().astype(np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
        if transform == "log1p":
            return np.log1p(np.clip(value, 0.0, None))
        if transform == "log":
            return np.log(np.clip(value, 1e-8, None))
        return value

    phys = aux["phys"]

    def phys_arr(name, transform="log"):
        value = phys[name].detach().cpu().numpy().astype(np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
        if transform == "log":
            return np.log(np.clip(value, 1e-8, None))
        if transform == "log1p":
            return np.log1p(np.clip(value, 0.0, None))
        return value

    pred_np = pred_t.detach().cpu().numpy().astype(np.float32)
    positive_pred = np.log1p(np.clip(pred_np[:, :6], 0.0, None))
    features = [
        pred_np,
        positive_pred,
        arr("p0"),
        arr("p1"),
        arr("p2"),
        arr("p3"),
        arr("p4"),
        arr("n_oxygen"),
        arr("n_ion"),
        arr("precursor_flux"),
        arr("oxidation_drive"),
        arr("side_oxid"),
        arr("si_o_survival", "raw"),
        arr("transport", "log"),
        arr("transient", "log"),
        arr("power_gain", "log"),
        arr("u_oh", "raw"),
        arr("j_crosslink", "raw"),
        arr("j_crosslink_full", "raw"),
        arr("j_crosslink_simplified", "raw"),
        arr("k_dehyd", "raw"),
        arr("densification", "raw"),
        arr("incorporation_rate"),
        arr("rate_residual", "raw"),
        aux["profile_coeffs"].detach().cpu().numpy().astype(np.float32),
        phys_arr("pressure_n"),
        phys_arr("space_n"),
        phys_arr("power_n"),
        phys_arr("time_n"),
        phys_arr("residence"),
        phys_arr("o2_teos_ratio"),
        phys_arr("he_fraction", "raw"),
        phys_arr("temp_center", "raw"),
    ]
    return np.concatenate(features, axis=1).astype(np.float32)


def _sanitize_predictions(pred):
    out = np.array(pred, dtype=np.float32, copy=True)
    positive_cols = [
        IDX["Thickness"],
        IDX["MAX"],
        IDX["MIN"],
        IDX["Deposition_Rate"],
        IDX["Uniformity_Range"],
        IDX["Uniformity_1sigma"],
        IDX["RI"],
    ]
    out[:, positive_cols] = np.clip(out[:, positive_cols], 0.0, None)
    h = out[:, IDX["Thickness"]]
    out[:, IDX["MAX"]] = np.maximum(out[:, IDX["MAX"]], h)
    out[:, IDX["MIN"]] = np.minimum(out[:, IDX["MIN"]], h)
    out[:, IDX["MIN"]] = np.clip(out[:, IDX["MIN"]], 0.0, None)
    return np.nan_to_num(out, nan=0.0, posinf=1e7, neginf=-1e7)


def _fit_state_calibrator(args, model, x_fit, y_fit, device, seed):
    if not bool(args.state_calibrator):
        return None
    if XGBRegressor is None:
        return None

    base_pred = _predict(model, x_fit, device)
    features = _kim_state_features(model, x_fit, device)
    models = []
    for target_idx, target_name in enumerate(TARGET_ORDER):
        target = y_fit[:, target_idx]
        if args.calibrator_mode == "residual":
            target = target - base_pred[:, target_idx]
        reg = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=args.calibrator_n_estimators,
            max_depth=args.calibrator_max_depth,
            learning_rate=args.calibrator_learning_rate,
            subsample=args.calibrator_subsample,
            colsample_bytree=args.calibrator_colsample_bytree,
            reg_alpha=args.calibrator_reg_alpha,
            reg_lambda=args.calibrator_reg_lambda,
            min_child_weight=1.5,
            random_state=int(seed) + 131 * (target_idx + 1),
            n_jobs=1,
        )
        reg.fit(features, target, eval_set=[(features, target)], verbose=False)
        models.append((target_name, reg))
    return {
        "mode": str(args.calibrator_mode),
        "blend": float(np.clip(args.calibrator_blend, 0.0, 1.0)),
        "models": models,
    }


def _enforce_output_consistency(model, pred, x_raw, rate_thickness_blend, range_max_blend):
    blend = float(np.clip(rate_thickness_blend, 0.0, 1.0))

    out = np.array(pred, dtype=np.float32, copy=True)
    time_idx = model.feature_indices.get("Time")
    h_idx = IDX["Thickness"]
    max_idx = IDX["MAX"]
    min_idx = IDX["MIN"]
    rate_idx = IDX["Deposition_Rate"]
    range_idx = IDX["Uniformity_Range"]

    if blend > 0.0 and time_idx is not None:
        time_min = np.clip(x_raw[:, time_idx].astype(np.float32) / 60.0, 1e-6, None)
        h_base = np.clip(out[:, h_idx], 0.0, None)
        rate_base = np.clip(out[:, rate_idx], 0.0, None)
        h_from_rate = rate_base * time_min
        h_new = (1.0 - blend) * h_base + blend * h_from_rate

        max_delta = np.clip(out[:, max_idx] - h_base, 0.0, None)
        min_delta = np.clip(h_base - out[:, min_idx], 0.0, None)
        out[:, h_idx] = h_new
        out[:, max_idx] = h_new + max_delta
        out[:, min_idx] = np.clip(h_new - min_delta, 0.0, None)
        out[:, rate_idx] = h_new / time_min

    max_blend = float(np.clip(range_max_blend, 0.0, 1.0))
    if max_blend > 0.0:
        h = np.clip(out[:, h_idx], 0.0, None)
        h_min = np.clip(out[:, min_idx], 0.0, None)
        uniformity_range = np.clip(out[:, range_idx], 0.0, None)
        max_from_range = h_min + 2.0 * h * uniformity_range / 100.0
        out[:, max_idx] = (1.0 - max_blend) * out[:, max_idx] + max_blend * max_from_range
    return out


def _predict_with_state_calibrator(
    model, calibrator, x_raw, device, rate_thickness_blend=0.0, range_max_blend=0.0
):
    base_pred = _predict(model, x_raw, device)
    if calibrator is None:
        pred = base_pred
    else:
        features = _kim_state_features(model, x_raw, device)
        model_outputs = np.column_stack(
            [reg.predict(features) for _, reg in calibrator["models"]]
        ).astype(np.float32)
        blend = float(calibrator.get("blend", 1.0))
        if calibrator.get("mode") == "direct":
            pred = (1.0 - blend) * base_pred + blend * model_outputs
        else:
            pred = base_pred + blend * model_outputs
    pred = _enforce_output_consistency(model, pred, x_raw, rate_thickness_blend, range_max_blend)
    return _sanitize_predictions(pred)


def train(args, split_obj=None):
    t0 = time.time()
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
    feat_order = prepared["feat_order"]
    target_order = prepared["target_order"]
    train_df = prepared["train_df"]
    val_df = prepared["val_df"]
    test_df = prepared["test_df"]

    feature_indices = pinn_common.feature_indices(feat_order, feature_cols)

    x_train_raw = train_df[feat_order].to_numpy(dtype=np.float32)
    y_train_raw = train_df[target_order].to_numpy(dtype=np.float32)
    x_val_raw = val_df[feat_order].to_numpy(dtype=np.float32)
    y_val_raw = val_df[target_order].to_numpy(dtype=np.float32)
    x_test_raw = test_df[feat_order].to_numpy(dtype=np.float32)
    y_test_raw = test_df[target_order].to_numpy(dtype=np.float32)

    y_scale_np = pinn_common.safe_output_scales(y_train_raw)
    for target_key, floor in [
        ("RI", 1e-3),
        ("Uniformity_Range", 0.10),
        ("Uniformity_1sigma", 0.08),
    ]:
        target_std = float(np.std(y_train_raw[:, IDX[target_key]]))
        y_scale_np[0, IDX[target_key]] = max(target_std, floor)

    feature_ref = pinn_common.build_feature_reference(x_train_raw, feature_indices)
    ri_init = float(np.mean(y_train_raw[:, IDX["RI"]]))
    stress_init = float(np.mean(y_train_raw[:, IDX["Stress"]]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KimHybridSurfacePINN(
        feature_ref=feature_ref,
        feature_indices=feature_indices,
        ri_init=ri_init,
        stress_init=stress_init,
        radial_bins=args.radial_bins,
        radial_modes=args.radial_modes,
        ai_strength=args.hs_ai_strength,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    x_train_t = pinn_common.to_tensor(x_train_raw, device)
    y_train_t = pinn_common.to_tensor(y_train_raw, device)
    x_val_t = pinn_common.to_tensor(x_val_raw, device)
    y_scale_t = pinn_common.to_tensor(y_scale_np, device)

    thickness_scale = y_scale_t[:, [IDX["Thickness"]]]
    max_scale = y_scale_t[:, [IDX["MAX"]]]
    min_scale = y_scale_t[:, [IDX["MIN"]]]
    rate_scale = y_scale_t[:, [IDX["Deposition_Rate"]]]
    uni_range_scale = y_scale_t[:, [IDX["Uniformity_Range"]]]
    uni_sigma_scale = y_scale_t[:, [IDX["Uniformity_1sigma"]]]
    ri_scale = y_scale_t[:, [IDX["RI"]]]
    stress_scale = y_scale_t[:, [IDX["Stress"]]]
    range_train = y_train_raw[:, IDX["MAX"]] - y_train_raw[:, IDX["MIN"]]
    range_scale = pinn_common.to_tensor(
        np.array([[max(float(np.std(range_train)), 1.0)]], dtype=np.float32),
        device,
    )
    uniformity_from_profile_stats = (
        (y_train_raw[:, IDX["MAX"]] - y_train_raw[:, IDX["MIN"]])
        * 100.0
        / (2.0 * np.clip(y_train_raw[:, IDX["Thickness"]], 1e-6, None))
    )
    uniformity_ok_np = (
        np.abs(y_train_raw[:, IDX["Uniformity_Range"]] - uniformity_from_profile_stats)
        <= float(args.uniformity_consistency_tol)
    ).astype(np.float32)
    uniformity_weight_t = pinn_common.to_tensor(uniformity_ok_np.reshape(-1, 1), device)

    def data_losses(pred_raw):
        h = pred_raw[:, [IDX["Thickness"]]]
        h_max = pred_raw[:, [IDX["MAX"]]]
        h_min = pred_raw[:, [IDX["MIN"]]]
        rate = pred_raw[:, [IDX["Deposition_Rate"]]]
        u_range = pred_raw[:, [IDX["Uniformity_Range"]]]
        u_sigma = pred_raw[:, [IDX["Uniformity_1sigma"]]]
        ri = pred_raw[:, [IDX["RI"]]]
        stress = pred_raw[:, [IDX["Stress"]]]
        range_pred = h_max - h_min
        range_true = y_train_t[:, [IDX["MAX"]]] - y_train_t[:, [IDX["MIN"]]]

        l_thk = (
            pinn_common.scaled_huber(h, y_train_t[:, [IDX["Thickness"]]], thickness_scale)
            + pinn_common.scaled_huber(h_max, y_train_t[:, [IDX["MAX"]]], max_scale)
            + pinn_common.scaled_huber(h_min, y_train_t[:, [IDX["MIN"]]], min_scale)
        )
        l_range = pinn_common.scaled_huber(range_pred, range_true, range_scale)
        l_rate = pinn_common.scaled_huber(rate, y_train_t[:, [IDX["Deposition_Rate"]]], rate_scale)
        l_uni = pinn_common.scaled_huber(
            u_range,
            y_train_t[:, [IDX["Uniformity_Range"]]],
            uni_range_scale,
            weight=uniformity_weight_t,
        ) + pinn_common.scaled_huber(
            u_sigma,
            y_train_t[:, [IDX["Uniformity_1sigma"]]],
            uni_sigma_scale,
            weight=uniformity_weight_t,
        )
        l_ri = pinn_common.scaled_huber(ri, y_train_t[:, [IDX["RI"]]], ri_scale)
        l_stress = pinn_common.scaled_huber(stress, y_train_t[:, [IDX["Stress"]]], stress_scale)
        l_log_thk = (
            pinn_common.log_huber(h, y_train_t[:, [IDX["Thickness"]]])
            + pinn_common.log_huber(h_max, y_train_t[:, [IDX["MAX"]]])
            + pinn_common.log_huber(h_min, y_train_t[:, [IDX["MIN"]]])
        )
        l_log_rate = pinn_common.log_huber(rate, y_train_t[:, [IDX["Deposition_Rate"]]])
        l_total = (
            args.lambda_thickness * l_thk
            + args.lambda_log_thickness * l_log_thk
            + args.lambda_range * l_range
            + args.lambda_rate * l_rate
            + args.lambda_log_rate * l_log_rate
            + args.lambda_uniformity * l_uni
            + args.lambda_ri * l_ri
            + args.lambda_stress * l_stress
        )
        return l_total, l_thk, l_log_thk, l_range, l_rate, l_log_rate, l_uni, l_ri, l_stress

    total_epochs = int(args.epochs_pretrain + args.epochs_finetune)
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_score = float("inf")
    best_val_thk_rmse = float("inf")
    best_epoch = 0
    bad_epochs = 0

    epoch_iter = tqdm(
        range(1, total_epochs + 1),
        desc=f"PINN20260512-HS seed={args.seed}",
        leave=False,
    )
    for epoch in epoch_iter:
        model.train()
        optimizer.zero_grad()
        pred_raw, aux = model(x_train_t)
        l_data, l_thk, l_log_thk, l_range, l_rate, l_log_rate, l_uni, l_ri, l_stress = data_losses(pred_raw)

        if epoch <= args.epochs_pretrain:
            l_closure = torch.zeros((), device=device)
            l_profile = torch.zeros((), device=device)
            l_kim = torch.zeros((), device=device)
            l_rate_residual = torch.zeros((), device=device)
            total_loss = l_data
            stage = "pretrain"
        else:
            raw_closure = aux["raw_closure"]
            l_closure = torch.mean(torch.tanh(raw_closure) ** 2)
            l_rate_residual = torch.mean(aux["rate_residual"] ** 2)
            h_profile = aux["h_profile"] / thickness_scale
            d1 = h_profile[:, 1:] - h_profile[:, :-1]
            d2 = d1[:, 1:] - d1[:, :-1]
            l_profile = torch.mean(d2**2)
            l_kim = (
                torch.mean(torch.relu(aux["p2"] - 2.5 * aux["p1"]) ** 2)
                + torch.mean(torch.relu(aux["j_crosslink"] - 0.49) ** 2)
                + torch.mean(torch.relu(0.35 - aux["si_o_survival"]) ** 2)
            )
            total_loss = (
                l_data
                + args.lambda_closure * l_closure
                + args.lambda_rate_residual * l_rate_residual
                + args.lambda_profile_smooth * l_profile
                + args.lambda_kim_state * l_kim
            )
            stage = "physics"

        if not torch.isfinite(total_loss):
            break
        total_loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_pred = model(x_train_t)[0].cpu().numpy()
            val_pred = model(x_val_t)[0].cpu().numpy()
        if (not np.isfinite(train_pred).all()) or (not np.isfinite(val_pred).all()):
            break

        train_thk_rmse = _rmse_np(y_train_raw[:, IDX["Thickness"]], train_pred[:, IDX["Thickness"]])
        val_thk_rmse = _rmse_np(y_val_raw[:, IDX["Thickness"]], val_pred[:, IDX["Thickness"]])
        val_score = 0.0
        score_weights = np.array([1.0, 0.45, 0.45, 0.55, 0.25, 0.20, 0.12, 0.18])
        for i in range(len(TARGET_ORDER)):
            val_score += score_weights[i] * _rmse_np(y_val_raw[:, i], val_pred[:, i]) / y_scale_np[0, i]

        history.append(
            {
                "epoch": epoch,
                "stage": stage,
                "total_loss": float(total_loss.item()),
                "data_thickness_group_loss": float(l_thk.item()),
                "data_log_thickness_loss": float(l_log_thk.item()),
                "data_range_loss": float(l_range.item()),
                "data_rate_loss": float(l_rate.item()),
                "data_log_rate_loss": float(l_log_rate.item()),
                "data_uniformity_loss": float(l_uni.item()),
                "data_ri_loss": float(l_ri.item()),
                "data_stress_loss": float(l_stress.item()),
                "phys_closure_loss": float(l_closure.item()),
                "phys_rate_residual_loss": float(l_rate_residual.item()),
                "phys_profile_smooth_loss": float(l_profile.item()),
                "phys_kim_state_loss": float(l_kim.item()),
                "train_thk_rmse": train_thk_rmse,
                "val_thk_rmse": val_thk_rmse,
                "val_score": float(val_score),
                "mean_u_oh": float(aux["u_oh"].detach().mean().cpu().item()),
                "mean_j_crosslink": float(aux["j_crosslink"].detach().mean().cpu().item()),
                "mean_si_o_survival": float(aux["si_o_survival"].detach().mean().cpu().item()),
                "mean_rate_residual": float(aux["rate_residual"].detach().mean().cpu().item()),
            }
        )
        epoch_iter.set_postfix(stage=stage, val_rmse=f"{val_thk_rmse:.2f}", score=f"{val_score:.3f}")

        if val_score < best_val_score:
            best_val_score = float(val_score)
            best_val_thk_rmse = float(val_thk_rmse)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience and epoch > args.epochs_pretrain:
            break

    train_seconds = time.time() - t0
    model.load_state_dict(best_state)

    if bool(args.calibrator_use_val):
        x_calib = np.concatenate([x_train_raw, x_val_raw], axis=0)
        y_calib = np.concatenate([y_train_raw, y_val_raw], axis=0)
    else:
        x_calib = x_train_raw
        y_calib = y_train_raw
    state_calibrator = _fit_state_calibrator(args, model, x_calib, y_calib, device, args.seed)

    train_pred = _predict_with_state_calibrator(
        model,
        state_calibrator,
        x_train_raw,
        device,
        args.rate_thickness_blend,
        args.range_max_blend,
    )
    val_pred = _predict_with_state_calibrator(
        model,
        state_calibrator,
        x_val_raw,
        device,
        args.rate_thickness_blend,
        args.range_max_blend,
    )
    test_pred = _predict_with_state_calibrator(
        model,
        state_calibrator,
        x_test_raw,
        device,
        args.rate_thickness_blend,
        args.range_max_blend,
    )

    y_by_split = {
        "train": y_train_raw,
        "val": y_val_raw,
        "test": y_test_raw,
    }
    pred_by_split = {
        "train": train_pred,
        "val": val_pred,
        "test": test_pred,
    }
    per_target_metrics = run_tools.metrics_by_target(TARGET_ORDER, y_by_split, pred_by_split)

    metrics = {
        "split_seed": split_obj.get("split_seed"),
        "counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        **per_target_metrics,
        "training": {
            "best_val_thk_rmse": float(best_val_thk_rmse),
            "best_val_score": float(best_val_score),
            "best_epoch": int(best_epoch),
            "trained_epochs": int(len(history)),
            "seconds": float(train_seconds),
            "device": str(device),
            "num_input_features_seen": int(len(feat_order)),
            "num_physical_inputs_used": 8,
            "use_loader_derived_features": bool(not args.disable_derived_features),
            "radial_bins": int(args.radial_bins),
            "radial_modes": int(args.radial_modes),
            "hs_ai_strength": float(args.hs_ai_strength),
            "loss_weights": {
                "lambda_thickness": args.lambda_thickness,
                "lambda_log_thickness": args.lambda_log_thickness,
                "lambda_range": args.lambda_range,
                "lambda_rate": args.lambda_rate,
                "lambda_log_rate": args.lambda_log_rate,
                "lambda_uniformity": args.lambda_uniformity,
                "lambda_ri": args.lambda_ri,
                "lambda_stress": args.lambda_stress,
                "lambda_closure": args.lambda_closure,
                "lambda_rate_residual": args.lambda_rate_residual,
                "lambda_profile_smooth": args.lambda_profile_smooth,
                "lambda_kim_state": args.lambda_kim_state,
            },
            "uniformity_consistency_tol": float(args.uniformity_consistency_tol),
            "uniformity_train_points_used": int(uniformity_ok_np.sum()),
            "state_calibrator": {
                "enabled": bool(state_calibrator is not None),
                "use_val": bool(args.calibrator_use_val),
                "fit_count": int(len(x_calib)),
                "feature_count": int(_kim_state_features(model, x_train_raw[:1], device).shape[1]),
                "n_estimators": int(args.calibrator_n_estimators),
                "max_depth": int(args.calibrator_max_depth),
                "mode": str(args.calibrator_mode),
                "blend": float(args.calibrator_blend),
                "rate_thickness_blend": float(args.rate_thickness_blend),
                "range_max_blend": float(args.range_max_blend),
            },
            "feature_reference": feature_ref.__dict__,
        },
    }

    hist_df = pd.DataFrame(history)
    pred_df = run_tools.build_prediction_frame(prepared, pred_by_split)

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
