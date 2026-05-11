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
    import torch.nn.functional as F
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
    if raw_arg == default_name:
        out_dir = Path("output") / split_tag / model_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / default_name

    out_path = Path(raw_arg)
    if not out_path.is_absolute() and out_path.parent == Path("."):
        out_dir = Path("output") / split_tag / model_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / out_path.name
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("布尔参数必须是 true/false")


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


class ReactionInformedMsPINN(nn.Module):
    def __init__(
        self,
        in_dim=8,
        hidden=160,
        radial_bins=33,
        radial_modes=3,
        thickness_residual_scale=0.0,
    ):
        super().__init__()
        self.radial_bins = radial_bins
        self.radial_modes = radial_modes

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 96),
            nn.SiLU(),
        )

        self.species_head = nn.Sequential(nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 5))
        self.plasma_head = nn.Sequential(nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 6))
        self.kinetics_head = nn.Sequential(nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 6))
        self.profile_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.Tanh(),
            nn.Linear(64, radial_modes),
        )
        self.gas_field_head = nn.Sequential(
            nn.Linear(96 + 3 + 7, 96),
            nn.SiLU(),
            nn.Linear(96, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        self.surface_field_head = nn.Sequential(
            nn.Linear(96 + 2 + 3 + 4, 96),
            nn.SiLU(),
            nn.Linear(96, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )
        self.h_field_head = nn.Sequential(
            nn.Linear(96 + 2 + 3 + 2 + 1, 96),
            nn.SiLU(),
            nn.Linear(96, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        self.thickness_residual_head = nn.Sequential(
            nn.Linear(96, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.ri_bias_head = nn.Sequential(nn.Linear(96, 32), nn.Tanh(), nn.Linear(32, 1))
        self.stress_bias_head = nn.Sequential(nn.Linear(96, 32), nn.Tanh(), nn.Linear(32, 1))

        nn.init.zeros_(self.thickness_residual_head[-1].weight)
        nn.init.zeros_(self.thickness_residual_head[-1].bias)

        r = torch.linspace(0.0, 1.0, radial_bins)
        basis = [r**2, r**4, torch.cos(torch.tensor(np.pi, dtype=r.dtype) * r)]
        while len(basis) < radial_modes:
            basis.append(r ** (len(basis) + 2))
        radial_basis = torch.stack(
            [
                (b - b.mean()) / (torch.max(torch.abs(b - b.mean())) + 1e-6)
                for b in basis[:radial_modes]
            ],
            dim=0,
        )
        self.register_buffer("r_grid", r)
        self.register_buffer("radial_basis", radial_basis)
        self.register_buffer(
            "thickness_residual_scale",
            torch.tensor(float(thickness_residual_scale), dtype=torch.float32),
        )

    def _continuous_radial_basis(self, r: torch.Tensor):
        basis = [r**2 - (1.0 / 3.0), r**4 - (1.0 / 5.0), torch.cos(np.pi * r)]
        while len(basis) < self.radial_modes:
            power = len(basis) + 2
            basis.append(r**power - (1.0 / (power + 1.0)))

        normalized = []
        for b in basis[: self.radial_modes]:
            scale = torch.amax(torch.abs(b), dim=1, keepdim=True) + 1e-6
            normalized.append(b / scale)
        return torch.cat(normalized, dim=-1)

    @staticmethod
    def _expand_cond(tensor: torch.Tensor, points: int):
        return tensor.unsqueeze(1).expand(-1, points, -1)

    def evaluate_gas_fields(self, aux, gas_coords: torch.Tensor):
        points = gas_coords.shape[1]
        z_coord = gas_coords[..., [0]]
        r_coord = gas_coords[..., [1]]
        tau_coord = gas_coords[..., [2]]

        latent = self._expand_cond(aux["latent"], points)
        cond = torch.cat(
            [
                self._expand_cond(aux["sio_eff"], points),
                self._expand_cond(aux["o_eff"], points),
                self._expand_cond(aux["o2_ion_eff"], points),
                self._expand_cond(aux["phi_p"], points),
                self._expand_cond(aux["tau_res"], points),
                self._expand_cond(aux["thermal_budget"], points),
                self._expand_cond(aux["n_e_proxy"], points),
            ],
            dim=-1,
        )
        gas_in = torch.cat([latent, gas_coords, cond], dim=-1)
        gas_raw = self.gas_field_head(gas_in.reshape(-1, gas_in.shape[-1])).reshape(
            gas_coords.shape[0],
            points,
            3,
        )

        sio_base = torch.log1p(self._expand_cond(aux["sio_eff"], points))
        o_base = torch.log1p(self._expand_cond(aux["o_eff"], points))
        ion_base = torch.log1p(self._expand_cond(aux["o2_ion_eff"], points))
        phi_p = self._expand_cond(aux["phi_p"], points)
        tau_res = self._expand_cond(aux["tau_res"], points)

        c_sio = F.softplus(
            gas_raw[..., [0]] + sio_base + 0.12 * phi_p * (1.0 - z_coord) - 0.08 * tau_coord
        )
        c_o = F.softplus(
            gas_raw[..., [1]] + o_base + 0.10 * phi_p - 0.10 * z_coord - 0.05 * r_coord**2
        )
        c_ion = F.softplus(
            gas_raw[..., [2]]
            + ion_base
            + 0.16 * phi_p
            - 0.18 * z_coord
            - 0.08 * r_coord**2
            + 0.05 * tau_res
        )

        return {
            "c_sio": c_sio,
            "c_o": c_o,
            "c_ion": c_ion,
        }

    def evaluate_surface_fields(self, aux, surface_coords: torch.Tensor, gas_surface=None):
        points = surface_coords.shape[1]
        if gas_surface is None:
            z_coord = torch.ones(
                surface_coords.shape[0],
                points,
                1,
                device=surface_coords.device,
                dtype=surface_coords.dtype,
            )
            gas_surface = self.evaluate_gas_fields(aux, torch.cat([z_coord, surface_coords], dim=-1))

        latent = self._expand_cond(aux["latent"], points)
        cond = torch.cat(
            [
                gas_surface["c_sio"],
                gas_surface["c_o"],
                gas_surface["c_ion"],
                self._expand_cond(aux["phi_p"], points),
                self._expand_cond(aux["tau_res"], points),
                self._expand_cond(aux["thermal_budget"], points),
                self._expand_cond(aux["phi_nuc"], points),
            ],
            dim=-1,
        )
        surface_in = torch.cat([latent, surface_coords, cond], dim=-1)
        surface_raw = self.surface_field_head(
            surface_in.reshape(-1, surface_in.shape[-1])
        ).reshape(surface_coords.shape[0], points, 2)

        theta = torch.sigmoid(
            surface_raw[..., [0]]
            + 0.32 * gas_surface["c_o"]
            + 0.18 * gas_surface["c_ion"]
            - 0.10 * gas_surface["c_sio"]
        )
        eta = torch.sigmoid(
            surface_raw[..., [1]]
            + 0.26 * gas_surface["c_sio"] * theta
            + 0.14 * theta**2
            + 0.08 * self._expand_cond(aux["thermal_budget"], points)
        )

        return {
            "theta": theta,
            "eta": eta,
        }

    def evaluate_surface_rate(self, aux, surface_coords: torch.Tensor, gas_surface=None, surface_fields=None):
        points = surface_coords.shape[1]
        if gas_surface is None:
            z_coord = torch.ones(
                surface_coords.shape[0],
                points,
                1,
                device=surface_coords.device,
                dtype=surface_coords.dtype,
            )
            gas_surface = self.evaluate_gas_fields(aux, torch.cat([z_coord, surface_coords], dim=-1))
        if surface_fields is None:
            surface_fields = self.evaluate_surface_fields(aux, surface_coords, gas_surface)

        radial_basis = self._continuous_radial_basis(surface_coords[..., [0]])
        profile_coeffs = self._expand_cond(aux["profile_coeffs"], points)
        center_edge_bias = 0.05 * torch.tanh(
            0.4 * self._expand_cond(aux["o2_ion_eff"], points)
            - 0.15 * self._expand_cond(aux["tau_res"], points)
        )
        radial_factor = torch.clamp(
            1.0 + torch.sum(profile_coeffs * radial_basis, dim=-1, keepdim=True) + center_edge_bias * radial_basis[..., [0]],
            min=0.2,
        )

        alpha1 = 0.30 + 0.20 * torch.sigmoid(self._expand_cond(aux["k_oxid"], points))
        alpha2 = 0.24 + 0.18 * torch.sigmoid(self._expand_cond(aux["k_ads"], points))
        alpha3 = 0.16 + 0.14 * torch.sigmoid(self._expand_cond(aux["k_condense"], points))
        temp_factor = 0.75 + 0.25 * torch.sigmoid(self._expand_cond(aux["thermal_budget"], points))

        rate_drive = (
            alpha1 * gas_surface["c_sio"] * gas_surface["c_o"] * surface_fields["theta"]
            + alpha2 * gas_surface["c_sio"] * surface_fields["theta"]
            + alpha3 * surface_fields["theta"] ** 2
        )
        dep_rate = torch.relu(
            temp_factor
            * radial_factor
            * rate_drive
            * (1.0 + 0.12 * surface_fields["eta"] + 0.08 * self._expand_cond(aux["phi_nuc"], points))
            - 0.035 * self._expand_cond(aux["k_etch"], points) * gas_surface["c_ion"]
        )
        return dep_rate

    def evaluate_h_field(self, aux, surface_coords, gas_surface, surface_fields, time_minutes):
        points = surface_coords.shape[1]
        latent = self._expand_cond(aux["latent"], points)
        time_cond = self._expand_cond(time_minutes, points)
        surface_rate = self.evaluate_surface_rate(aux, surface_coords, gas_surface, surface_fields)

        h_in = torch.cat(
            [
                latent,
                surface_coords,
                gas_surface["c_sio"],
                gas_surface["c_o"],
                gas_surface["c_ion"],
                surface_fields["theta"],
                surface_fields["eta"],
                time_cond,
            ],
            dim=-1,
        )
        raw = self.h_field_head(h_in.reshape(-1, h_in.shape[-1]))
        raw = raw.reshape(surface_coords.shape[0], points, 1)

        tau = surface_coords[..., [1]]
        base_h = surface_rate * time_cond * tau
        correction = 1.0 + 0.03 * torch.tanh(raw)
        h = torch.relu(base_h * correction)
        return h

    def forward(self, x, time_minutes):
        eps = 1e-6
        z = self.encoder(x)

        species_raw = self.species_head(z)
        sio_eff = F.softplus(species_raw[:, [0]])
        o_eff = F.softplus(species_raw[:, [1]])
        o2_ion_eff = F.softplus(species_raw[:, [2]])
        tau_res = F.softplus(species_raw[:, [3]])
        thermal_budget = F.softplus(species_raw[:, [4]])

        plasma_raw = self.plasma_head(z)
        phi_p = F.softplus(plasma_raw[:, [0]])
        v_z = F.softplus(plasma_raw[:, [1]])
        d_sio = F.softplus(plasma_raw[:, [2]])
        d_o = F.softplus(plasma_raw[:, [3]])
        d_ion = F.softplus(plasma_raw[:, [4]])
        n_e_proxy = F.softplus(plasma_raw[:, [5]])

        kinetics_raw = self.kinetics_head(z)
        k_ads = F.softplus(kinetics_raw[:, [0]])
        k_oxid = F.softplus(kinetics_raw[:, [1]])
        k_condense = F.softplus(kinetics_raw[:, [2]])
        k_etch = F.softplus(kinetics_raw[:, [3]])
        k_sat = F.softplus(kinetics_raw[:, [4]])
        phi_nuc = torch.sigmoid(kinetics_raw[:, [5]])

        theta_sioh_drive = k_ads * sio_eff * (1.0 + 0.15 * tau_res + 0.08 * phi_p)
        theta_sioh = torch.sigmoid(theta_sioh_drive + 0.35 * k_oxid * o_eff - 0.12 * o2_ion_eff)

        theta_network_drive = k_condense * theta_sioh * (
            1.0 + 0.30 * o_eff + 0.12 * thermal_budget
        )
        theta_network = torch.sigmoid(theta_network_drive)
        theta_open_target = torch.sigmoid(1.0 - 1.15 * theta_network_drive + 0.35 * theta_sioh)
        theta_open = torch.sigmoid(
            torch.logit(theta_open_target.clamp(0.05, 0.95)) + 0.08 * tau_res - 0.12 * o2_ion_eff
        )
        densification = torch.sigmoid(
            0.9 * theta_network_drive + 0.15 * o2_ion_eff + 0.08 * thermal_budget
        )

        legacy_surface_flux = torch.sqrt(sio_eff * (o_eff + eps)) * theta_open
        legacy_gross_rate = legacy_surface_flux * (1.0 + 0.18 * o2_ion_eff + 0.08 * tau_res)
        legacy_dep_rate = torch.relu(
            (0.25 + 0.75 * phi_nuc) * legacy_gross_rate / (1.0 + k_sat * legacy_gross_rate)
            - 0.035 * k_etch * o2_ion_eff
        )

        profile_raw = 0.16 * torch.tanh(self.profile_head(z))

        aux_core = {
            "latent": z,
            "sio_eff": sio_eff,
            "o_eff": o_eff,
            "o2_ion_eff": o2_ion_eff,
            "tau_res": tau_res,
            "thermal_budget": thermal_budget,
            "phi_p": phi_p,
            "v_z": v_z,
            "d_sio": d_sio,
            "d_o": d_o,
            "d_ion": d_ion,
            "n_e_proxy": n_e_proxy,
            "k_ads": k_ads,
            "k_oxid": k_oxid,
            "k_condense": k_condense,
            "k_etch": k_etch,
            "k_sat": k_sat,
            "phi_nuc": phi_nuc,
            "profile_coeffs": profile_raw,
        }

        main_r = self.r_grid.unsqueeze(0).unsqueeze(-1).expand(x.shape[0], -1, -1)

        # Use time integration only as a bounded correction on the tau=1 rate.
        tau_schedule = ((0.25, 0.20), (0.60, 0.30), (1.00, 0.50))
        main_gas = None
        main_surface = None
        main_rate = None
        main_rate_int = torch.zeros_like(main_r)
        for tau_value, tau_weight in tau_schedule:
            main_tau = torch.full_like(main_r, float(tau_value))
            main_surface_coords = torch.cat([main_r, main_tau], dim=-1)
            main_gas_coords = torch.cat([torch.ones_like(main_r), main_surface_coords], dim=-1)

            gas_tau = self.evaluate_gas_fields(aux_core, main_gas_coords)
            surface_tau = self.evaluate_surface_fields(aux_core, main_surface_coords, gas_tau)
            rate_tau = self.evaluate_surface_rate(aux_core, main_surface_coords, gas_tau, surface_tau)

            main_rate_int = main_rate_int + tau_weight * rate_tau
            if abs(tau_value - 1.00) < 1e-6:
                main_gas = gas_tau
                main_surface = surface_tau
                main_rate = rate_tau

        log_ratio = torch.log((main_rate_int + eps) / (main_rate + eps))
        time_int_strength = 0.10
        main_rate_eff = main_rate * torch.exp(time_int_strength * torch.tanh(log_ratio))

        main_h = main_rate_eff * time_minutes.unsqueeze(1)
        h_profile = torch.relu(main_h).squeeze(-1)
        delta_h = self.thickness_residual_scale * torch.tanh(self.thickness_residual_head(z))
        h_profile = torch.relu(h_profile + delta_h)

        h_mean = torch.mean(h_profile, dim=1, keepdim=True)
        h_max = torch.max(h_profile, dim=1, keepdim=True).values
        h_min = torch.min(h_profile, dim=1, keepdim=True).values
        h_std = torch.std(h_profile, dim=1, keepdim=True, unbiased=False)
        dep_rate = h_mean / (time_minutes + eps)

        uniformity_range = (h_max - h_min) * 100.0 / (2.0 * (h_mean + eps))
        uniformity_1sigma = h_std * 100.0 / (h_mean + eps)

        ri_struct = (
            1.38
            + 0.16 * densification
            + 0.06 * theta_sioh
            - 0.03 * (o2_ion_eff / (1.0 + o2_ion_eff))
        )
        ri_pred = ri_struct + 0.05 * torch.tanh(self.ri_bias_head(z))

        stress_struct = 80.0 * (
            0.55 * o2_ion_eff + 0.30 * theta_network - 0.25 * theta_sioh - 0.12 * tau_res
        )
        stress_pred = stress_struct + 20.0 * torch.tanh(self.stress_bias_head(z))

        out = torch.cat(
            [
                h_mean,
                h_max,
                h_min,
                dep_rate,
                uniformity_range,
                uniformity_1sigma,
                ri_pred,
                stress_pred,
            ],
            dim=1,
        )

        aux = {
            "latent": z,
            "sio_eff": sio_eff,
            "o_eff": o_eff,
            "o2_ion_eff": o2_ion_eff,
            "tau_res": tau_res,
            "thermal_budget": thermal_budget,
            "phi_p": phi_p,
            "v_z": v_z,
            "d_sio": d_sio,
            "d_o": d_o,
            "d_ion": d_ion,
            "n_e_proxy": n_e_proxy,
            "k_ads": k_ads,
            "k_oxid": k_oxid,
            "k_condense": k_condense,
            "k_etch": k_etch,
            "k_sat": k_sat,
            "phi_nuc": phi_nuc,
            "theta_sioh": theta_sioh,
            "theta_network": theta_network,
            "theta_open": theta_open,
            "theta_open_target": theta_open_target,
            "densification": densification,
            "surface_flux": legacy_surface_flux,
            "gross_rate": legacy_gross_rate,
            "legacy_dep_rate": legacy_dep_rate,
            "main_gas": main_gas,
            "main_surface": main_surface,
            "main_rate": main_rate,
            "main_rate_int": main_rate_int,
            "main_rate_eff": main_rate_eff,
            "main_h": main_h,
            "h_profile": h_profile,
            "thickness_residual": delta_h,
            "profile_coeffs": profile_raw,
            "ri_struct": ri_struct,
            "stress_struct": stress_struct,
        }
        return out, aux


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


def add_model_args(parser):
    parser.add_argument("--epochs-pretrain", type=int, default=300)
    parser.add_argument("--epochs-finetune", type=int, default=900)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--disable-derived-features",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool_arg,
        choices=[True, False],
        metavar="{true,false}",
        help="关闭派生特征，可选 true/false；省略值时等价于 true",
    )
    parser.add_argument("--lambda-thickness", type=float, default=3.0, help="Thickness/MAX/MIN 数据项权重")
    parser.add_argument("--lambda-range", type=float, default=0.5, help="MAX-MIN 范围数据项权重")
    parser.add_argument("--lambda-rate", type=float, default=0.2, help="沉积速率数据项权重")
    parser.add_argument("--lambda-uniformity", type=float, default=0.02, help="均匀性数据项权重")
    parser.add_argument("--lambda-ri", type=float, default=0.0, help="折射率数据项权重")
    parser.add_argument("--lambda-stress", type=float, default=0.0, help="应力数据项权重")
    parser.add_argument("--lambda-ht", type=float, default=0.0)
    parser.add_argument("--lambda-maxmin", type=float, default=0.1)
    parser.add_argument("--lambda-uniformity-def", type=float, default=0.0)
    parser.add_argument("--lambda-mono-t", type=float, default=0.0)
    parser.add_argument("--lambda-nonneg-h", type=float, default=0.05)
    parser.add_argument("--lambda-nonneg-r", type=float, default=0.05)
    parser.add_argument("--lambda-species", type=float, default=0.03)
    parser.add_argument("--lambda-surface", type=float, default=0.03)
    parser.add_argument("--lambda-profile", type=float, default=0.03)
    parser.add_argument("--lambda-property", type=float, default=0.0)
    parser.add_argument("--lambda-gas", type=float, default=0.02)
    parser.add_argument("--lambda-flux", type=float, default=0.01)
    parser.add_argument("--lambda-surface-pde", type=float, default=0.02)
    parser.add_argument("--lambda-field-consistency", type=float, default=0.02)
    parser.add_argument("--lambda-h-pde", type=float, default=0.0)
    parser.add_argument("--lambda-rate-species-mono", type=float, default=0.0)
    parser.add_argument("--lambda-axis-bc", type=float, default=0.0)
    parser.add_argument("--lambda-inlet-bc", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--radial-bins", type=int, default=55, help="径向 profile 采样点数")
    parser.add_argument("--radial-modes", type=int, default=4, help="低阶径向模态数")
    parser.add_argument("--gas-collocation", type=int, default=4, help="每个样本的气相 collocation 点数")
    parser.add_argument("--surface-collocation", type=int, default=4, help="每个样本的表面 collocation 点数")
    parser.add_argument("--grad-clip-norm", type=float, default=5.0, help="梯度裁剪范数上限")
    parser.add_argument(
        "--use-physical-space-scale",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool_arg,
        choices=[True, False],
        metavar="{true,false}",
        help="对z方向导数使用Space物理尺度修正，可选 true/false；省略值时等价于 true",
    )
    parser.add_argument(
        "--disable-adaptive-loss",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool_arg,
        choices=[True, False],
        metavar="{true,false}",
        help="关闭 adaptive uncertainty weighting，直接使用固定损失权重，可选 true/false；省略值时等价于 true",
    )
    return parser


def build_arg_parser():
    parser = argparse.ArgumentParser(description="轻量机理 PINN: 有效活性种-表面反应-径向厚度场 8 输出预测")
    parser.add_argument("--data", default="datasets/PE_TEOS.csv", help="CSV 数据路径")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pred-out", default="pinn_predictions.csv", help="预测结果 CSV")
    parser.add_argument("--metrics-out", default="pinn_metrics.json", help="指标 JSON")
    parser.add_argument("--loss-out", default="pinn_loss_history.csv", help="训练过程损失 CSV")
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
    feat_order = prepared["feat_order"]
    target_order = prepared["target_order"]
    derived_feat_order = prepared["derived_feat_order"]
    train_df = prepared["train_df"]
    val_df = prepared["val_df"]
    test_df = prepared["test_df"]

    feature_indices = {name: feat_order.index(col) for name, col in feature_cols.items()}
    derived_indices = {
        name: (feat_order.index(name) if name in feat_order else None) for name in derived_feat_order
    }
    time_idx = feature_indices["Time"]
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

    thickness_residual_scale = 0.04 * float(np.std(y_train_raw[:, idx["Thickness"]]))

    x_scaler = StandardScalerNumpy().fit(x_train_raw)

    x_train = x_scaler.transform(x_train_raw).astype(np.float32)
    x_val = x_scaler.transform(x_val_raw).astype(np.float32)
    x_test = x_scaler.transform(x_test_raw).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReactionInformedMsPINN(
        in_dim=len(feat_order),
        hidden=160,
        radial_bins=args.radial_bins,
        radial_modes=args.radial_modes,
        thickness_residual_scale=thickness_residual_scale,
    ).to(device)

    adaptive_log_vars = nn.ParameterDict(
        {
            "data": nn.Parameter(torch.tensor(0.0, device=device)),
            "ht": nn.Parameter(torch.tensor(0.0, device=device)),
            "maxmin": nn.Parameter(torch.tensor(0.0, device=device)),
            "uniformity_def": nn.Parameter(torch.tensor(0.0, device=device)),
            "mono_t": nn.Parameter(torch.tensor(0.0, device=device)),
            "nonneg_h": nn.Parameter(torch.tensor(0.0, device=device)),
            "nonneg_r": nn.Parameter(torch.tensor(0.0, device=device)),
            "species": nn.Parameter(torch.tensor(0.0, device=device)),
            "surface": nn.Parameter(torch.tensor(0.0, device=device)),
            "profile": nn.Parameter(torch.tensor(0.0, device=device)),
            "property": nn.Parameter(torch.tensor(0.0, device=device)),
            "gas": nn.Parameter(torch.tensor(0.0, device=device)),
            "flux": nn.Parameter(torch.tensor(0.0, device=device)),
            "surface_pde": nn.Parameter(torch.tensor(0.0, device=device)),
            "field_consistency": nn.Parameter(torch.tensor(0.0, device=device)),
            "h_pde": nn.Parameter(torch.tensor(0.0, device=device)),
            "rate_species_mono": nn.Parameter(torch.tensor(0.0, device=device)),
            "axis_bc": nn.Parameter(torch.tensor(0.0, device=device)),
            "inlet_bc": nn.Parameter(torch.tensor(0.0, device=device)),
        }
    )

    trainable_params = list(model.parameters())
    if not args.disable_adaptive_loss:
        trainable_params += list(adaptive_log_vars.parameters())
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    x_train_t = to_tensor(x_train, device)
    y_train_t = to_tensor(y_train_raw, device)
    x_val_t = to_tensor(x_val, device)
    y_val_t = to_tensor(y_val_raw, device)

    train_time_minutes_t = to_tensor(x_train_raw[:, [time_idx]] / 60.0, device)
    val_time_minutes_t = to_tensor(x_val_raw[:, [time_idx]] / 60.0, device)
    test_time_minutes_t = to_tensor(x_test_raw[:, [time_idx]] / 60.0, device)

    pressure_train_raw_t = to_tensor(x_train_raw[:, [feature_indices["Pressure"]]], device)
    o2_train_raw_t = to_tensor(x_train_raw[:, [feature_indices["O2"]]], device)
    teos_train_raw_t = to_tensor(x_train_raw[:, [feature_indices["TEOS"]]], device)
    he_train_raw_t = to_tensor(x_train_raw[:, [feature_indices["He"]]], device)
    time_train_raw_t = to_tensor(x_train_raw[:, [feature_indices["Time"]]], device)
    space_train_raw_t = to_tensor(x_train_raw[:, [feature_indices["Space"]]], device)
    temp_train_raw_t = to_tensor(x_train_raw[:, [feature_indices["Temperature"]]], device)
    power_train_raw_t = to_tensor(x_train_raw[:, [feature_indices["Power"]]], device)

    x_std_t = to_tensor(x_scaler.std_.reshape(1, -1).astype(np.float32), device)

    y_scale_np = y_train_raw.std(axis=0).astype(np.float32)
    y_scale_np[y_scale_np < 1.0] = 1.0
    y_scale_t = to_tensor(y_scale_np.reshape(1, -1), device)

    thickness_scale_t = y_scale_t[:, [idx["Thickness"]]]
    max_scale_t = y_scale_t[:, [idx["MAX"]]]
    min_scale_t = y_scale_t[:, [idx["MIN"]]]
    range_train_np = y_train_raw[:, idx["MAX"]] - y_train_raw[:, idx["MIN"]]
    range_scale_np = np.array([[max(float(np.std(range_train_np)), 1.0)]], dtype=np.float32)
    range_scale_t = to_tensor(range_scale_np, device)
    rate_scale_t = y_scale_t[:, [idx["Deposition_Rate"]]]
    uniformity_range_scale_t = y_scale_t[:, [idx["Uniformity_Range"]]]
    uniformity_sigma_scale_t = y_scale_t[:, [idx["Uniformity_1sigma"]]]
    ri_scale_t = y_scale_t[:, [idx["RI"]]]
    stress_scale_t = y_scale_t[:, [idx["Stress"]]]

    def scaled_mse(pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor):
        return torch.mean(((pred - target) / scale) ** 2)

    def relative_mse(pred: torch.Tensor, target: torch.Tensor, floor: float = 1.0):
        scale = torch.clamp(torch.abs(target.detach()), min=floor)
        return torch.mean(((pred - target) / scale) ** 2)

    def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def add_chain_term(acc: torch.Tensor, grads: torch.Tensor, feat_name: str, derivative: torch.Tensor):
        feat_idx = derived_indices.get(feat_name)
        if feat_idx is None:
            return acc
        return acc + grads[:, [feat_idx]] * derivative / x_std_t[:, [feat_idx]]

    def total_raw_derivative(grads: torch.Tensor, raw_name: str):
        base_idx = feature_indices[raw_name]
        deriv = grads[:, [base_idx]] / x_std_t[:, [base_idx]]

        total_flow = torch.clamp(o2_train_raw_t + teos_train_raw_t + he_train_raw_t, min=1e-6)
        teos_safe = torch.clamp(teos_train_raw_t, min=1e-6)
        teos_sq = torch.clamp(teos_train_raw_t**2, min=1e-6)
        pressure_safe = torch.clamp(pressure_train_raw_t, min=1e-6)

        if raw_name == "Time":
            deriv = add_chain_term(deriv, grads, "drv_Power_Time", power_train_raw_t)
            deriv = add_chain_term(deriv, grads, "drv_Temp_Time", temp_train_raw_t)
        elif raw_name == "TEOS":
            deriv = add_chain_term(deriv, grads, "drv_Total_Flow", torch.ones_like(teos_train_raw_t))
            deriv = add_chain_term(deriv, grads, "drv_O2_TEOS_Ratio", -o2_train_raw_t / teos_sq)
            deriv = add_chain_term(deriv, grads, "drv_HE_TEOS_Ratio", -he_train_raw_t / teos_sq)
            deriv = add_chain_term(
                deriv,
                grads,
                "drv_TEOS_Partial_Fraction",
                (o2_train_raw_t + he_train_raw_t) / (total_flow**2),
            )
            deriv = add_chain_term(
                deriv,
                grads,
                "drv_O2_Partial_Fraction",
                -o2_train_raw_t / (total_flow**2),
            )
        elif raw_name == "O2":
            deriv = add_chain_term(deriv, grads, "drv_Total_Flow", torch.ones_like(o2_train_raw_t))
            deriv = add_chain_term(deriv, grads, "drv_O2_TEOS_Ratio", 1.0 / teos_safe)
            deriv = add_chain_term(
                deriv,
                grads,
                "drv_TEOS_Partial_Fraction",
                -teos_train_raw_t / (total_flow**2),
            )
            deriv = add_chain_term(
                deriv,
                grads,
                "drv_O2_Partial_Fraction",
                (teos_train_raw_t + he_train_raw_t) / (total_flow**2),
            )
        elif raw_name == "Power":
            deriv = add_chain_term(deriv, grads, "drv_Power_Time", time_train_raw_t)
            deriv = add_chain_term(deriv, grads, "drv_Power_Pressure_Ratio", 1.0 / pressure_safe)

        return deriv

    def weighted_term(name: str, raw_loss: torch.Tensor):
        if args.disable_adaptive_loss:
            return raw_loss
        log_var = adaptive_log_vars[name]
        stabilized_loss = torch.clamp(raw_loss, min=1e-8)
        return torch.exp(-log_var) * stabilized_loss + log_var

    gas_collocation_points = (
        (0.25, 0.25, 0.5),
        (0.25, 0.75, 0.5),
        (0.75, 0.25, 0.5),
        (0.75, 0.75, 0.5),
    )
    surface_collocation_points = (
        (0.25, 0.5),
        (0.75, 0.5),
        (0.25, 1.0),
        (0.75, 1.0),
    )

    def sample_fixed_coords(num_samples: int, base_points, num_points: int):
        coords = torch.tensor(base_points, device=device, dtype=torch.float32)
        if num_points > coords.shape[0]:
            repeats = (num_points + coords.shape[0] - 1) // coords.shape[0]
            coords = coords.repeat(repeats, 1)
        coords = coords[:num_points]
        coords = coords.unsqueeze(0).expand(num_samples, -1, -1).clone()
        return coords.requires_grad_(True)

    def build_surface_gas_coords(surface_coords: torch.Tensor):
        z_coord = torch.ones(
            surface_coords.shape[0],
            surface_coords.shape[1],
            1,
            device=surface_coords.device,
            dtype=surface_coords.dtype,
        )
        return torch.cat([z_coord, surface_coords], dim=-1)

    def expand_sample_cond(tensor: torch.Tensor, num_points: int):
        return tensor.unsqueeze(1).expand(-1, num_points, -1)

    def has_nonfinite_gradients(params) -> bool:
        for param in params:
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return True
        return False

    total_epochs = args.epochs_pretrain + args.epochs_finetune
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    best_val_score = float("inf")
    best_epoch = 0
    bad_epochs = 0
    nonfinite_stop_reason = None
    nonfinite_stop_epoch = None

    t0 = time.time()
    epoch_iter = tqdm(
        range(1, total_epochs + 1),
        desc=f"PINN20260511 seed={args.seed}",
        leave=False,
    )
    for epoch in epoch_iter:
        model.train()
        optimizer.zero_grad()

        physics_stage = epoch > args.epochs_pretrain
        if physics_stage:
            x_in = x_train_t.detach().clone().requires_grad_(True)
        else:
            x_in = x_train_t
        pred_raw, aux = model(x_in, train_time_minutes_t)
        h_pred_raw = pred_raw[:, [idx["Thickness"]]]
        max_pred_raw = pred_raw[:, [idx["MAX"]]]
        min_pred_raw = pred_raw[:, [idx["MIN"]]]
        r_pred_raw = pred_raw[:, [idx["Deposition_Rate"]]]
        u_range_pred_raw = pred_raw[:, [idx["Uniformity_Range"]]]
        u_sigma_pred_raw = pred_raw[:, [idx["Uniformity_1sigma"]]]
        ri_pred_raw = pred_raw[:, [idx["RI"]]]
        stress_pred_raw = pred_raw[:, [idx["Stress"]]]

        l_data_thk = (
            scaled_mse(h_pred_raw, y_train_t[:, [idx["Thickness"]]], thickness_scale_t)
            + scaled_mse(max_pred_raw, y_train_t[:, [idx["MAX"]]], max_scale_t)
            + scaled_mse(min_pred_raw, y_train_t[:, [idx["MIN"]]], min_scale_t)
        )
        range_pred_raw = max_pred_raw - min_pred_raw
        range_true_raw = y_train_t[:, [idx["MAX"]]] - y_train_t[:, [idx["MIN"]]]
        l_data_range = scaled_mse(range_pred_raw, range_true_raw, range_scale_t)
        l_data_rate = scaled_mse(r_pred_raw, y_train_t[:, [idx["Deposition_Rate"]]], rate_scale_t)
        l_data_uni = (
            scaled_mse(
                u_range_pred_raw,
                y_train_t[:, [idx["Uniformity_Range"]]],
                uniformity_range_scale_t,
            )
            + scaled_mse(
                u_sigma_pred_raw,
                y_train_t[:, [idx["Uniformity_1sigma"]]],
                uniformity_sigma_scale_t,
            )
        )
        l_data_ri = scaled_mse(ri_pred_raw, y_train_t[:, [idx["RI"]]], ri_scale_t)
        l_data_stress = scaled_mse(stress_pred_raw, y_train_t[:, [idx["Stress"]]], stress_scale_t)
        l_data = (
            args.lambda_thickness * l_data_thk
            + args.lambda_range * l_data_range
            + args.lambda_rate * l_data_rate
            + args.lambda_uniformity * l_data_uni
            + args.lambda_ri * l_data_ri
            + args.lambda_stress * l_data_stress
        )

        if not physics_stage:
            zero = torch.zeros((), device=device)
            l_ht = zero
            l_uniformity_def = zero
            l_maxmin = zero
            l_mono_t = zero
            l_nonneg_h = zero
            l_nonneg_r = zero
            l_species = zero
            l_surface = zero
            l_profile = zero
            l_property = zero
            l_gas = zero
            l_flux = zero
            l_surface_pde = zero
            l_field_consistency = zero
            l_h_pde = zero
            l_rate_species_mono = zero
            l_axis_bc = zero
            l_inlet_bc = zero
            total_loss = weighted_term("data", l_data)
            stage = "pretrain"
        else:
            zero = torch.zeros((), device=device)
            l_ht = zero

            eps = 1e-6
            u_range_def = (max_pred_raw - min_pred_raw) * 100.0 / (2.0 * (h_pred_raw + eps))
            u_sigma_def = (max_pred_raw - min_pred_raw) * 100.0 / (6.0 * (h_pred_raw + eps))
            l_uniformity_def = torch.mean(
                ((u_range_pred_raw - u_range_def) / uniformity_range_scale_t) ** 2
            ) + torch.mean(
                ((u_sigma_pred_raw - u_sigma_def) / uniformity_sigma_scale_t) ** 2
            )

            l_maxmin = (
                torch.mean((torch.relu(h_pred_raw - max_pred_raw) / max_scale_t) ** 2)
                + torch.mean((torch.relu(min_pred_raw - h_pred_raw) / min_scale_t) ** 2)
                + torch.mean((torch.relu(min_pred_raw - max_pred_raw) / max_scale_t) ** 2)
            )

            dh_grad_all = torch.autograd.grad(
                h_pred_raw,
                x_in,
                grad_outputs=torch.ones_like(h_pred_raw),
                create_graph=True,
                retain_graph=True,
            )[0]
            dh_dt = total_raw_derivative(dh_grad_all, "Time")

            l_mono_t = torch.mean(torch.relu(-dh_dt) ** 2)
            l_nonneg_h = (
                torch.mean((torch.relu(-h_pred_raw) / thickness_scale_t) ** 2)
                + torch.mean((torch.relu(-max_pred_raw) / max_scale_t) ** 2)
                + torch.mean((torch.relu(-min_pred_raw) / min_scale_t) ** 2)
            )
            l_nonneg_r = (
                torch.mean((torch.relu(-r_pred_raw) / rate_scale_t) ** 2)
                + torch.mean((torch.relu(-u_range_pred_raw) / uniformity_range_scale_t) ** 2)
                + torch.mean((torch.relu(-u_sigma_pred_raw) / uniformity_sigma_scale_t) ** 2)
            )

            sio_grad_all = torch.autograd.grad(
                aux["sio_eff"],
                x_in,
                grad_outputs=torch.ones_like(aux["sio_eff"]),
                create_graph=True,
                retain_graph=True,
            )[0]
            o_grad_all = torch.autograd.grad(
                aux["o_eff"],
                x_in,
                grad_outputs=torch.ones_like(aux["o_eff"]),
                create_graph=True,
                retain_graph=True,
            )[0]
            ion_grad_all = torch.autograd.grad(
                aux["o2_ion_eff"],
                x_in,
                grad_outputs=torch.ones_like(aux["o2_ion_eff"]),
                create_graph=True,
                retain_graph=True,
            )[0]

            d_sio_d_teos = total_raw_derivative(sio_grad_all, "TEOS")
            d_o_d_o2 = total_raw_derivative(o_grad_all, "O2")
            d_ion_d_power = total_raw_derivative(ion_grad_all, "Power")

            l_species = (
                torch.mean(torch.relu(-d_sio_d_teos) ** 2)
                + torch.mean(torch.relu(-d_o_d_o2) ** 2)
                + torch.mean(torch.relu(-d_ion_d_power) ** 2)
            )

            l_surface = torch.mean((aux["theta_open"] - aux["theta_open_target"]) ** 2)

            h_profile = aux["h_profile"] / thickness_scale_t
            profile_d1 = h_profile[:, 1:] - h_profile[:, :-1]
            profile_d2 = profile_d1[:, 1:] - profile_d1[:, :-1]
            l_profile = torch.mean(profile_d2 ** 2)

            l_property = scaled_mse(ri_pred_raw, aux["ri_struct"], ri_scale_t) + scaled_mse(
                stress_pred_raw,
                aux["stress_struct"],
                stress_scale_t,
            )

            gas_coords = sample_fixed_coords(
                x_train_t.shape[0],
                gas_collocation_points,
                args.gas_collocation,
            )
            gas_fields = model.evaluate_gas_fields(aux, gas_coords)

            gas_surface_coords = gas_coords[:, :, 1:]
            gas_surface_gas_coords = build_surface_gas_coords(gas_surface_coords)
            gas_fields_at_surface = model.evaluate_gas_fields(aux, gas_surface_gas_coords)
            gas_surface_state = model.evaluate_surface_fields(aux, gas_surface_coords, gas_fields_at_surface)

            c_sio = gas_fields["c_sio"]
            c_o = gas_fields["c_o"]
            c_ion = gas_fields["c_ion"]

            grad_c_sio = torch.autograd.grad(
                c_sio,
                gas_coords,
                grad_outputs=torch.ones_like(c_sio),
                create_graph=True,
                retain_graph=True,
            )[0]
            grad_c_o = torch.autograd.grad(
                c_o,
                gas_coords,
                grad_outputs=torch.ones_like(c_o),
                create_graph=True,
                retain_graph=True,
            )[0]
            grad_c_ion = torch.autograd.grad(
                c_ion,
                gas_coords,
                grad_outputs=torch.ones_like(c_ion),
                create_graph=True,
                retain_graph=True,
            )[0]

            d2_c_sio_z = torch.autograd.grad(
                grad_c_sio[..., [0]],
                gas_coords,
                grad_outputs=torch.ones_like(grad_c_sio[..., [0]]),
                create_graph=True,
                retain_graph=True,
            )[0][..., [0]]
            d2_c_sio_r = torch.autograd.grad(
                grad_c_sio[..., [1]],
                gas_coords,
                grad_outputs=torch.ones_like(grad_c_sio[..., [1]]),
                create_graph=True,
                retain_graph=True,
            )[0][..., [1]]
            d2_c_o_z = torch.autograd.grad(
                grad_c_o[..., [0]],
                gas_coords,
                grad_outputs=torch.ones_like(grad_c_o[..., [0]]),
                create_graph=True,
                retain_graph=True,
            )[0][..., [0]]
            d2_c_o_r = torch.autograd.grad(
                grad_c_o[..., [1]],
                gas_coords,
                grad_outputs=torch.ones_like(grad_c_o[..., [1]]),
                create_graph=True,
                retain_graph=True,
            )[0][..., [1]]
            d2_c_ion_z = torch.autograd.grad(
                grad_c_ion[..., [0]],
                gas_coords,
                grad_outputs=torch.ones_like(grad_c_ion[..., [0]]),
                create_graph=True,
                retain_graph=True,
            )[0][..., [0]]
            d2_c_ion_r = torch.autograd.grad(
                grad_c_ion[..., [1]],
                gas_coords,
                grad_outputs=torch.ones_like(grad_c_ion[..., [1]]),
                create_graph=True,
                retain_graph=True,
            )[0][..., [1]]

            gas_points = gas_coords.shape[1]
            teos_drive = torch.log1p(expand_sample_cond(teos_train_raw_t, gas_points))
            o2_drive = torch.log1p(expand_sample_cond(o2_train_raw_t, gas_points))
            phi_p_g = expand_sample_cond(aux["phi_p"], gas_points)
            v_z_g = expand_sample_cond(aux["v_z"], gas_points)
            d_sio_g = expand_sample_cond(aux["d_sio"], gas_points)
            d_o_g = expand_sample_cond(aux["d_o"], gas_points)
            d_ion_g = expand_sample_cond(aux["d_ion"], gas_points)
            n_e_g = expand_sample_cond(aux["n_e_proxy"], gas_points)
            k_ads_g = expand_sample_cond(aux["k_ads"], gas_points)
            k_oxid_g = expand_sample_cond(aux["k_oxid"], gas_points)
            k_condense_g = expand_sample_cond(aux["k_condense"], gas_points)
            k_etch_g = expand_sample_cond(aux["k_etch"], gas_points)
            k_sat_g = expand_sample_cond(aux["k_sat"], gas_points)
            t_g = torch.clamp(expand_sample_cond(train_time_minutes_t, gas_points), min=1e-3)

            if args.use_physical_space_scale:
                space_g = torch.clamp(expand_sample_cond(space_train_raw_t, gas_points), min=1e-3)
                dc_sio_dz = grad_c_sio[..., [0]] / space_g
                dc_o_dz = grad_c_o[..., [0]] / space_g
                dc_ion_dz = grad_c_ion[..., [0]] / space_g
                d2_c_sio_z_eff = d2_c_sio_z / (space_g**2)
                d2_c_o_z_eff = d2_c_o_z / (space_g**2)
                d2_c_ion_z_eff = d2_c_ion_z / (space_g**2)
            else:
                dc_sio_dz = grad_c_sio[..., [0]]
                dc_o_dz = grad_c_o[..., [0]]
                dc_ion_dz = grad_c_ion[..., [0]]
                d2_c_sio_z_eff = d2_c_sio_z
                d2_c_o_z_eff = d2_c_o_z
                d2_c_ion_z_eff = d2_c_ion_z

            k_sio_gen = 0.12 + 0.18 * torch.sigmoid(k_ads_g)
            k_sio_loss = 0.04 + 0.10 * torch.sigmoid(k_etch_g)
            k_sio_surf = 0.10 + 0.12 * torch.sigmoid(k_condense_g)
            g_sio = k_sio_gen * teos_drive * phi_p_g * c_o / (0.5 + k_sat_g + c_o)
            l_sio = k_sio_loss * c_sio + k_sio_surf * c_sio * gas_surface_state["theta"]

            k_o_diss = 0.14 + 0.18 * torch.sigmoid(k_oxid_g)
            k_o_rec = 0.03 + 0.06 * torch.sigmoid(k_etch_g)
            k_o_teos = 0.08 + 0.10 * torch.sigmoid(k_ads_g)
            k_o_surf = 0.10 + 0.08 * torch.sigmoid(k_condense_g)
            g_o = 2.0 * k_o_diss * o2_drive * phi_p_g
            l_o = (
                k_o_rec * c_o**2
                + k_o_teos * c_o * teos_drive
                + k_o_surf * c_o * c_sio * gas_surface_state["theta"]
            )

            k_ion_gen = 0.10 + 0.16 * torch.sigmoid(k_oxid_g)
            k_ion_rec = 0.03 + 0.07 * torch.sigmoid(k_etch_g)
            k_ion_wall = 0.04 + 0.06 * torch.sigmoid(k_sat_g)
            k_ion_surf = 0.08 + 0.08 * torch.sigmoid(k_ads_g)
            g_ion = k_ion_gen * o2_drive * phi_p_g
            l_ion = (
                k_ion_rec * c_ion * n_e_g
                + k_ion_wall * c_ion
                + k_ion_surf * c_ion * (1.0 - gas_surface_state["theta"])
            )

            gas_res_sio = (
                grad_c_sio[..., [2]] / t_g
                + v_z_g * dc_sio_dz
                - d_sio_g * (d2_c_sio_z_eff + d2_c_sio_r)
                - g_sio
                + l_sio
            )
            gas_res_o = (
                grad_c_o[..., [2]] / t_g
                + v_z_g * dc_o_dz
                - d_o_g * (d2_c_o_z_eff + d2_c_o_r)
                - g_o
                + l_o
            )
            gas_res_ion = (
                grad_c_ion[..., [2]] / t_g
                + v_z_g * dc_ion_dz
                - d_ion_g * (d2_c_ion_z_eff + d2_c_ion_r)
                - g_ion
                + l_ion
            )
            l_gas = torch.mean(gas_res_sio**2) + torch.mean(gas_res_o**2) + torch.mean(gas_res_ion**2)

            surface_coords = sample_fixed_coords(
                x_train_t.shape[0],
                surface_collocation_points,
                args.surface_collocation,
            )
            flux_coords = build_surface_gas_coords(surface_coords)
            gas_surface = model.evaluate_gas_fields(aux, flux_coords)
            surface_fields = model.evaluate_surface_fields(aux, surface_coords, gas_surface)
            surface_rate = model.evaluate_surface_rate(aux, surface_coords, gas_surface, surface_fields)
            h_field = model.evaluate_h_field(aux, surface_coords, gas_surface, surface_fields, train_time_minutes_t)

            grad_flux_sio = torch.autograd.grad(
                gas_surface["c_sio"],
                flux_coords,
                grad_outputs=torch.ones_like(gas_surface["c_sio"]),
                create_graph=True,
                retain_graph=True,
            )[0]
            grad_flux_o = torch.autograd.grad(
                gas_surface["c_o"],
                flux_coords,
                grad_outputs=torch.ones_like(gas_surface["c_o"]),
                create_graph=True,
                retain_graph=True,
            )[0]

            surface_points = surface_coords.shape[1]
            d_sio_s = expand_sample_cond(aux["d_sio"], surface_points)
            d_o_s = expand_sample_cond(aux["d_o"], surface_points)
            k_sio_surf_s = 0.10 + 0.12 * torch.sigmoid(expand_sample_cond(aux["k_condense"], surface_points))
            k_o_surf_s = 0.10 + 0.08 * torch.sigmoid(expand_sample_cond(aux["k_condense"], surface_points))
            t_s = torch.clamp(expand_sample_cond(train_time_minutes_t, surface_points), min=1e-3)

            if args.use_physical_space_scale:
                space_s = torch.clamp(expand_sample_cond(space_train_raw_t, surface_points), min=1e-3)
                flux_sio = -d_sio_s * grad_flux_sio[..., [0]] / space_s
                flux_o = -d_o_s * grad_flux_o[..., [0]] / space_s
            else:
                flux_sio = -d_sio_s * grad_flux_sio[..., [0]]
                flux_o = -d_o_s * grad_flux_o[..., [0]]

            sink_sio = k_sio_surf_s * gas_surface["c_sio"] * surface_fields["theta"]
            sink_o = k_o_surf_s * gas_surface["c_o"] * gas_surface["c_sio"] * surface_fields["theta"]
            l_flux = torch.mean((flux_sio - sink_sio) ** 2) + torch.mean((flux_o - sink_o) ** 2)

            h_grad = torch.autograd.grad(
                h_field,
                surface_coords,
                grad_outputs=torch.ones_like(h_field),
                create_graph=True,
                retain_graph=True,
            )[0]
            dh_dt_field = h_grad[..., [1]] / t_s
            l_h_pde = torch.mean(((dh_dt_field - surface_rate) / rate_scale_t) ** 2)

            theta_grad = torch.autograd.grad(
                surface_fields["theta"],
                surface_coords,
                grad_outputs=torch.ones_like(surface_fields["theta"]),
                create_graph=True,
                retain_graph=True,
            )[0]
            eta_grad = torch.autograd.grad(
                surface_fields["eta"],
                surface_coords,
                grad_outputs=torch.ones_like(surface_fields["eta"]),
                create_graph=True,
                retain_graph=True,
            )[0]

            k_theta_o = 0.14 + 0.14 * torch.sigmoid(expand_sample_cond(aux["k_oxid"], surface_points))
            k_theta_ion = 0.10 + 0.10 * torch.sigmoid(expand_sample_cond(aux["k_ads"], surface_points))
            k_theta_sio = 0.10 + 0.10 * torch.sigmoid(expand_sample_cond(aux["k_condense"], surface_points))
            k_theta_cond = 0.05 + 0.08 * torch.sigmoid(expand_sample_cond(aux["k_condense"], surface_points))
            k_eta_sio = 0.12 + 0.10 * torch.sigmoid(expand_sample_cond(aux["k_ads"], surface_points))
            k_eta_cond = 0.06 + 0.08 * torch.sigmoid(expand_sample_cond(aux["k_condense"], surface_points))
            k_eta_relax = 0.04 + 0.06 * torch.sigmoid(expand_sample_cond(aux["k_etch"], surface_points))

            theta_rhs = (
                k_theta_o * gas_surface["c_o"] * (1.0 - surface_fields["theta"])
                + k_theta_ion * gas_surface["c_ion"] * (1.0 - surface_fields["theta"])
                - k_theta_sio * gas_surface["c_sio"] * surface_fields["theta"]
                - 2.0 * k_theta_cond * surface_fields["theta"] ** 2
            )
            eta_rhs = (
                k_eta_sio * gas_surface["c_sio"] * surface_fields["theta"]
                + k_eta_cond * surface_fields["theta"] ** 2
                - k_eta_relax * surface_fields["eta"]
            )
            surf_res_theta = theta_grad[..., [1]] / t_s - theta_rhs
            surf_res_eta = eta_grad[..., [1]] / t_s - eta_rhs
            l_surface_pde = torch.mean(surf_res_theta**2) + torch.mean(surf_res_eta**2)

            gas_mono = {
                "c_sio": gas_surface["c_sio"].detach().clone().requires_grad_(True),
                "c_o": gas_surface["c_o"].detach().clone().requires_grad_(True),
                "c_ion": gas_surface["c_ion"].detach().clone().requires_grad_(True),
            }
            surface_mono = model.evaluate_surface_fields(aux, surface_coords, gas_mono)
            rate_mono = model.evaluate_surface_rate(aux, surface_coords, gas_mono, surface_mono)
            d_rate_d_sio = torch.autograd.grad(
                rate_mono,
                gas_mono["c_sio"],
                grad_outputs=torch.ones_like(rate_mono),
                create_graph=True,
                retain_graph=True,
            )[0]
            d_rate_d_o = torch.autograd.grad(
                rate_mono,
                gas_mono["c_o"],
                grad_outputs=torch.ones_like(rate_mono),
                create_graph=True,
                retain_graph=True,
            )[0]
            d_rate_d_ion = torch.autograd.grad(
                rate_mono,
                gas_mono["c_ion"],
                grad_outputs=torch.ones_like(rate_mono),
                create_graph=True,
                retain_graph=True,
            )[0]
            l_rate_species_mono = (
                torch.mean(torch.relu(-d_rate_d_sio) ** 2)
                + torch.mean(torch.relu(-d_rate_d_o) ** 2)
                + 0.3 * torch.mean(torch.relu(-d_rate_d_ion) ** 2)
            )

            axis_coords = torch.rand(
                (x_train_t.shape[0], args.gas_collocation, 3),
                device=device,
                dtype=torch.float32,
            )
            axis_coords[..., 1] = 0.0
            axis_coords.requires_grad_(True)
            axis_gas = model.evaluate_gas_fields(aux, axis_coords)
            l_axis_bc = torch.zeros((), device=device)
            for gas_key in ["c_sio", "c_o", "c_ion"]:
                grad_axis = torch.autograd.grad(
                    axis_gas[gas_key],
                    axis_coords,
                    grad_outputs=torch.ones_like(axis_gas[gas_key]),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                l_axis_bc = l_axis_bc + torch.mean(grad_axis[..., [1]] ** 2)

            inlet_coords = torch.rand(
                (x_train_t.shape[0], args.gas_collocation, 3),
                device=device,
                dtype=torch.float32,
            )
            inlet_coords[..., 0] = 0.0
            inlet_coords.requires_grad_(True)
            inlet_gas = model.evaluate_gas_fields(aux, inlet_coords)
            inlet_points = inlet_coords.shape[1]
            teos_inlet = torch.log1p(expand_sample_cond(teos_train_raw_t, inlet_points))
            o2_inlet = torch.log1p(expand_sample_cond(o2_train_raw_t, inlet_points))
            phi_p_inlet = expand_sample_cond(aux["phi_p"], inlet_points)
            target_sio_inlet = teos_inlet * phi_p_inlet / (1.0 + teos_inlet)
            target_o_inlet = o2_inlet * phi_p_inlet / (1.0 + o2_inlet)
            target_ion_inlet = o2_inlet * phi_p_inlet / (1.0 + o2_inlet)
            l_inlet_bc = (
                relative_mse(inlet_gas["c_sio"], target_sio_inlet, floor=0.5)
                + relative_mse(inlet_gas["c_o"], target_o_inlet, floor=0.5)
                + relative_mse(inlet_gas["c_ion"], target_ion_inlet, floor=0.5)
            )

            main_surface_gas = aux["main_gas"]
            main_surface_state = aux["main_surface"]
            l_field_consistency = (
                relative_mse(torch.mean(main_surface_gas["c_sio"], dim=1), aux["sio_eff"], floor=0.5)
                + relative_mse(torch.mean(main_surface_gas["c_o"], dim=1), aux["o_eff"], floor=0.5)
                + relative_mse(torch.mean(main_surface_gas["c_ion"], dim=1), aux["o2_ion_eff"], floor=0.5)
                + relative_mse(torch.mean(main_surface_state["theta"], dim=1), aux["theta_sioh"], floor=0.2)
                + relative_mse(torch.mean(main_surface_state["eta"], dim=1), aux["theta_network"], floor=0.2)
            )

            total_loss = (
                weighted_term("data", l_data)
                + args.lambda_ht * weighted_term("ht", l_ht)
                + args.lambda_maxmin * weighted_term("maxmin", l_maxmin)
                + args.lambda_uniformity_def * weighted_term("uniformity_def", l_uniformity_def)
                + args.lambda_mono_t * weighted_term("mono_t", l_mono_t)
                + args.lambda_nonneg_h * weighted_term("nonneg_h", l_nonneg_h)
                + args.lambda_nonneg_r * weighted_term("nonneg_r", l_nonneg_r)
                + args.lambda_species * weighted_term("species", l_species)
                + args.lambda_surface * weighted_term("surface", l_surface)
                + args.lambda_profile * weighted_term("profile", l_profile)
                + args.lambda_property * weighted_term("property", l_property)
                + args.lambda_gas * weighted_term("gas", l_gas)
                + args.lambda_flux * weighted_term("flux", l_flux)
                + args.lambda_surface_pde * weighted_term("surface_pde", l_surface_pde)
                + args.lambda_field_consistency * weighted_term("field_consistency", l_field_consistency)
                + args.lambda_h_pde * weighted_term("h_pde", l_h_pde)
                + args.lambda_rate_species_mono * weighted_term("rate_species_mono", l_rate_species_mono)
                + args.lambda_axis_bc * weighted_term("axis_bc", l_axis_bc)
                + args.lambda_inlet_bc * weighted_term("inlet_bc", l_inlet_bc)
            )
            stage = "physics"

        if not torch.isfinite(total_loss):
            nonfinite_stop_reason = "total_loss"
            nonfinite_stop_epoch = epoch
            print(f"警告: epoch {epoch} total_loss 出现非有限值，回退到最佳权重并提前停止")
            model.load_state_dict(best_state)
            break

        total_loss.backward()

        if has_nonfinite_gradients(trainable_params):
            nonfinite_stop_reason = "gradients"
            nonfinite_stop_epoch = epoch
            print(f"警告: epoch {epoch} 梯度出现非有限值，回退到最佳权重并提前停止")
            model.load_state_dict(best_state)
            optimizer.zero_grad()
            break

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)

        optimizer.step()

        if not args.disable_adaptive_loss:
            for log_var in adaptive_log_vars.values():
                log_var.data.clamp_(-6.0, 6.0)

        model.eval()
        with torch.no_grad():
            train_pred_raw = model(x_train_t, train_time_minutes_t)[0]
            val_pred_raw = model(x_val_t, val_time_minutes_t)[0]

            train_pred_np = train_pred_raw.cpu().numpy()
            val_pred_np = val_pred_raw.cpu().numpy()

            if (not np.isfinite(train_pred_np).all()) or (not np.isfinite(val_pred_np).all()):
                nonfinite_stop_reason = "predictions"
                nonfinite_stop_epoch = epoch
                print(f"警告: epoch {epoch} 预测出现非有限值，回退到最佳权重并提前停止")
                model.load_state_dict(best_state)
                break

            train_thk_rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_train_raw[:, idx["Thickness"]],
                        train_pred_np[:, idx["Thickness"]],
                    )
                )
            )
            val_thk_rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_val_raw[:, idx["Thickness"]],
                        val_pred_np[:, idx["Thickness"]],
                    )
                )
            )
            val_max_rmse = rmse_np(y_val_raw[:, idx["MAX"]], val_pred_np[:, idx["MAX"]])
            val_min_rmse = rmse_np(y_val_raw[:, idx["MIN"]], val_pred_np[:, idx["MIN"]])
            val_range_true = y_val_raw[:, idx["MAX"]] - y_val_raw[:, idx["MIN"]]
            val_range_pred = val_pred_np[:, idx["MAX"]] - val_pred_np[:, idx["MIN"]]
            val_range_rmse = rmse_np(val_range_true, val_range_pred)
            val_score = (
                val_thk_rmse
                + 0.4 * val_max_rmse
                + 0.4 * val_min_rmse
                + 0.3 * val_range_rmse
            )

        history.append(
            {
                "epoch": epoch,
                "stage": stage,
                "total_loss": float(total_loss.item()),
                "data_thickness_group_loss": float(l_data_thk.item()),
                "data_range_loss": float(l_data_range.item()),
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
                "phys_species_loss": float(l_species.item()),
                "phys_surface_loss": float(l_surface.item()),
                "phys_profile_loss": float(l_profile.item()),
                "phys_property_loss": float(l_property.item()),
                "phys_gas_loss": float(l_gas.item()),
                "phys_flux_loss": float(l_flux.item()),
                "phys_surface_pde_loss": float(l_surface_pde.item()),
                "phys_field_consistency_loss": float(l_field_consistency.item()),
                "phys_h_pde_loss": float(l_h_pde.item()),
                "phys_rate_species_mono_loss": float(l_rate_species_mono.item()),
                "phys_axis_bc_loss": float(l_axis_bc.item()),
                "phys_inlet_bc_loss": float(l_inlet_bc.item()),
                "train_thk_rmse": train_thk_rmse,
                "val_thk_rmse": val_thk_rmse,
                "val_score": val_score,
            }
        )
        epoch_iter.set_postfix(
            stage=stage,
            train_rmse=f"{train_thk_rmse:.2f}",
            val_score=f"{val_score:.2f}",
        )

        if val_score < best_val_score:
            best_val_score = val_score
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

    def predict_raw(x_np, t_minutes_np):
        model.eval()
        x_t = to_tensor(x_np, device)
        t_t = to_tensor(t_minutes_np, device)
        with torch.no_grad():
            pred = model(x_t, t_t)[0].cpu().numpy()
        if not np.isfinite(pred).all():
            print("警告: 最终预测仍包含非有限值，已使用数值截断兜底")
            pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=0.0)
        return pred

    train_pred = predict_raw(x_train, x_train_raw[:, [time_idx]] / 60.0)
    val_pred = predict_raw(x_val, x_val_raw[:, [time_idx]] / 60.0)
    test_pred = predict_raw(x_test, x_test_raw[:, [time_idx]] / 60.0)

    if not np.isfinite(best_val_rmse):
        best_val_rmse = rmse_np(y_val_raw[:, idx["Thickness"]], val_pred[:, idx["Thickness"]])
    if not np.isfinite(best_val_score):
        best_val_score = (
            rmse_np(y_val_raw[:, idx["Thickness"]], val_pred[:, idx["Thickness"]])
            + 0.4 * rmse_np(y_val_raw[:, idx["MAX"]], val_pred[:, idx["MAX"]])
            + 0.4 * rmse_np(y_val_raw[:, idx["MIN"]], val_pred[:, idx["MIN"]])
            + 0.3
            * rmse_np(
                y_val_raw[:, idx["MAX"]] - y_val_raw[:, idx["MIN"]],
                val_pred[:, idx["MAX"]] - val_pred[:, idx["MIN"]],
            )
        )

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
            "best_val_score": float(best_val_score),
            "best_epoch": int(best_epoch),
            "trained_epochs": int(len(history)),
            "seconds": float(train_seconds),
            "device": str(device),
            "num_input_features": int(len(feat_order)),
            "num_outputs": int(len(target_order)),
            "use_derived_features": bool(not args.disable_derived_features),
            "stopped_on_nonfinite": bool(nonfinite_stop_reason is not None),
            "nonfinite_stop_reason": nonfinite_stop_reason,
            "nonfinite_stop_epoch": int(nonfinite_stop_epoch) if nonfinite_stop_epoch is not None else None,
            "mechanistic_backbone": "effective_species_surface_radial_bounded_time_integration_correction",
            "loss_weights": {
                "lambda_thickness": args.lambda_thickness,
                "lambda_range": args.lambda_range,
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
                "lambda_species": args.lambda_species,
                "lambda_surface": args.lambda_surface,
                "lambda_profile": args.lambda_profile,
                "lambda_property": args.lambda_property,
                "lambda_gas": args.lambda_gas,
                "lambda_flux": args.lambda_flux,
                "lambda_surface_pde": args.lambda_surface_pde,
                "lambda_field_consistency": args.lambda_field_consistency,
                "lambda_h_pde": args.lambda_h_pde,
                "lambda_rate_species_mono": args.lambda_rate_species_mono,
                "lambda_axis_bc": args.lambda_axis_bc,
                "lambda_inlet_bc": args.lambda_inlet_bc,
            },
            "grad_clip_norm": float(args.grad_clip_norm),
            "use_physical_space_scale": bool(args.use_physical_space_scale),
            "disable_adaptive_loss": bool(args.disable_adaptive_loss),
            "adaptive_log_vars": (
                {}
                if args.disable_adaptive_loss
                else {k: float(v.detach().cpu().item()) for k, v in adaptive_log_vars.items()}
            ),
            "radial_bins": int(args.radial_bins),
            "radial_modes": int(args.radial_modes),
            "gas_collocation": int(args.gas_collocation),
            "surface_collocation": int(args.surface_collocation),
            "thickness_residual_scale": float(thickness_residual_scale),
            "target_loss_scales": {
                metric_key_map[name]: float(y_scale_np[i]) for i, name in enumerate(target_internal_order)
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
