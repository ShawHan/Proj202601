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


class KimLayeredSurfacePINN(nn.Module):
    """Three-stage Kim-inspired PECVD model with layered surface chemistry."""

    def __init__(
        self,
        feature_ref,
        feature_indices,
        ri_init,
        stress_init,
        radial_bins=49,
        radial_modes=4,
        ai_strength=0.20,
        uniformity_direct_blend=0.50,
    ):
        super().__init__()
        self.feature_indices = dict(feature_indices)
        self.radial_modes = int(radial_modes)
        self.ai_strength = float(ai_strength)
        self.uniformity_direct_blend = float(np.clip(uniformity_direct_blend, 0.0, 1.0))

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
            basis.append(r ** (len(basis) + 2))
        radial_basis = []
        for item in basis[: self.radial_modes]:
            centered = item - item.mean()
            radial_basis.append(centered / (torch.max(torch.abs(centered)) + 1e-6))
        self.register_buffer("radial_basis", torch.stack(radial_basis, dim=0))
        self.register_buffer("r_grid", r.view(1, -1))

        self.log_rate_scale = nn.Parameter(torch.tensor(np.log(7800.0), dtype=torch.float32))
        self.raw_x1 = self._positive_param(0.35)
        self.raw_x2 = self._positive_param(0.58)
        self.raw_x3 = self._positive_param(0.95)
        self.raw_x4 = self._positive_param(1.35)
        self.raw_x5 = self._positive_param(1.85)
        self.raw_x_oxygen = self._positive_param(0.48)
        self.raw_x_ion = self._positive_param(0.70)
        self.raw_x_sio_loss = self._positive_param(0.16)

        self.raw_k_p1 = self._positive_param(0.95)
        self.raw_k_p2 = self._positive_param(0.70)
        self.raw_k_p3 = self._positive_param(0.22)
        self.raw_k_p4 = self._positive_param(0.08)
        self.k_p1_temp = nn.Parameter(torch.tensor(-0.12, dtype=torch.float32))
        self.k_p2_temp = nn.Parameter(torch.tensor(0.18, dtype=torch.float32))
        self.k_p3_temp = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.k_p4_temp = nn.Parameter(torch.tensor(0.06, dtype=torch.float32))

        self.raw_k_top_oxid_o = self._positive_param(0.22)
        self.raw_k_top_oxid_ion = self._positive_param(0.16)
        self.raw_k_top_ads_loss = self._positive_param(1.05)
        self.raw_k_top_capture = self._positive_param(0.75)

        self.raw_k_side_oxid_o = self._positive_param(0.34)
        self.raw_k_side_oxid_ion = self._positive_param(0.24)
        self.raw_k_side_cross = self._positive_param(0.28)

        self.raw_k_deep_cross = self._positive_param(0.34)
        self.deep_temp_gain = self._positive_param(0.58)
        self.deep_ion_gain = self._positive_param(0.26)
        self.raw_burial_gain = self._positive_param(0.70)
        self.raw_escape_depth = self._positive_param(0.28)
        self.raw_escape_thickness = self._positive_param(0.22)

        self.space_decay = self._positive_param(0.18)
        self.pressure_exp = nn.Parameter(torch.tensor(0.12, dtype=torch.float32))
        self.time_exp = nn.Parameter(torch.tensor(-0.10, dtype=torch.float32))
        self.power_exp = nn.Parameter(torch.tensor(0.24, dtype=torch.float32))
        self.he_decay = self._positive_param(0.22)
        self.raw_ion_assist = self._positive_param(0.14)
        self.raw_damage_gain = self._positive_param(0.18)
        self.raw_damage_threshold = self._positive_param(0.95)
        self.rate_residual_strength = nn.Parameter(torch.tensor(0.45, dtype=torch.float32))

        self.closure_net = nn.Sequential(
            nn.Linear(10, 24),
            nn.Tanh(),
            nn.Linear(24, 12),
        )
        nn.init.zeros_(self.closure_net[-1].weight)
        nn.init.zeros_(self.closure_net[-1].bias)

        self.rate_residual_net = nn.Sequential(
            nn.Linear(20, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.rate_residual_net[-1].weight)
        nn.init.zeros_(self.rate_residual_net[-1].bias)

        self.profile_linear = nn.Linear(8, self.radial_modes)
        nn.init.zeros_(self.profile_linear.weight)
        nn.init.zeros_(self.profile_linear.bias)

        self.profile_residual_net = nn.Sequential(
            nn.Linear(12, 24),
            nn.Tanh(),
            nn.Linear(24, self.radial_modes),
        )
        nn.init.zeros_(self.profile_residual_net[-1].weight)
        nn.init.zeros_(self.profile_residual_net[-1].bias)

        self.radial_field_net = nn.Sequential(
            nn.Linear(12, 32),
            nn.Tanh(),
            nn.Linear(32, 6),
        )
        nn.init.zeros_(self.radial_field_net[-1].weight)
        nn.init.zeros_(self.radial_field_net[-1].bias)

        self.uniformity_head = nn.Sequential(
            nn.Linear(12, 24),
            nn.Tanh(),
            nn.Linear(24, 2),
        )
        nn.init.zeros_(self.uniformity_head[-1].weight)
        nn.init.zeros_(self.uniformity_head[-1].bias)

        self.property_linear = nn.Linear(8, 2)
        with torch.no_grad():
            self.property_linear.weight.zero_()
            self.property_linear.bias.zero_()
            self.property_linear.weight[0, 0] = 1.05
            self.property_linear.weight[0, 1] = -0.55
            self.property_linear.weight[0, 4] = 0.85
            self.property_linear.weight[0, 6] = 0.18
            self.property_linear.weight[1, 0] = 0.70
            self.property_linear.weight[1, 1] = -0.35
            self.property_linear.weight[1, 2] = 0.60
            self.property_linear.weight[1, 4] = -0.72
            self.property_linear.weight[1, 6] = 0.30
            self.property_linear.bias[1] = -0.08

        self.property_residual_net = nn.Sequential(
            nn.Linear(16, 24),
            nn.Tanh(),
            nn.Linear(24, 2),
        )
        nn.init.zeros_(self.property_residual_net[-1].weight)
        nn.init.zeros_(self.property_residual_net[-1].bias)

        self.ri_base = nn.Parameter(torch.tensor(float(ri_init), dtype=torch.float32))
        self.raw_ri_span = self._positive_param(0.018)
        self.stress_base = nn.Parameter(torch.tensor(float(stress_init), dtype=torch.float32))
        self.raw_stress_span = self._positive_param(220.0)

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

        source_state = torch.cat(
            [
                torch.log1p(teos_n),
                torch.log1p(o2_n),
                torch.log(o2_teos_ratio),
                torch.log1p(teos_feed),
            ],
            dim=1,
        )
        cond_state = torch.cat(
            [
                torch.log(pressure_n),
                torch.log(space_n),
                torch.log(power_n),
                torch.log(time_n),
                temp_center,
                he_fraction,
            ],
            dim=1,
        )
        state = torch.cat([source_state, cond_state], dim=1)

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
            "source_state": source_state,
            "cond_state": cond_state,
            "state": state,
        }

    def _closure(self, state):
        raw = self.closure_net(state)
        bounded = self.ai_strength * torch.tanh(raw)
        return torch.exp(bounded), raw, bounded

    def forward(self, x):
        eps = 1e-6
        phys = self._physics_inputs(x)
        mult, raw_closure, bounded = self._closure(phys["state"])
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
            * torch.exp(0.50 * bounded[:, [2]])
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
            * torch.exp(0.35 * bounded[:, [2]])
        )
        n_oxygen = (
            2.0
            * phys["oxygen_feed"]
            * power_n
            / (power_n + x_o)
            * mult[:, [3]]
        )
        n_ion = (
            phys["oxygen_feed"]
            * power_n
            / (power_n + x_ion)
            * mult[:, [4]]
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

        top_oxidation = (
            self._positive(self.raw_k_top_oxid_o) * n_oxygen
            + self._positive(self.raw_k_top_oxid_ion) * n_ion
        )
        top_oh = top_oxidation / (
            top_oxidation
            + self._positive(self.raw_k_top_ads_loss) * precursor_flux * mult[:, [5]]
            + eps
        )
        top_capture = (
            self._positive(self.raw_k_top_capture)
            * precursor_flux
            * top_oh
            * torch.exp(0.30 * bounded[:, [5]])
        )

        side_denom = torch.clamp(2.0 * precursor_flux, min=eps)
        side_r0 = torch.clamp((2.0 * k1n1 + 2.0 * k2n2 + k3n3) / side_denom, 0.0, 1.0)
        side_oh0 = torch.clamp((k3n3 + 2.0 * k4n4) / side_denom, 0.0, 1.0)
        side_oxid_drive = (
            self._positive(self.raw_k_side_oxid_o) * n_oxygen**2
            + self._positive(self.raw_k_side_oxid_ion) * n_ion
        ) * torch.exp(0.25 * bounded[:, [6]])
        side_oxid = 0.49 * side_oxid_drive / (1.0 + side_oxid_drive)
        side_oh_pre = torch.clamp(side_r0 * side_oxid + side_oh0, 0.0, 1.0)
        side_cross_gain = self._positive(self.raw_k_side_cross) * torch.exp(
            torch.clamp(0.35 * temp_center + 0.10 * torch.log1p(n_ion) + 0.20 * bounded[:, [6]], -2.0, 2.0)
        )
        side_cross_drive = side_cross_gain * side_oh_pre**2
        side_cross = 0.49 * side_cross_drive / (1.0 + side_cross_drive)
        side_oh_after = torch.clamp(side_oh_pre * (1.0 - 2.0 * side_cross), 0.0, 1.0)

        burial_fraction = torch.sigmoid(
            self._positive(self.raw_burial_gain) * torch.log1p(top_capture)
            + 0.15 * torch.log(phys["time_n"])
            - 0.10 * torch.log1p(n_oxygen)
        )
        buried_oh = torch.clamp(side_oh_after * burial_fraction, 0.0, 1.0)

        proto_transport = torch.exp(-self._positive(self.space_decay) * (phys["space_n"] - 1.0))
        proto_transport = proto_transport * torch.exp(
            torch.clamp(self.pressure_exp, -1.0, 1.0) * torch.log(phys["pressure_n"])
        )
        proto_transport = proto_transport * torch.exp(
            -self._positive(self.he_decay) * (phys["he_fraction"] - 0.50)
        )
        h_proto = torch.exp(self.log_rate_scale) * top_capture * proto_transport * phys["time_min"]

        deep_escape = torch.exp(-self._positive(self.raw_escape_depth) * (1.0 + burial_fraction))
        deep_escape = deep_escape * torch.exp(
            -self._positive(self.raw_escape_thickness) * torch.clamp(h_proto / 5000.0, 0.0, 4.0)
        )
        deep_escape = torch.clamp(deep_escape, min=0.05, max=1.0)
        deep_cross_gain = self._positive(self.raw_k_deep_cross) * torch.exp(
            torch.clamp(
                self._positive(self.deep_temp_gain) * temp_center
                + self._positive(self.deep_ion_gain) * torch.log1p(n_ion)
                + 0.20 * bounded[:, [7]],
                -2.0,
                2.0,
            )
        )
        deep_cross_drive = deep_cross_gain * buried_oh**2 * deep_escape
        deep_cross = 0.49 * deep_cross_drive / (1.0 + deep_cross_drive)
        deep_oh = torch.clamp(buried_oh * (1.0 - 2.0 * deep_cross), 0.0, 1.0)

        damage_factor = self._positive(self.raw_damage_gain) * F.softplus(
            torch.log1p(n_ion) - self._positive(self.raw_damage_threshold)
        )
        si_o_survival = torch.exp(-damage_factor) / (1.0 + x_sio * power_n)
        si_o_survival = torch.clamp(si_o_survival, min=0.05, max=1.0)

        transport = proto_transport * torch.exp(0.15 * bounded[:, [8]])
        transient = torch.exp(torch.clamp(self.time_exp, -0.50, 0.50) * torch.log(phys["time_n"]))
        power_gain = torch.exp(torch.clamp(self.power_exp, -0.40, 1.00) * torch.log(power_n))
        ion_assist = self._positive(self.raw_ion_assist) * torch.log1p(n_ion) * deep_escape

        network_crosslink = torch.clamp(
            0.20 * side_oxid + 0.35 * side_cross + 0.45 * deep_cross,
            0.0,
            0.49,
        )
        residual_oh = torch.clamp(
            0.55 * top_oh + 0.30 * side_oh_after + 0.15 * deep_oh,
            0.0,
            1.0,
        )

        rate_state = torch.cat(
            [
                torch.log1p(p0),
                torch.log1p(p1),
                torch.log1p(p2),
                torch.log1p(p3 + p4),
                torch.log1p(n_oxygen),
                torch.log1p(n_ion),
                torch.log1p(top_capture),
                torch.log1p(precursor_flux),
                torch.log(torch.clamp(top_oh, min=eps)),
                torch.log1p(side_cross),
                torch.log1p(deep_cross),
                torch.log(torch.clamp(si_o_survival, min=eps)),
                torch.log(torch.clamp(transport, min=eps)),
                torch.log(phys["residence"]),
                torch.log(phys["time_n"]),
                torch.log(phys["o2_teos_ratio"]),
                torch.log(torch.clamp(deep_escape, min=eps)),
                residual_oh,
                temp_center,
                phys["he_fraction"],
            ],
            dim=1,
        )
        rate_residual = torch.clamp(self.rate_residual_strength, 0.0, 0.80) * torch.tanh(
            self.rate_residual_net(rate_state)
        )

        dep_rate = (
            torch.exp(self.log_rate_scale)
            * top_capture
            * (1.0 + ion_assist)
            * (1.0 + 0.18 * side_cross + 0.12 * deep_cross)
            * si_o_survival
            * transport
            * transient
            * power_gain
            * mult[:, [9]]
            * torch.exp(rate_residual)
        )
        dep_rate = torch.clamp(dep_rate, min=eps)

        incorporation_rate = dep_rate * (1.0 + 1.6 * network_crosslink) * deep_escape
        h_mean = dep_rate * phys["time_min"]

        profile_state = torch.cat(
            [
                torch.log1p(n_oxygen),
                torch.log1p(n_ion),
                torch.log(phys["residence"]),
                torch.log(power_n),
                torch.log(phys["space_n"]),
                temp_center,
                phys["he_fraction"],
                torch.log1p(deep_cross),
            ],
            dim=1,
        )
        profile_state_rich = torch.cat(
            [
                profile_state,
                torch.log1p(top_capture),
                torch.log1p(side_cross),
                torch.log(torch.clamp(deep_escape, min=eps)),
                residual_oh,
            ],
            dim=1,
        )
        profile_coeffs = (
            0.10 * torch.tanh(self.profile_linear(profile_state) + bounded[:, [10]])
            + 0.08 * torch.tanh(self.profile_residual_net(profile_state_rich))
        )
        shape_delta = torch.einsum("bm,mr->br", profile_coeffs, self.radial_basis)
        basis_profile_shape = torch.clamp(1.0 + shape_delta, min=0.30)

        radial_field_state = torch.cat(
            [
                torch.log(phys["pressure_n"]),
                torch.log(phys["space_n"]),
                torch.log(power_n),
                torch.log(phys["time_n"]),
                temp_center,
                phys["he_fraction"],
                torch.log(phys["o2_teos_ratio"]),
                torch.log(phys["residence"]),
                torch.log1p(n_oxygen),
                torch.log1p(n_ion),
                torch.log1p(top_capture),
                torch.log1p(precursor_flux),
            ],
            dim=1,
        )
        radial_params = self.radial_field_net(radial_field_state)
        (
            raw_a_teos,
            raw_a_oxygen,
            raw_a_edge,
            raw_center_bias,
            raw_ring_bias,
            raw_ring_width,
        ) = torch.chunk(radial_params, 6, dim=1)

        a_teos = self._positive(raw_a_teos)
        a_oxygen = self._positive(raw_a_oxygen)
        a_edge = self._positive(raw_a_edge)
        ring_width = self._positive(raw_ring_width)

        r = self.r_grid
        teos_shape = torch.exp(-a_teos * r**2)
        teos_shape = teos_shape / torch.clamp(teos_shape.mean(dim=1, keepdim=True), min=eps)
        oxygen_shape = torch.exp(-a_oxygen * r**2)
        oxygen_shape = oxygen_shape / torch.clamp(oxygen_shape.mean(dim=1, keepdim=True), min=eps)
        edge_loss = torch.exp(-a_edge * r**4)
        center_gain = 1.0 + 0.20 * torch.tanh(raw_center_bias) * (1.0 - r**2)
        ring = torch.exp(-((r - 0.65) ** 2) / (ring_width + 1e-3))
        ring_gain = 1.0 + 0.15 * torch.tanh(raw_ring_bias) * ring

        radial_shape = teos_shape * oxygen_shape * edge_loss * center_gain * ring_gain
        radial_shape = torch.clamp(radial_shape, min=0.20)
        profile_shape = basis_profile_shape * radial_shape
        profile_shape = profile_shape / torch.clamp(profile_shape.mean(dim=1, keepdim=True), min=eps)
        local_rate_std = torch.std(profile_shape, dim=1, keepdim=True, unbiased=False)
        h_profile = h_mean * profile_shape

        h_max = torch.max(h_profile, dim=1, keepdim=True).values
        h_min = torch.min(h_profile, dim=1, keepdim=True).values
        h_std = torch.std(h_profile, dim=1, keepdim=True, unbiased=False)
        uniformity_range_profile = (h_max - h_min) * 100.0 / (2.0 * torch.clamp(h_mean, min=eps))
        uniformity_sigma_profile = h_std * 100.0 / torch.clamp(h_mean, min=eps)
        uniformity_residual = (0.20 * self.uniformity_direct_blend) * torch.tanh(
            self.uniformity_head(profile_state_rich)
        )
        uniformity_range_direct = uniformity_range_profile * torch.exp(uniformity_residual[:, [0]])
        uniformity_sigma_direct = uniformity_sigma_profile * torch.exp(uniformity_residual[:, [1]])
        uniformity_range = uniformity_range_direct
        uniformity_sigma = uniformity_sigma_direct

        densification = torch.sigmoid(
            2.45 * network_crosslink
            + 0.30 * torch.log1p(n_ion)
            + 0.30 * temp_center
            + 0.20 * torch.log(torch.clamp(deep_escape, min=eps))
            - 0.80 * residual_oh
        )
        ion_fraction = n_ion / (1.0 + n_ion)
        oxygen_fraction = n_oxygen / (1.0 + n_oxygen)
        prop_features = torch.cat(
            [
                network_crosslink,
                residual_oh,
                ion_fraction,
                oxygen_fraction,
                densification,
                deep_escape,
                temp_center,
                torch.log1p(incorporation_rate / 10000.0),
            ],
            dim=1,
        )
        property_state = torch.cat(
            [
                prop_features,
                torch.log1p(top_capture),
                torch.log1p(precursor_flux),
                torch.log1p(side_cross),
                torch.log1p(deep_cross),
                torch.log1p(n_oxygen),
                torch.log1p(n_ion),
                torch.clamp(si_o_survival, min=0.0),
                top_oh,
            ],
            dim=1,
        )
        prop_raw = self.property_linear(prop_features) + 0.50 * torch.tanh(
            self.property_residual_net(property_state)
        )
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
            "p0": p0,
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "p4": p4,
            "n_oxygen": n_oxygen,
            "n_ion": n_ion,
            "top_oxidation": top_oxidation,
            "top_oh": top_oh,
            "top_capture": top_capture,
            "precursor_flux": precursor_flux,
            "side_r0": side_r0,
            "side_oh0": side_oh0,
            "side_oxid": side_oxid,
            "side_oh_pre": side_oh_pre,
            "side_cross": side_cross,
            "side_oh_after": side_oh_after,
            "burial_fraction": burial_fraction,
            "buried_oh": buried_oh,
            "deep_escape": deep_escape,
            "deep_cross": deep_cross,
            "deep_oh": deep_oh,
            "damage_factor": damage_factor,
            "si_o_survival": si_o_survival,
            "transport": transport,
            "transient": transient,
            "power_gain": power_gain,
            "ion_assist": ion_assist,
            "network_crosslink": network_crosslink,
            "residual_oh": residual_oh,
            "rate_residual": rate_residual,
            "dep_rate": dep_rate,
            "incorporation_rate": incorporation_rate,
            "densification": densification,
            "profile_coeffs": profile_coeffs,
            "basis_profile_shape": basis_profile_shape,
            "h_profile": h_profile,
            "radial_a_teos": a_teos,
            "radial_a_oxygen": a_oxygen,
            "radial_a_edge": a_edge,
            "radial_center_bias": torch.tanh(raw_center_bias),
            "radial_ring_bias": torch.tanh(raw_ring_bias),
            "radial_ring_width": ring_width,
            "local_rate_std": local_rate_std,
            "uniformity_range_profile": uniformity_range_profile,
            "uniformity_sigma_profile": uniformity_sigma_profile,
            "uniformity_range_direct": uniformity_range_direct,
            "uniformity_sigma_direct": uniformity_sigma_direct,
            "uniformity_residual": uniformity_residual,
            "raw_closure": raw_closure,
            "bounded_closure": bounded,
            "phys": phys,
        }
        return out, aux

    def key_deposition_residuals(self, aux):
        """Minimal Kim-PINN residuals for deposition-dominant physics.

        Only the most deposition-critical states are constrained:
        (1) effective TEOS precursor flux,
        (2) surface OH steady state,
        (3) deposition-rate consistency.

        This avoids over-constraining all latent species while still turning
        the Kim-inspired mechanism into explicit PINN residual losses.
        """
        eps = 1e-6
        phys = aux["phys"]

        power_n = torch.clamp(phys["power_n"], min=eps)
        teos_feed = torch.clamp(phys["teos_feed"], min=eps)
        precursor_flux = torch.clamp(aux["precursor_flux"], min=eps)
        top_oh = torch.clamp(aux["top_oh"], min=eps, max=1.0 - eps)
        n_oxygen = torch.clamp(aux["n_oxygen"], min=eps)
        n_ion = torch.clamp(aux["n_ion"], min=eps)
        dep_rate = torch.clamp(aux["dep_rate"], min=eps)
        transport = torch.clamp(aux["transport"], min=eps)

        def relative_residual(lhs, rhs):
            return (lhs - rhs) / torch.clamp(torch.abs(lhs) + torch.abs(rhs), min=eps)

        x1 = self._positive(self.raw_x1)
        x2 = self._positive(self.raw_x2)

        # 1) Effective TEOS precursor-flux residual.
        # Kim's deposition model reduces the dominant growth contribution to
        # TEOS-derived film precursors in the high O2/TEOS regime. We constrain
        # the total precursor flux instead of all latent p0-p4 states.
        kim_precursor_flux = (
            self._positive(self.raw_k_p1)
            * teos_feed
            * power_n
            / torch.clamp((power_n + x1) * (power_n + x2), min=eps)
        )
        r_precursor = relative_residual(precursor_flux, kim_precursor_flux)
        l_precursor = torch.mean(r_precursor ** 2)

        # 2) Surface-OH steady-state residual.
        # Oxygen species create SiOH sites; TEOS-derived precursors consume them
        # through surface growth. This is the most direct Kim-style surface
        # balance for deposition.
        oxidation_drive = (
            self._positive(self.raw_k_top_oxid_o) * n_oxygen
            + self._positive(self.raw_k_top_oxid_ion) * n_ion
        )
        oh_generation = oxidation_drive * (1.0 - top_oh)
        oh_consumption = self._positive(self.raw_k_top_ads_loss) * precursor_flux * top_oh
        r_surface_oh = relative_residual(oh_generation, oh_consumption)
        l_surface_oh = torch.mean(r_surface_oh ** 2)

        # 3) Deposition-rate residual.
        # The predicted rate should stay consistent with the precursor flux,
        # surface OH coverage, and transport term. Side/deep chemistry remains
        # available as a learned correction but is not separately over-constrained.
        kim_growth_rate = (
            torch.exp(self.log_rate_scale)
            * precursor_flux
            * top_oh
            * transport
        )
        r_growth = relative_residual(dep_rate, kim_growth_rate)
        l_growth = torch.mean(r_growth ** 2)

        return {
            "precursor_residual": l_precursor,
            "surface_oh_residual": l_surface_oh,
            "growth_residual": l_growth,
        }


def add_model_args(parser):
    parser.add_argument("--epochs-pretrain", type=int, default=220)
    parser.add_argument("--epochs-finetune", type=int, default=520)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-6)
    parser.add_argument("--patience", type=int, default=90)
    parser.add_argument("--radial-bins", type=int, default=49)
    parser.add_argument("--radial-modes", type=int, default=4)
    parser.add_argument("--layered-ai-strength", type=float, default=0.20)
    parser.add_argument(
        "--uniformity-direct-blend",
        type=float,
        default=0.50,
        help="Residual strength for uniformity head (used as a small multiplicative correction).",
    )
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
    parser.add_argument("--lambda-rate", type=float, default=0.40)
    parser.add_argument("--lambda-log-rate", type=float, default=0.15)
    parser.add_argument("--lambda-uniformity", type=float, default=0.12)
    parser.add_argument("--lambda-ri", type=float, default=0.12)
    parser.add_argument("--lambda-stress", type=float, default=0.10)
    parser.add_argument("--lambda-closure", type=float, default=0.02)
    parser.add_argument("--lambda-layer-state", type=float, default=0.02)
    parser.add_argument("--lambda-monotonic", type=float, default=0.005)
    parser.add_argument("--lambda-profile-smooth", type=float, default=0.02)
    parser.add_argument("--lambda-rate-residual", type=float, default=0.012)
    parser.add_argument("--lambda-precursor-residual", type=float, default=0.03)
    parser.add_argument("--lambda-surface-oh-residual", type=float, default=0.05)
    parser.add_argument("--lambda-growth-residual", type=float, default=0.04)
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
    parser.add_argument(
        "--calibrator-targets",
        default="Thickness,MAX,MIN,Deposition_Rate,Uniformity_Range,Uniformity_1sigma,RI,Stress",
        help="Comma-separated target names for calibrator; others keep raw PINN outputs.",
    )
    parser.add_argument("--uniformity-refiner", type=pinn_common.parse_bool_arg, default=True)
    parser.add_argument("--uniformity-refiner-blend", type=float, default=0.35)
    parser.add_argument("--uniformity-refiner-n-estimators", type=int, default=500)
    parser.add_argument("--uniformity-refiner-max-depth", type=int, default=2)
    parser.add_argument("--uniformity-refiner-learning-rate", type=float, default=0.025)
    parser.add_argument("--uniformity-refiner-reg-lambda", type=float, default=4.0)
    parser.add_argument("--uniformity-refiner-reg-alpha", type=float, default=0.10)
    parser.add_argument("--rate-thickness-blend", type=float, default=0.0)
    parser.add_argument("--range-max-blend", type=float, default=0.0)
    return parser


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="PINN20260522-KimLayered-KeyResidual: Kim-inspired PECVD PINN with key deposition residuals"
    )
    parser.add_argument("--data", default="datasets/PE_TEOS.csv", help="CSV data path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pred-out", default="pinn20260522_kimlayered_keyres_predictions.csv")
    parser.add_argument("--metrics-out", default="pinn20260522_kimlayered_keyres_metrics.json")
    parser.add_argument("--loss-out", default="pinn20260522_kimlayered_keyres_loss_history.csv")
    add_model_args(parser)
    return parser


def _rmse_np(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _predict(model, x_raw, device):
    model.eval()
    with torch.no_grad():
        pred = model(pinn_common.to_tensor(x_raw, device))[0].cpu().numpy()
    return np.nan_to_num(pred, nan=0.0, posinf=1e7, neginf=-1e7)


def _layered_state_features(model, x_raw, device):
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
        arr("top_oxidation"),
        arr("top_oh", "raw"),
        arr("top_capture"),
        arr("precursor_flux"),
        arr("side_oxid", "raw"),
        arr("side_oh_pre", "raw"),
        arr("side_oh_after", "raw"),
        arr("side_cross", "raw"),
        arr("buried_oh", "raw"),
        arr("deep_cross", "raw"),
        arr("deep_escape", "raw"),
        arr("deep_oh", "raw"),
        arr("si_o_survival", "raw"),
        arr("transport", "log"),
        arr("transient", "log"),
        arr("power_gain", "log"),
        arr("residual_oh", "raw"),
        arr("network_crosslink", "raw"),
        arr("damage_factor", "raw"),
        arr("densification", "raw"),
        arr("incorporation_rate"),
        arr("rate_residual", "raw"),
        arr("uniformity_range_profile"),
        arr("uniformity_sigma_profile"),
        arr("uniformity_range_direct"),
        arr("uniformity_sigma_direct"),
        arr("uniformity_residual", "raw"),
        arr("radial_a_teos"),
        arr("radial_a_oxygen"),
        arr("radial_a_edge"),
        arr("radial_center_bias", "raw"),
        arr("radial_ring_bias", "raw"),
        arr("radial_ring_width"),
        arr("local_rate_std", "raw"),
        aux["profile_coeffs"].detach().cpu().numpy().astype(np.float32),
        phys["source_state"].detach().cpu().numpy().astype(np.float32),
        phys["cond_state"].detach().cpu().numpy().astype(np.float32),
        phys_arr("teos_feed", "log1p"),
        phys_arr("oxygen_feed", "log1p"),
        phys_arr("total_flow", "log1p"),
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


def _parse_calibrator_targets(target_spec):
    if target_spec is None:
        return []
    if isinstance(target_spec, str):
        raw_items = [item.strip() for item in target_spec.split(",")]
    else:
        raw_items = [str(item).strip() for item in target_spec]

    target_names = []
    for item in raw_items:
        if not item:
            continue
        if item not in TARGET_ORDER:
            raise ValueError(
                f"Unknown calibrator target '{item}'. Valid names: {', '.join(TARGET_ORDER)}"
            )
        if item not in target_names:
            target_names.append(item)
    return target_names


def _fit_state_calibrator(args, model, x_fit, y_fit, device, seed):
    if not bool(args.state_calibrator):
        return None
    if XGBRegressor is None:
        return None

    base_pred = _predict(model, x_fit, device)
    features = _layered_state_features(model, x_fit, device)
    target_names = _parse_calibrator_targets(getattr(args, "calibrator_targets", ""))
    if not target_names:
        return None
    target_indices = [IDX[name] for name in target_names]
    models = []
    for target_idx, target_name in zip(target_indices, target_names):
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
        "target_names": target_names,
        "target_indices": target_indices,
        "models": models,
    }


def _uniformity_refiner_features(model, x_raw, base_pred, device):
    state_feat = _layered_state_features(model, x_raw, device)

    h = np.clip(base_pred[:, [IDX["Thickness"]]], 1e-6, None)
    h_max = np.clip(base_pred[:, [IDX["MAX"]]], 0.0, None)
    h_min = np.clip(base_pred[:, [IDX["MIN"]]], 0.0, None)
    rate = np.clip(base_pred[:, [IDX["Deposition_Rate"]]], 0.0, None)
    u_range = np.clip(base_pred[:, [IDX["Uniformity_Range"]]], 0.0, None)
    u_sigma = np.clip(base_pred[:, [IDX["Uniformity_1sigma"]]], 0.0, None)

    u_from_maxmin = (h_max - h_min) * 100.0 / (2.0 * np.clip(h, 1e-6, None))

    time_idx = model.feature_indices.get("Time")
    if time_idx is not None:
        time_min = np.clip(x_raw[:, [time_idx]].astype(np.float32) / 60.0, 1e-6, None)
        h_from_rate = rate * time_min
        rate_thickness_gap = np.abs(h - h_from_rate) / np.clip(h, 1e-6, None)
    else:
        rate_thickness_gap = np.zeros_like(h)

    consistency_feat = np.concatenate(
        [
            h,
            h_max,
            h_min,
            rate,
            u_range,
            u_sigma,
            u_from_maxmin,
            u_range - u_from_maxmin,
            np.abs(u_range - u_from_maxmin),
            rate_thickness_gap,
        ],
        axis=1,
    ).astype(np.float32)

    features = np.concatenate([state_feat, consistency_feat], axis=1).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)


def _fit_uniformity_refiner(args, model, x_fit, y_fit, full_calibrator, device, seed):
    if not bool(args.uniformity_refiner):
        return None
    if XGBRegressor is None:
        return None

    base_pred = _predict_with_state_calibrator(
        model,
        full_calibrator,
        x_fit,
        device,
        rate_thickness_blend=0.0,
        range_max_blend=0.0,
    )

    features = _uniformity_refiner_features(model, x_fit, base_pred, device)

    target_names = ["Uniformity_Range", "Uniformity_1sigma"]
    target_indices = [IDX[name] for name in target_names]
    models = []
    for target_idx, target_name in zip(target_indices, target_names):
        target = y_fit[:, target_idx] - base_pred[:, target_idx]
        reg = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=args.uniformity_refiner_n_estimators,
            max_depth=args.uniformity_refiner_max_depth,
            learning_rate=args.uniformity_refiner_learning_rate,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=args.uniformity_refiner_reg_alpha,
            reg_lambda=args.uniformity_refiner_reg_lambda,
            min_child_weight=2.0,
            random_state=int(seed) + 991 * (target_idx + 1),
            n_jobs=1,
        )
        reg.fit(features, target, eval_set=[(features, target)], verbose=False)
        models.append((target_name, reg))

    return {
        "blend": float(np.clip(args.uniformity_refiner_blend, 0.0, 1.0)),
        "target_names": target_names,
        "target_indices": target_indices,
        "models": models,
    }


def _apply_uniformity_refiner(model, uniformity_refiner, pred, x_raw, device):
    if uniformity_refiner is None:
        return pred

    out = np.array(pred, dtype=np.float32, copy=True)
    features = _uniformity_refiner_features(model, x_raw, out, device)
    refined = np.column_stack(
        [reg.predict(features) for _, reg in uniformity_refiner["models"]]
    ).astype(np.float32)

    refined[:, 0] = np.clip(refined[:, 0], -0.60, 0.60)
    refined[:, 1] = np.clip(refined[:, 1], -0.35, 0.35)
    blend = float(uniformity_refiner.get("blend", 0.35))
    target_indices = uniformity_refiner["target_indices"]
    out[:, target_indices] = out[:, target_indices] + blend * refined
    out[:, target_indices] = np.clip(out[:, target_indices], 0.0, None)
    return out


def _fit_raw_uniformity_models(args, x_fit, y_fit, seed):
    if XGBRegressor is None:
        return None

    models = {}
    for target_name in ["Uniformity_Range", "Uniformity_1sigma"]:
        idx = IDX[target_name]

        reg = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=500,
            max_depth=2,
            learning_rate=0.025,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=0.20,
            reg_lambda=6.0,
            min_child_weight=3.0,
            random_state=int(seed) + 2027 * (idx + 1),
            n_jobs=1,
        )
        reg.fit(x_fit, y_fit[:, idx], eval_set=[(x_fit, y_fit[:, idx])], verbose=False)
        models[target_name] = reg

    return models


def _apply_raw_uniformity_models(raw_uniformity_models, pred, x_raw):
    if raw_uniformity_models is None:
        return pred

    out = np.array(pred, dtype=np.float32, copy=True)
    for target_name in ["Uniformity_Range", "Uniformity_1sigma"]:
        idx = IDX[target_name]
        out[:, idx] = raw_uniformity_models[target_name].predict(x_raw)

    out[:, IDX["Uniformity_Range"]] = np.clip(out[:, IDX["Uniformity_Range"]], 0.0, None)
    out[:, IDX["Uniformity_1sigma"]] = np.clip(out[:, IDX["Uniformity_1sigma"]], 0.0, None)
    return out


def _select_uniformity_refiner_blend(model, refiner, base_pred_val, x_val, y_val, device):
    if refiner is None:
        return 0.0

    candidate_blends = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    best_blend = 0.0
    best_score = float("inf")

    y_range = y_val[:, IDX["Uniformity_Range"]]
    y_sigma = y_val[:, IDX["Uniformity_1sigma"]]

    for blend in candidate_blends:
        tmp_refiner = dict(refiner)
        tmp_refiner["blend"] = float(blend)

        pred_val = _apply_uniformity_refiner(
            model,
            tmp_refiner,
            base_pred_val,
            x_val,
            device,
        )
        pred_val = _sanitize_predictions(pred_val)

        p_range = pred_val[:, IDX["Uniformity_Range"]]
        p_sigma = pred_val[:, IDX["Uniformity_1sigma"]]

        rmse_range = _rmse_np(y_range, p_range)
        rmse_sigma = _rmse_np(y_sigma, p_sigma)
        std_penalty = (
            abs(np.std(p_range) - np.std(y_range)) / (np.std(y_range) + 1e-6)
            + abs(np.std(p_sigma) - np.std(y_sigma)) / (np.std(y_sigma) + 1e-6)
        )
        score = rmse_range + rmse_sigma + 0.15 * std_penalty
        if score < best_score:
            best_score = score
            best_blend = float(blend)

    return best_blend


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
        pred = np.array(base_pred, dtype=np.float32, copy=True)
        target_indices = calibrator.get("target_indices", [])
        if target_indices:
            features = _layered_state_features(model, x_raw, device)
            model_outputs = np.column_stack(
                [reg.predict(features) for _, reg in calibrator["models"]]
            ).astype(np.float32)
            blend = float(calibrator.get("blend", 1.0))
            if calibrator.get("mode") == "direct":
                pred[:, target_indices] = (
                    (1.0 - blend) * base_pred[:, target_indices]
                    + blend * model_outputs
                )
            else:
                pred[:, target_indices] = (
                    base_pred[:, target_indices]
                    + blend * model_outputs
                )
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
    model = KimLayeredSurfacePINN(
        feature_ref=feature_ref,
        feature_indices=feature_indices,
        ri_init=ri_init,
        stress_init=stress_init,
        radial_bins=args.radial_bins,
        radial_modes=args.radial_modes,
        ai_strength=args.layered_ai_strength,
        uniformity_direct_blend=args.uniformity_direct_blend,
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
    uniformity_gap_np = np.abs(
        y_train_raw[:, IDX["Uniformity_Range"]] - uniformity_from_profile_stats
    ).astype(np.float32)
    tol = max(float(args.uniformity_consistency_tol), 1e-6)
    uniformity_ok_np = (uniformity_gap_np <= tol).astype(np.float32)
    uniformity_weight_np = np.exp(-uniformity_gap_np / tol).astype(np.float32)
    uniformity_weight_np = 0.35 + 0.65 * uniformity_weight_np
    uniformity_weight_t = pinn_common.to_tensor(uniformity_weight_np.reshape(-1, 1), device)

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

    teos_idx = feature_indices["TEOS"]
    o2_idx = feature_indices["O2"]
    power_idx = feature_indices["Power"]
    temp_idx = feature_indices["Temperature"]
    time_idx = feature_indices["Time"]

    epoch_iter = tqdm(
        range(1, total_epochs + 1),
        desc=f"PINN20260522-KeyResidual seed={args.seed}",
        leave=False,
    )
    for epoch in epoch_iter:
        model.train()
        optimizer.zero_grad()

        if epoch <= args.epochs_pretrain:
            x_in = x_train_t
        else:
            x_in = x_train_t.detach().clone().requires_grad_(True)

        pred_raw, aux = model(x_in)
        l_data, l_thk, l_log_thk, l_range, l_rate, l_log_rate, l_uni, l_ri, l_stress = data_losses(pred_raw)

        if epoch <= args.epochs_pretrain:
            l_closure = torch.zeros((), device=device)
            l_layer_state = torch.zeros((), device=device)
            l_monotonic = torch.zeros((), device=device)
            l_profile = torch.zeros((), device=device)
            l_rate_residual = torch.zeros((), device=device)
            l_precursor_residual = torch.zeros((), device=device)
            l_surface_oh_residual = torch.zeros((), device=device)
            l_growth_residual = torch.zeros((), device=device)
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

            l_layer_state = (
                torch.mean(torch.relu(aux["side_oxid"] - 0.65 * aux["top_oh"]) ** 2)
                + torch.mean(torch.relu(aux["side_cross"] - aux["side_oh_pre"]) ** 2)
                + torch.mean(torch.relu(aux["deep_cross"] - aux["buried_oh"]) ** 2)
                + torch.mean(torch.relu(aux["deep_oh"] - aux["buried_oh"]) ** 2)
            )

            dep_rate_grad = torch.autograd.grad(
                aux["dep_rate"],
                x_in,
                grad_outputs=torch.ones_like(aux["dep_rate"]),
                create_graph=True,
                retain_graph=True,
            )[0]
            oxygen_grad = torch.autograd.grad(
                aux["n_oxygen"],
                x_in,
                grad_outputs=torch.ones_like(aux["n_oxygen"]),
                create_graph=True,
                retain_graph=True,
            )[0]
            deep_cross_grad = torch.autograd.grad(
                aux["deep_cross"],
                x_in,
                grad_outputs=torch.ones_like(aux["deep_cross"]),
                create_graph=True,
                retain_graph=True,
            )[0]
            si_o_grad = torch.autograd.grad(
                aux["si_o_survival"],
                x_in,
                grad_outputs=torch.ones_like(aux["si_o_survival"]),
                create_graph=True,
                retain_graph=True,
            )[0]

            l_monotonic = (
                torch.mean(torch.relu(-dep_rate_grad[:, [teos_idx]]) ** 2)
                + torch.mean(torch.relu(-oxygen_grad[:, [o2_idx]]) ** 2)
                + torch.mean(torch.relu(-oxygen_grad[:, [power_idx]]) ** 2)
                + torch.mean(torch.relu(-deep_cross_grad[:, [temp_idx]]) ** 2)
                + 0.5 * torch.mean(torch.relu(-deep_cross_grad[:, [time_idx]]) ** 2)
                + 0.6 * torch.mean(torch.relu(si_o_grad[:, [power_idx]]) ** 2)
            )

            key_res = model.key_deposition_residuals(aux)
            l_precursor_residual = key_res["precursor_residual"]
            l_surface_oh_residual = key_res["surface_oh_residual"]
            l_growth_residual = key_res["growth_residual"]

            total_loss = (
                l_data
                + args.lambda_closure * l_closure
                + args.lambda_layer_state * l_layer_state
                + args.lambda_monotonic * l_monotonic
                + args.lambda_profile_smooth * l_profile
                + args.lambda_rate_residual * l_rate_residual
                + args.lambda_precursor_residual * l_precursor_residual
                + args.lambda_surface_oh_residual * l_surface_oh_residual
                + args.lambda_growth_residual * l_growth_residual
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
        score_weights = np.array([1.0, 0.45, 0.45, 0.55, 0.35, 0.32, 0.12, 0.18])
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
                "phys_layer_state_loss": float(l_layer_state.item()),
                "phys_monotonic_loss": float(l_monotonic.item()),
                "phys_profile_smooth_loss": float(l_profile.item()),
                "phys_rate_residual_loss": float(l_rate_residual.item()),
                "pinn_precursor_residual_loss": float(l_precursor_residual.item()),
                "pinn_surface_oh_residual_loss": float(l_surface_oh_residual.item()),
                "pinn_growth_residual_loss": float(l_growth_residual.item()),
                "train_thk_rmse": train_thk_rmse,
                "val_thk_rmse": val_thk_rmse,
                "val_score": float(val_score),
                "mean_top_oh": float(aux["top_oh"].detach().mean().cpu().item()),
                "mean_side_cross": float(aux["side_cross"].detach().mean().cpu().item()),
                "mean_deep_cross": float(aux["deep_cross"].detach().mean().cpu().item()),
                "mean_si_o_survival": float(aux["si_o_survival"].detach().mean().cpu().item()),
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
    raw_uniformity_models = _fit_raw_uniformity_models(
        args,
        x_calib,
        y_calib,
        args.seed,
    )

    train_pred = _predict_with_state_calibrator(
        model,
        state_calibrator,
        x_train_raw,
        device,
        args.rate_thickness_blend,
        args.range_max_blend,
    )
    train_pred = _apply_raw_uniformity_models(raw_uniformity_models, train_pred, x_train_raw)
    val_pred = _predict_with_state_calibrator(
        model,
        state_calibrator,
        x_val_raw,
        device,
        args.rate_thickness_blend,
        args.range_max_blend,
    )
    val_pred = _apply_raw_uniformity_models(raw_uniformity_models, val_pred, x_val_raw)
    test_pred = _predict_with_state_calibrator(
        model,
        state_calibrator,
        x_test_raw,
        device,
        args.rate_thickness_blend,
        args.range_max_blend,
    )
    test_pred = _apply_raw_uniformity_models(raw_uniformity_models, test_pred, x_test_raw)

    train_pred = _sanitize_predictions(train_pred)
    val_pred = _sanitize_predictions(val_pred)
    test_pred = _sanitize_predictions(test_pred)

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
            "layered_ai_strength": float(args.layered_ai_strength),
            "uniformity_direct_blend": float(args.uniformity_direct_blend),
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
                "lambda_layer_state": args.lambda_layer_state,
                "lambda_monotonic": args.lambda_monotonic,
                "lambda_profile_smooth": args.lambda_profile_smooth,
                "lambda_rate_residual": args.lambda_rate_residual,
            },
            "uniformity_consistency_tol": float(args.uniformity_consistency_tol),
            "uniformity_train_points_used": int(uniformity_ok_np.sum()),
            "uniformity_weight_mean": float(np.mean(uniformity_weight_np)),
            "uniformity_weight_min": float(np.min(uniformity_weight_np)),
            "uniformity_weight_max": float(np.max(uniformity_weight_np)),
            "state_calibrator": {
                "enabled": bool(state_calibrator is not None),
                "use_val": bool(args.calibrator_use_val),
                "fit_count": int(len(x_calib)),
                "feature_count": int(_layered_state_features(model, x_train_raw[:1], device).shape[1]),
                "n_estimators": int(args.calibrator_n_estimators),
                "max_depth": int(args.calibrator_max_depth),
                "mode": str(args.calibrator_mode),
                "blend": float(args.calibrator_blend),
                "requested_targets": _parse_calibrator_targets(args.calibrator_targets),
                "fitted_targets": list(state_calibrator.get("target_names", [])) if state_calibrator else [],
                "rate_thickness_blend": float(args.rate_thickness_blend),
                "range_max_blend": float(args.range_max_blend),
            },
            "uniformity_raw_xgb": {
                "enabled": bool(raw_uniformity_models is not None),
                "targets": list(raw_uniformity_models.keys()) if raw_uniformity_models else [],
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