"""Calibrate benchmark-unit gains from the screened stimulus generator."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from coupling_sources.core.data import StimulusSpec
from coupling_sources.core.stimuli import generate_stimulus

DEFAULT_GAIN_CALIBRATION_SEED = 20260711
DEFAULT_GAIN_CALIBRATION_REPLICATES = 2000

GAIN_CALIBRATION_DESIGN_FIELDS = (
    "n_sources",
    "n_trials_full",
    "n_trials_partial",
    "sigma",
    "lam",
    "coupled_sources",
    "resultant_floor_full",
    "resultant_floor_leave_one_out",
)


@dataclass(frozen=True)
class GainCalibration:
    s_coupled: float
    s_coupled_se: float
    s_mixed: float
    s_mixed_se: float
    s_independent: float
    s_independent_se: float
    delta_y: float
    delta_y_se: float
    delta_z: float
    delta_z_se: float
    g_star_y: float
    g_star_y_se: float
    g_star_z: float
    g_star_z_se: float


@dataclass(frozen=True)
class GainCalibrationStability:
    g_star_y_relative_mc_se: float
    g_star_z_relative_mc_se: float
    max_relative_mc_se: float
    g_star_y_mc_interval_95: tuple[float, float]
    g_star_z_mc_interval_95: tuple[float, float]


def gain_calibration_key(spec: StimulusSpec) -> tuple[object, ...]:
    """Return the stimulus fields that determine calibrated gain constants."""
    return tuple(getattr(spec, name) for name in GAIN_CALIBRATION_DESIGN_FIELDS)


def gain_calibration_design(spec: StimulusSpec) -> dict[str, object]:
    """Return the serialized stimulus fields that determine calibrated gains."""
    design = {
        name: getattr(spec, name) for name in GAIN_CALIBRATION_DESIGN_FIELDS
    }
    design["coupled_sources"] = list(spec.coupled_sources)
    return design


def gain_calibration_spec(
    design: StimulusSpec,
    *,
    seed: int,
    n_monte_carlo_replicates: int,
) -> StimulusSpec:
    """Turn a design into independent Monte Carlo trial-list replicates."""
    return replace(
        design,
        seed=seed,
        n_participants=n_monte_carlo_replicates,
        n_rounds=1,
    )


def gain_calibration_record(
    spec: StimulusSpec,
    calibration: GainCalibration,
    *,
    mode: Literal["auto", "fixed"],
) -> dict[str, object]:
    """Return the reproducible design and estimates behind calibrated gains."""
    return {
        "mode": mode,
        "seed": spec.seed,
        "n_monte_carlo_replicates": spec.n_participants,
        **gain_calibration_design(spec),
        "estimates": asdict(calibration),
        "stability": asdict(gain_calibration_stability(calibration)),
    }


def gain_calibration_stability(
    calibration: GainCalibration,
) -> GainCalibrationStability:
    """Summarize Monte Carlo uncertainty in the two calibrated gains."""
    relative_y = calibration.g_star_y_se / calibration.g_star_y
    relative_z = calibration.g_star_z_se / calibration.g_star_z
    half_width_y = 1.96 * calibration.g_star_y_se
    half_width_z = 1.96 * calibration.g_star_z_se
    return GainCalibrationStability(
        g_star_y_relative_mc_se=relative_y,
        g_star_z_relative_mc_se=relative_z,
        max_relative_mc_se=max(relative_y, relative_z),
        g_star_y_mc_interval_95=(
            calibration.g_star_y - half_width_y,
            calibration.g_star_y + half_width_y,
        ),
        g_star_z_mc_interval_95=(
            calibration.g_star_z - half_width_z,
            calibration.g_star_z + half_width_z,
        ),
    )


def _mean_and_se(values: NDArray[np.float64]) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(values.size))


def _mean_signal_by_role(
    visible_reports: NDArray[np.float64],
    coupled_sources: tuple[int, int],
) -> dict[str, NDArray[np.float64]]:
    """Return each list's mean visible-pair signal by structural role."""
    reports = np.asarray(visible_reports, dtype=np.float64)
    if reports.ndim < 3 or reports.shape[-1] != 4:
        raise ValueError("visible_reports must have shape (list, ..., trial, 4)")
    reports = reports.reshape((reports.shape[0], -1, reports.shape[-1]))

    visible_sources = np.isfinite(reports)
    safe_reports = np.nan_to_num(reports, nan=0.0)
    off_diagonal = ~np.eye(reports.shape[-1], dtype=bool)
    visible_pairs = (
        visible_sources[..., :, None]
        & visible_sources[..., None, :]
        & off_diagonal
    )
    pair_agreement = np.cos(
        safe_reports[..., :, None] - safe_reports[..., None, :]
    )
    pair_count = np.maximum(
        visible_pairs.sum(axis=(-2, -1), keepdims=True),
        1,
    )
    mean_agreement = (
        (pair_agreement * visible_pairs).sum(axis=(-2, -1), keepdims=True)
        / pair_count
    )
    relative_agreement = 0.5 * (pair_agreement - mean_agreement) * visible_pairs

    coupled = tuple(coupled_sources)
    independent = tuple(
        source for source in range(reports.shape[-1]) if source not in coupled
    )
    role_pairs = {
        "coupled": (coupled,),
        "mixed": tuple((i, j) for i in coupled for j in independent),
        "independent": (independent,),
    }

    role_means: dict[str, NDArray[np.float64]] = {}
    for role, pairs in role_pairs.items():
        values = np.stack(
            [relative_agreement[..., i, j] for i, j in pairs],
            axis=-1,
        )
        observed = np.stack(
            [visible_pairs[..., i, j] for i, j in pairs],
            axis=-1,
        )
        counts = observed.sum(axis=(1, 2))
        if np.any(counts == 0):
            raise ValueError(
                f"no visible {role} pairs in at least one Monte Carlo replicate"
            )
        role_means[role] = values.sum(axis=(1, 2)) / counts
    return role_means


def calibrate_gain_constants(spec: StimulusSpec) -> GainCalibration:
    """Estimate role means, branch score gaps, and calibrated gains."""
    if spec.n_participants < 2:
        raise ValueError(
            "gain calibration requires at least two Monte Carlo replicates"
        )
    if spec.n_rounds != 1:
        raise ValueError("gain calibration requires n_rounds=1 per replicate")
    if spec.n_trials_full + spec.n_trials_partial == 0:
        raise ValueError("gain calibration requires at least one trial per list")

    stimuli = generate_stimulus(spec)
    role_means = _mean_signal_by_role(
        np.asarray(stimuli.visible_reports),
        spec.coupled_sources,
    )

    s_coupled = np.asarray(role_means["coupled"], dtype=np.float64)
    s_mixed = np.asarray(role_means["mixed"], dtype=np.float64)
    s_independent = np.asarray(role_means["independent"], dtype=np.float64)
    delta_y_values = (s_coupled - s_independent) / 3.0
    delta_z_values = s_coupled - s_mixed

    s_coupled_mean, s_coupled_se = _mean_and_se(s_coupled)
    s_mixed_mean, s_mixed_se = _mean_and_se(s_mixed)
    s_independent_mean, s_independent_se = _mean_and_se(s_independent)
    delta_y, delta_y_se = _mean_and_se(delta_y_values)
    delta_z, delta_z_se = _mean_and_se(delta_z_values)
    if delta_y <= 0.0 or delta_z <= 0.0:
        raise ValueError(
            "calibrated score gaps must be positive; "
            f"got delta_y={delta_y:.6g}, delta_z={delta_z:.6g}"
        )

    log_ratio_y = math.log(1.0 + spec.lam)
    log_ratio_z = math.log((1.0 + 2.0 * spec.lam) / (1.0 - spec.lam))
    g_star_y = log_ratio_y / delta_y
    g_star_z = log_ratio_z / delta_z
    g_star_y_se = log_ratio_y * delta_y_se / delta_y**2
    g_star_z_se = log_ratio_z * delta_z_se / delta_z**2

    return GainCalibration(
        s_coupled=s_coupled_mean,
        s_coupled_se=s_coupled_se,
        s_mixed=s_mixed_mean,
        s_mixed_se=s_mixed_se,
        s_independent=s_independent_mean,
        s_independent_se=s_independent_se,
        delta_y=delta_y,
        delta_y_se=delta_y_se,
        delta_z=delta_z,
        delta_z_se=delta_z_se,
        g_star_y=g_star_y,
        g_star_y_se=g_star_y_se,
        g_star_z=g_star_z,
        g_star_z_se=g_star_z_se,
    )
