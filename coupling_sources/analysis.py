from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from coupling_sources.core.constants import HIDDEN_SOURCE_PROBE
from coupling_sources.core.data import SimulationBatch


@dataclass(frozen=True)
class EffectSpec:
    label: str
    color: str
    params: Mapping[str, float]


def simulation_frame(simulation: SimulationBatch) -> pd.DataFrame:
    responses = np.asarray(simulation.responses)
    n_participants, n_rounds, n_trials = responses.shape
    participant, round_, trial = np.indices(responses.shape)
    stimuli = simulation.stimuli
    visible_reports = np.asarray(stimuli.visible_reports)
    params = simulation.participant_parameters
    if params is None:
        raise ValueError("simulation is missing participant_parameters")
    diagnostics = simulation.diagnostics
    if diagnostics is None:
        raise ValueError("simulation is missing diagnostics")
    pair_scores = np.asarray(diagnostics.pair_scores)
    off_diagonal = 1.0 - np.eye(pair_scores.shape[-1])
    mean_pair_score = (pair_scores * off_diagonal).sum(axis=(-2, -1)) / max(int(off_diagonal.sum()), 1)
    is_probe = (jnp.asarray(stimuli.trial_type) == HIDDEN_SOURCE_PROBE) & jnp.asarray(stimuli.valid_trials)
    truth_log_prob = dist.VonMises(diagnostics.truth_mean_angle, params["vmConcentration"]).log_prob(
        simulation.truth_estimates
    )
    hidden_log_prob = dist.VonMises(diagnostics.hidden_mean_angle, params["vmConcentration"]).log_prob(
        simulation.hidden_predictions
    )
    response_log_prob = jnp.where(
        stimuli.valid_trials,
        jnp.where(is_probe, hidden_log_prob, truth_log_prob),
        jnp.nan,
    )
    return pd.DataFrame(
        {
            "participant": participant.ravel() + 1,
            "round": round_.ravel() + 1,
            "trial": trial.ravel() + 1,
            "response": responses.ravel(),
            "truth_estimate": np.asarray(simulation.truth_estimates).ravel(),
            "hidden_prediction": np.asarray(simulation.hidden_predictions).ravel(),
            "hidden_source": np.asarray(stimuli.hidden_source).ravel(),
            "trial_type": np.asarray(stimuli.trial_type).ravel(),
            "valid_trial": np.asarray(stimuli.valid_trials).ravel().astype(bool),
            "n_visible_sources": np.isfinite(visible_reports).sum(axis=-1).ravel(),
            "is_probe": np.asarray(is_probe).ravel().astype(bool),
            "truth_mean_angle": np.asarray(diagnostics.truth_mean_angle).ravel(),
            "hidden_mean_angle": np.asarray(diagnostics.hidden_mean_angle).ravel(),
            "response_mean_angle": np.asarray(diagnostics.response_mean_angle).ravel(),
            "response_log_prob": np.asarray(response_log_prob).ravel(),
            "mean_pair_score": mean_pair_score.ravel(),
        }
    )


def posterior_frame(samples: Mapping[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for parameter, values in samples.items():
        array = np.asarray(values)
        for chain_index, chain_values in enumerate(array, start=1):
            for draw_index, value in enumerate(chain_values, start=1):
                rows.append({"parameter": parameter, "chain": chain_index, "draw": draw_index, "value": float(value)})
    return pd.DataFrame(rows)


def plot_posterior_histograms(frame: pd.DataFrame, true_values: Mapping[str, float] | None = None) -> tuple[Figure, np.ndarray]:
    parameters = list(frame["parameter"].drop_duplicates())
    ncols = 2
    nrows = int(np.ceil(max(len(parameters), 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.2, 2.6 * nrows), squeeze=False)
    for ax, parameter in zip(axes.ravel(), parameters):
        values = frame.loc[frame.parameter == parameter, "value"]
        ax.hist(values, bins=35, alpha=0.72, density=True)
        if true_values is not None and parameter in true_values:
            ax.axvline(true_values[parameter], lw=2, label="simulated value")
            ax.legend(frameon=False, fontsize=8)
        ax.set(title=parameter, yticks=[])
        ax.spines[["top", "right", "left"]].set_visible(False)
    for ax in axes.ravel()[len(parameters):]:
        ax.set_axis_off()
    fig.tight_layout()
    return fig, axes


def plot_response_series(frame: pd.DataFrame) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    summary = cast(
        pd.DataFrame,
        frame.groupby("trial", as_index=False)[["response"]].mean(),
    )
    ax.plot(summary["trial"], summary["response"])
    ax.set(xlabel="trial", ylabel="mean response angle", title="Simulated responses")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax
