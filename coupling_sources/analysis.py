from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from coupling_sources.core.parameters.simulation import SimParams
from coupling_sources.core.response import ResponseTrace, compute_response_trace
from coupling_sources.core.simulation import SimulationResult, simulate_choices
from coupling_sources.core.stimuli import generate_stimuli
from coupling_sources.utils.angles import circular_difference_deg, optimal_weights


@dataclass(frozen=True)
class EffectSpec:
    label: str
    color: str
    params: SimParams


SOURCE_COLORS = ("#E69F00", "#56B4E9", "#009E73", "#A710DE")
SUMMARY_STYLES = {
    "truth": {"color": "black", "radius": 0.96, "lw": 2.2, "linestyle": "-"},
    "equal_mean": {"color": "#0072B2", "radius": 0.84, "lw": 1.8, "linestyle": "--"},
    "optimal_mean": {"color": "#D55E00", "radius": 0.90, "lw": 1.8, "linestyle": "-."},
}


def trace_frame(
    result: SimulationResult, n_sources: int, coupled: tuple[int, int]
) -> pd.DataFrame:
    choices = np.asarray(result.choices)
    n_rounds, n_trials = choices.shape
    weights = np.asarray(result.trace.source_weights)
    scores = np.asarray(result.trace.pre_update_agreement_scores)
    coupled_idx = np.asarray(coupled)
    independent_idx = np.asarray([i for i in range(n_sources) if i not in coupled])
    return pd.DataFrame({
        "round": np.repeat(np.arange(1, n_rounds + 1), n_trials),
        "trial": np.tile(np.arange(1, n_trials + 1), n_rounds),
        "coupled_agreement_score": scores[..., coupled[0], coupled[1]].ravel(),
        "coupled_weight": weights[..., coupled_idx].mean(axis=-1).ravel(),
        "independent_weight": weights[..., independent_idx].mean(axis=-1).ravel(),
        "mu_rad": np.asarray(result.trace.response_mean).ravel(),
        "truth_mu_rad": np.asarray(result.trace.truth_mean).ravel(),
        "hidden_mu_rad": np.asarray(result.trace.hidden_mean).ravel(),
        "choice_logpdf": np.asarray(result.trace.choice_logpdf).ravel(),
        "is_probe": np.asarray(result.trace.is_probe).ravel().astype(bool),
        "hidden_source": np.asarray(result.trace.hidden_source).ravel(),
    })


def plot_stimulus_circles(
    stimuli: pd.DataFrame,
    source_idx: Sequence[int],
    ncols: int = 4,
    source_colors: Sequence[str] = SOURCE_COLORS,
) -> tuple[Figure, np.ndarray]:
    nrows = max(1, int(np.ceil(len(stimuli) / ncols)))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10, 2.35 * nrows), subplot_kw={"projection": "polar"}
    )
    axes = np.atleast_1d(axes).ravel()

    for trial, ax in enumerate(axes, start=1):
        ax.set_ylim(0, 1.0)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["polar"].set_color("0.80")
        if trial > len(stimuli):
            ax.set_axis_off()
            continue

        row = stimuli.iloc[trial - 1]
        ax.set_title(f"trial {trial}", fontsize=9, pad=4)
        for source in source_idx:
            ax.annotate(
                "",
                xy=(np.deg2rad(row[source]), 0.66),
                xytext=(np.deg2rad(row[source]), 0.0),
                arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": source_colors[source]},
            )
        for column, style in SUMMARY_STYLES.items():
            ax.annotate(
                "",
                xy=(np.deg2rad(row[column]), style["radius"]),
                xytext=(np.deg2rad(row[column]), 0.0),
                arrowprops={
                    "arrowstyle": "-|>",
                    "lw": style["lw"],
                    "color": style["color"],
                    "linestyle": style["linestyle"],
                },
            )

    handles = [
        plt.Line2D([0], [0], color=source_colors[s], lw=1.8, label=str(s))
        for s in source_idx
    ]
    handles += [
        plt.Line2D([0], [0], color=s["color"], lw=s["lw"], linestyle=s["linestyle"], label=k)
        for k, s in SUMMARY_STYLES.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig, axes


def generate_effect_reports(
    key: jax.Array,
    n_runs: int,
    n_trials: int,
    n_sources: int,
    sigma: float | jax.Array,
    lam: float | jax.Array,
    coupled: tuple[int, int],
) -> jax.Array:
    batch = generate_stimuli(
        key,
        n_rounds=n_runs,
        n_trials_full=n_trials,
        n_trials_partial=0,
        n_sources=n_sources,
        sigma=sigma,
        lam=lam,
        coupled=coupled,
    )
    return jnp.moveaxis(batch.source_choices, 0, -1)


def _non_coupled_pair_mask(n_sources: int, coupled: tuple[int, int]) -> jax.Array:
    pair_mask = jnp.triu(jnp.ones((n_sources, n_sources), dtype=bool), k=1)
    return pair_mask.at[tuple(sorted(coupled))].set(False).astype(float)


def _agreement_targets(
    reports: jax.Array,
    n_sources: int,
    coupled: tuple[int, int],
) -> tuple[jax.Array, jax.Array]:
    distance = reports[..., :, None] - reports[..., None, :]
    agreement = jnp.cos(distance)
    non_coupled_mask = _non_coupled_pair_mask(n_sources, coupled)
    non_coupled_agreement = (
        jnp.sum(agreement * non_coupled_mask, axis=(-2, -1))
        / jnp.sum(non_coupled_mask)
    )
    return (
        jnp.mean(agreement[..., coupled[0], coupled[1]]),
        jnp.mean(non_coupled_agreement),
    )


def _effect_flow(reports: jax.Array, params: SimParams) -> ResponseTrace:
    source_choices = jnp.moveaxis(reports, -1, 0)
    hidden_source = jnp.full(reports.shape[:2], jnp.nan)
    return compute_response_trace(params, source_choices, hidden_source)


def _simulate_agreement_score_matrices(
    reports: jax.Array,
    params: SimParams,
    n_sources: int,
    coupled: tuple[int, int],
) -> tuple[jax.Array, jax.Array]:
    scores = _effect_flow(reports, params).post_update_agreement_scores
    non_coupled_mask = _non_coupled_pair_mask(n_sources, coupled)
    non_coupled_scores = (
        jnp.sum(scores * non_coupled_mask, axis=(-2, -1))
        / jnp.sum(non_coupled_mask)
    )
    return scores[..., coupled[0], coupled[1]], non_coupled_scores


def _expected_weight_target(
    gamma: float | jax.Array,
    coupled_agreement: jax.Array,
    non_coupled_agreement: jax.Array,
    n_sources: int,
    coupled: tuple[int, int],
) -> tuple[jax.Array, jax.Array]:
    scores = non_coupled_agreement * (1.0 - jnp.eye(n_sources))
    scores = scores.at[coupled[0], coupled[1]].set(coupled_agreement)
    scores = scores.at[coupled[1], coupled[0]].set(coupled_agreement)
    weights = optimal_weights(jnp.eye(n_sources) + jnp.asarray(gamma) * scores)
    coupled_idx = jnp.asarray(coupled)
    independent_idx = jnp.asarray([i for i in range(n_sources) if i not in coupled])
    return jnp.mean(weights[coupled_idx]), jnp.mean(weights[independent_idx])


def _simulate_weight_matrices(
    reports: jax.Array,
    params: SimParams,
    n_sources: int,
    coupled: tuple[int, int],
) -> tuple[jax.Array, jax.Array]:
    weights = _effect_flow(reports, params).source_weights
    coupled_idx = jnp.asarray(coupled)
    independent_idx = jnp.asarray([i for i in range(n_sources) if i not in coupled])
    return (
        jnp.mean(weights[..., coupled_idx], axis=-1),
        jnp.mean(weights[..., independent_idx], axis=-1),
    )


def _matrix_frame(matrix: jax.Array, sweep: str, panel: str) -> pd.DataFrame:
    values = np.asarray(matrix)
    frame = pd.DataFrame(values, columns=np.arange(1, values.shape[1] + 1))
    frame["run"] = np.arange(values.shape[0])
    frame["sweep"] = sweep
    frame["panel"] = panel
    long = frame.melt(
        id_vars=["run", "sweep", "panel"],
        var_name="trial",
        value_name="value",
    )
    long["trial"] = long.trial.astype(int)
    return long


def _agreement_score_frame(
    reports: jax.Array,
    spec: EffectSpec,
    n_sources: int,
    coupled: tuple[int, int],
) -> pd.DataFrame:
    coupled_matrix, non_coupled_matrix = _simulate_agreement_score_matrices(
        reports, spec.params, n_sources, coupled
    )
    return pd.concat(
        [
            _matrix_frame(coupled_matrix, spec.label, "Coupled pair"),
            _matrix_frame(non_coupled_matrix, spec.label, "Non-coupled pairs"),
        ],
        ignore_index=True,
    )


def agreement_sweep_frame(
    reports: jax.Array,
    specs: Sequence[EffectSpec],
    n_sources: int,
    coupled: tuple[int, int],
) -> pd.DataFrame:
    return pd.concat(
        [_agreement_score_frame(reports, spec, n_sources, coupled) for spec in specs],
        ignore_index=True,
    )


def fixed_agreement_targets_frame(
    reports: jax.Array,
    n_sources: int,
    coupled: tuple[int, int],
) -> pd.DataFrame:
    coupled_target, non_coupled_target = _agreement_targets(
        reports, n_sources, coupled
    )
    return pd.DataFrame([
        {"panel": "Coupled pair", "value": float(coupled_target), "color": "0.25"},
        {"panel": "Non-coupled pairs", "value": float(non_coupled_target), "color": "0.45"},
    ])



def _weight_frame(
    reports: jax.Array,
    spec: EffectSpec,
    n_sources: int,
    coupled: tuple[int, int],
) -> pd.DataFrame:
    coupled_matrix, independent_matrix = _simulate_weight_matrices(
        reports, spec.params, n_sources, coupled
    )
    return pd.concat(
        [
            _matrix_frame(coupled_matrix, spec.label, "Coupled sources"),
            _matrix_frame(independent_matrix, spec.label, "Independent sources"),
        ],
        ignore_index=True,
    )


def weight_sweep_frame(
    reports: jax.Array,
    specs: Sequence[EffectSpec],
    n_sources: int,
    coupled: tuple[int, int],
) -> pd.DataFrame:
    return pd.concat(
        [_weight_frame(reports, spec, n_sources, coupled) for spec in specs],
        ignore_index=True,
    )


def weight_targets_frame(
    reports: jax.Array,
    specs: Sequence[EffectSpec],
    n_sources: int,
    coupled: tuple[int, int],
) -> pd.DataFrame:
    coupled_agreement, non_coupled_agreement = _agreement_targets(
        reports, n_sources, coupled
    )
    return pd.DataFrame([
        {"panel": panel, "value": float(value), "color": spec.color}
        for spec in specs
        for panel, value in zip(
            ("Coupled sources", "Independent sources"),
            _expected_weight_target(
                spec.params.gamma,
                coupled_agreement,
                non_coupled_agreement,
                n_sources,
                coupled,
            ),
        )
    ])


def sweep_palette(specs: Sequence[EffectSpec]) -> dict[str, str]:
    return {spec.label: spec.color for spec in specs}


def _response_error_samples(
    key: jax.Array,
    kappa_p: float,
    n_runs: int,
    source_choices: jax.Array,
    hidden_source: jax.Array,
    response_mu_deg: np.ndarray,
) -> np.ndarray:
    keys = jax.random.split(key, n_runs)
    params = SimParams(alpha=0.20, gamma=0.80, kappa_p=kappa_p)
    choices = jax.vmap(
        lambda subkey: simulate_choices(subkey, params, source_choices, hidden_source).choices
    )(keys)
    choice_deg = np.degrees(np.asarray(choices[:, 0, :]))
    error = circular_difference_deg(choice_deg, response_mu_deg[None, :])
    return np.abs(error).ravel()


def response_error_frame(
    key: jax.Array,
    kappa_values: Sequence[float],
    n_runs: int,
    source_choices: jax.Array,
    hidden_source: jax.Array,
    response_mu_deg: np.ndarray,
) -> pd.DataFrame:
    keys = jax.random.split(key, len(kappa_values))
    return pd.concat(
        [
            pd.DataFrame({
                "kappa_p": kappa_p,
                "absolute_error_deg": _response_error_samples(
                    subkey, kappa_p, n_runs, source_choices, hidden_source, response_mu_deg
                ),
            })
            for subkey, kappa_p in zip(keys, kappa_values)
        ],
        ignore_index=True,
    )


def _separated_label_positions(
    values: pd.Series, ylim: tuple[float, float], min_gap_fraction: float = 0.065
) -> np.ndarray:
    values_array = values.to_numpy(dtype=float)
    if len(values_array) <= 1:
        return values_array

    low, high = ylim
    span = high - low
    min_gap = span * min_gap_fraction
    order = np.argsort(values_array)
    sorted_positions = values_array[order].copy()

    for idx in range(1, len(sorted_positions)):
        sorted_positions[idx] = max(sorted_positions[idx], sorted_positions[idx - 1] + min_gap)

    top_overflow = sorted_positions[-1] - (high - min_gap / 2)
    if top_overflow > 0:
        sorted_positions -= top_overflow

    for idx in range(len(sorted_positions) - 2, -1, -1):
        sorted_positions[idx] = min(sorted_positions[idx], sorted_positions[idx + 1] - min_gap)

    bottom_overflow = low + min_gap / 2 - sorted_positions[0]
    if bottom_overflow > 0:
        sorted_positions += bottom_overflow

    positions = np.empty_like(sorted_positions)
    positions[order] = sorted_positions
    return positions


def plot_effect_lines(
    data: pd.DataFrame,
    targets: pd.DataFrame,
    ylabel: str,
    ylim: tuple[float, float],
    palette: Mapping[str, str],
    n_trials: int,
) -> tuple[Figure, np.ndarray]:
    grid = sns.relplot(
        data=data,
        x="trial",
        y="value",
        hue="sweep",
        col="panel",
        kind="line",
        estimator="mean",
        errorbar="sd",
        palette=palette,
        height=3.2,
        aspect=1.15,
        facet_kws={"sharey": True},
    )
    grid.set_axis_labels("trial", ylabel)
    grid.set_titles("{col_name}")
    grid.set(ylim=ylim, xlim=(1, n_trials + 12))
    sns.move_legend(
        grid,
        "lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(palette),
        title=None,
        frameon=True,
        fontsize=9.0,
    )
    grid.figure.legends[-1].get_frame().set(facecolor="white", alpha=0.96, linewidth=0.0)

    for panel, ax in grid.axes_dict.items():
        panel_targets = targets[targets.panel == panel].reset_index(drop=True)
        label_positions = _separated_label_positions(panel_targets.value, ylim)
        for row, label_y in zip(panel_targets.itertuples(index=False), label_positions):
            ax.axhline(row.value, color=row.color, lw=1.1, ls="--", zorder=1)
            ax.plot(
                [n_trials + 0.8, n_trials + 3.1],
                [row.value, label_y],
                color=row.color,
                lw=0.8,
                alpha=0.75,
                clip_on=False,
                zorder=2,
            )
            ax.text(
                n_trials + 3.4,
                label_y,
                f"{row.value:.2f}",
                color=row.color,
                fontsize=8.5,
                va="center",
                ha="left",
                clip_on=False,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.8},
                zorder=3,
            )
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    grid.figure.subplots_adjust(bottom=0.28, right=0.88, top=0.88, wspace=0.16)
    return grid.figure, grid.axes


def plot_response_error(
    data: pd.DataFrame, kappa_values: Sequence[float]
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    sns.lineplot(
        data=data,
        x="kappa_p",
        y="absolute_error_deg",
        estimator=np.median,
        errorbar=("pi", 50),
        color="#0072B2",
        ax=ax,
    )
    ax.set(
        title="Response concentration",
        xlabel="kappa_p",
        ylabel="absolute response error (deg)",
        xlim=(float(np.min(kappa_values)), float(np.max(kappa_values))),
    )
    ax.set_xticks([2, 8, 16, 24, 32])
    ax.set_xticklabels(["2", "8", "16", "24", "32"])
    ax.minorticks_off()
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax



def plot_posterior_histograms(
    frame: pd.DataFrame, true_values: pd.Series
) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.2))
    for ax, parameter in zip(axes.ravel(), frame.columns):
        ax.hist(frame[parameter], bins=35, color="#0072B2", alpha=0.72, density=True)
        ax.axvline(true_values[parameter], color="#D55E00", lw=2, label="simulated value")
        ax.set(title=parameter, yticks=[])
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, axes
