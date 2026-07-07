"""Single active NumPyro model for learned coupling between sources."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from coupling_sources.core.constants import HIDDEN_SOURCE_PROBE
from coupling_sources.core.data import SimulationBatch, StimulusBatch
from coupling_sources.core.priors import coupling_prior_distributions
from coupling_sources.utils.angles import circular_mean


def model_coupling(batch: StimulusBatch | SimulationBatch) -> None:
    visible_reports = batch.visible_reports
    hidden_source = batch.hidden_source
    trial_type = batch.trial_type
    valid_trials = batch.valid_trials
    truth_estimates = getattr(batch, "truth_estimates", None)
    hidden_predictions = getattr(batch, "hidden_predictions", None)
    n_participants, n_rounds, n_trials, n_sources = visible_reports.shape

    priors = coupling_prior_distributions()
    alpha = numpyro.sample("alpha", priors["alpha"])
    beta_y = numpyro.sample("beta_y", priors["beta_y"])
    beta_z = numpyro.sample("beta_z", priors["beta_z"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    initial_pair_score = numpyro.deterministic("~initial_pair_scores", jnp.zeros(()))

    off_diagonal_bool = ~jnp.eye(n_sources, dtype=bool)
    pair_scores = jnp.broadcast_to(
        (1.0 - jnp.eye(n_sources)) * initial_pair_score,
        (n_participants, n_rounds, n_sources, n_sources),
    )

    with numpyro.plate("participants", n_participants, dim=-2):
        with numpyro.plate("rounds", n_rounds, dim=-1):
            for trial_index in range(n_trials):
                suffix = f"t{trial_index + 1}"
                reports_raw = numpyro.deterministic(
                    f"~visible_reports_{suffix}", visible_reports[:, :, trial_index, :]
                )
                hidden = numpyro.deterministic(
                    f"~hidden_source_{suffix}", hidden_source[:, :, trial_index]
                )
                trial = numpyro.deterministic(
                    f"~trial_type_{suffix}", trial_type[:, :, trial_index]
                )
                valid_trial = numpyro.deterministic(
                    f"~valid_trial_{suffix}", valid_trials[:, :, trial_index]
                )

                visible_sources = numpyro.deterministic(
                    f"~visible_source_mask_{suffix}", jnp.isfinite(reports_raw)
                )
                reports = jnp.nan_to_num(reports_raw, nan=0.0)
                visible_pairs = (
                    visible_sources[..., :, None]
                    & visible_sources[..., None, :]
                    & off_diagonal_bool
                )
                pair_agreement = jnp.cos(reports[..., :, None] - reports[..., None, :])
                visible_pair_values = pair_agreement * visible_pairs
                pair_count = jnp.maximum(
                    jnp.sum(visible_pairs, axis=(-2, -1), keepdims=True), 1
                )
                mean_visible_agreement = (
                    jnp.sum(visible_pair_values, axis=(-2, -1), keepdims=True)
                    / pair_count
                )
                relative_agreement = numpyro.deterministic(
                    f"~relative_agreement_{suffix}",
                    0.5 * (pair_agreement - mean_visible_agreement) * visible_pairs,
                )
                pre_update_pair_scores = numpyro.deterministic(
                    f"~pre_update_pair_scores_{suffix}", pair_scores
                )
                update_pairs = visible_pairs & valid_trial[..., None, None]
                pair_scores = numpyro.deterministic(
                    f"~pair_scores_{suffix}",
                    pre_update_pair_scores
                    + update_pairs
                    * alpha
                    * (relative_agreement - pre_update_pair_scores),
                )

                visible_count = jnp.maximum(
                    jnp.sum(visible_sources, axis=-1, keepdims=True), 1
                )
                truth_denominator = jnp.maximum(visible_count - 1, 1)
                other_visible_sources = (
                    visible_sources[..., None, :] & off_diagonal_bool
                )
                truth_source_scores = (
                    -jnp.sum(pair_scores * other_visible_sources, axis=-1)
                    / truth_denominator
                )
                truth_source_scores = numpyro.deterministic(
                    f"~truth_source_scores_{suffix}",
                    truth_source_scores * visible_sources,
                )

                safe_hidden_source = jnp.clip(hidden, 0, n_sources - 1)
                hidden_indices = jnp.broadcast_to(
                    safe_hidden_source[..., None, None], pair_scores.shape[:-1] + (1,)
                )
                probe_source_scores = numpyro.deterministic(
                    f"~probe_source_scores_{suffix}",
                    jnp.take_along_axis(pair_scores, hidden_indices, axis=-1)[..., 0]
                    * visible_sources,
                )

                log_visible_sources = jnp.log(visible_sources)
                truth_weights = numpyro.deterministic(
                    f"~truth_weights_{suffix}",
                    jax.nn.softmax(
                        truth_source_scores / beta_y + log_visible_sources, axis=-1
                    ),
                )
                probe_weights = numpyro.deterministic(
                    f"~probe_weights_{suffix}",
                    jax.nn.softmax(
                        probe_source_scores / beta_z + log_visible_sources, axis=-1
                    ),
                )

                truth_mean = numpyro.deterministic(
                    f"~truth_mean_angle_{suffix}", circular_mean(reports, truth_weights)
                )
                hidden_mean = numpyro.deterministic(
                    f"~hidden_mean_angle_{suffix}", circular_mean(reports, probe_weights)
                )

                is_probe = trial == HIDDEN_SOURCE_PROBE
                response_mean = numpyro.deterministic(
                    f"~response_mean_angle_{suffix}",
                    jnp.where(is_probe, hidden_mean, truth_mean),
                )
                response_obs_t = None
                if truth_estimates is not None and hidden_predictions is not None:
                    response_obs_t = jnp.where(
                        is_probe,
                        hidden_predictions[:, :, trial_index],
                        truth_estimates[:, :, trial_index],
                    )
                with numpyro.handlers.mask(mask=valid_trial):
                    numpyro.sample(
                        f"response_{suffix}",
                        dist.VonMises(response_mean, kappa),
                        obs=response_obs_t,
                    )
