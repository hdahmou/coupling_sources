"""Single active NumPyro model for learned coupling between sources.

The model is the preregistered Phase-2 learning model, written one
observe -> update -> respond step per trial. Every recorded site carries the
symbol used in the preregistration TeX:

    learningRate, gamma_y, gamma_z, vmConcentration
                       pooled parameters (sample sites)
    ~g_y, ~g_z       gains derived in ideal-observer benchmark units
    ~a_t*            a_{ij,t}: centred relative-agreement signal
    ~c_t*            c_{ij,p,t}: pair scores after the trial-t update
    ~η^y_t*, ~η^z_t* source-level response scores (redundancy / relatedness)
    ~w_t*, ~v_t*     softmax weights with gains g_y and g_z
    ~μ_t*, ~ν_t*     weighted circular means of the visible reports
    y_t*, z_t*       observed responses, y ~ VonMises(μ, vmConcentration), z ~ VonMises(ν, vmConcentration)

Conventions
-----------
* Deterministic bookkeeping sites are prefixed with ``~`` so they can be
  discarded from MCMC traces; the suffix ``t3`` is the trial number.
* Pairwise quantities (``~a_t*``, ``~c_t*``) are symmetric ``(..., S, S)``
  matrices with a zero diagonal. The initial off-diagonal score
  c_{ij,p,0} = c0 is a fixed model input recorded at ``~$c_0$``.
* Each trial is either a truth-estimation trial (observed at ``y_t*``) or a
  hidden-source-estimation trial (observed at ``z_t*``); the branch that does
  not apply, and invalid trials, are masked out of the likelihood.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from coupling_sources.core.constants import HIDDEN_SOURCE_PROBE, TRUTH_ESTIMATION
from coupling_sources.core.data import GainConstants, SimulationBatch
from coupling_sources.core.priors import CouplingPriors
from coupling_sources.utils.angles import centered_angle, circular_mean


def _relative_agreement(
    reports: jnp.ndarray, visible_pairs: jnp.ndarray
) -> jnp.ndarray:
    """a_{ij,t}: half of each visible pair's deviation from the mean agreement.

    Pairwise cosine agreement is centred on the mean over the visible pairs of
    the same trial, so each trial supplies only a contrast among visible
    pairs. Entries outside ``visible_pairs`` are zero.
    """

    pair_agreement = jnp.cos(reports[..., :, None] - reports[..., None, :])
    pair_count = jnp.maximum(jnp.sum(visible_pairs, axis=(-2, -1), keepdims=True), 1)
    mean_agreement = (
        jnp.sum(pair_agreement * visible_pairs, axis=(-2, -1), keepdims=True)
        / pair_count
    )
    return 0.5 * (pair_agreement - mean_agreement) * visible_pairs


def _truth_scores(
    pair_scores: jnp.ndarray,
    visible_sources: jnp.ndarray,
    off_diagonal: jnp.ndarray,
) -> jnp.ndarray:
    """η^y_i: negative mean coupling of source i with the other visible sources.

    The minus sign implements redundancy discounting; averaging keeps the
    score on the scale of a single pair score. Invisible sources score zero.
    """

    other_visible = visible_sources[..., None, :] & off_diagonal
    n_other = jnp.maximum(jnp.sum(visible_sources, axis=-1, keepdims=True) - 1, 1)
    return -jnp.sum(pair_scores * other_visible, axis=-1) / n_other * visible_sources


def _probe_scores(
    pair_scores: jnp.ndarray,
    hidden_source: jnp.ndarray,
    visible_sources: jnp.ndarray,
) -> jnp.ndarray:
    """η^z_i: coupling of visible source i with the hidden source h_t.

    The positive sign implements relatedness boosting. ``hidden_source`` is
    clipped so that truth-estimation trials (no hidden source) stay in range;
    their η^z is never used because those trials are masked out of ``z``.
    """

    n_sources = pair_scores.shape[-1]
    safe_hidden = jnp.clip(hidden_source, 0, n_sources - 1)
    hidden_index = jnp.broadcast_to(
        safe_hidden[..., None, None], pair_scores.shape[:-1] + (1,)
    )
    coupling_with_hidden = jnp.take_along_axis(pair_scores, hidden_index, axis=-1)
    return coupling_with_hidden[..., 0] * visible_sources


def _visible_softmax(
    scores: jnp.ndarray, gain: jnp.ndarray, visible_sources: jnp.ndarray
) -> jnp.ndarray:
    """Softmax over the visible sources only.

    ``log(visible)`` adds -inf to the logits of invisible sources, so their
    weight is exactly zero and the remaining weights sum to one.
    """

    no_visible_sources = ~jnp.any(visible_sources, axis=-1, keepdims=True)
    fallback_source = jnp.arange(visible_sources.shape[-1]) == 0
    safe_visible_sources = visible_sources | (no_visible_sources & fallback_source)
    return jax.nn.softmax(
        gain * scores + jnp.log(safe_visible_sources),
        axis=-1,
    )


def model_coupling(
    batch: SimulationBatch,
    c0: float | jax.Array,
    gains: GainConstants,
) -> None:
    stimuli = batch.stimuli
    visible_reports = jnp.asarray(stimuli.visible_reports, dtype=jnp.float64)
    hidden_sources = jnp.asarray(stimuli.hidden_source, dtype=jnp.int32)
    trial_types = jnp.asarray(stimuli.trial_type, dtype=jnp.int32)
    valid_trials = jnp.asarray(stimuli.valid_trials, dtype=bool)
    n_participants, n_rounds, n_trials, n_sources = visible_reports.shape
    off_diagonal = ~jnp.eye(n_sources, dtype=bool)

    priors = CouplingPriors.distributions()
    learningRate = jnp.asarray(numpyro.sample("learningRate", priors["learningRate"]))
    gamma_y = jnp.asarray(numpyro.sample("gamma_y", priors["gamma_y"]))
    gamma_z = jnp.asarray(numpyro.sample("gamma_z", priors["gamma_z"]))
    g_y = jnp.asarray(numpyro.deterministic("~$g_y$", gamma_y * gains.g_star_y))
    g_z = jnp.asarray(numpyro.deterministic("~$g_z$", gamma_z * gains.g_star_z))
    vmConcentration = jnp.asarray(
        numpyro.sample("vmConcentration", priors["vmConcentration"])
    )
    initial_score = jnp.asarray(
        numpyro.deterministic(
            "~$c_0$",
            jnp.asarray(c0, dtype=visible_reports.dtype),
        )
    )

    participants = numpyro.plate("participants", n_participants, dim=-2)
    rounds = numpyro.plate("rounds", n_rounds, dim=-1)
    with participants, rounds:
        pair_scores = jnp.broadcast_to(
            initial_score * off_diagonal,
            (n_participants, n_rounds, n_sources, n_sources),
        )

        for trial_index in range(n_trials):
            reports_raw = visible_reports[:, :, trial_index, :]
            hidden_source = hidden_sources[:, :, trial_index]
            valid_trial = valid_trials[:, :, trial_index]
            trial_type = trial_types[:, :, trial_index]
            is_truth = trial_type == TRUTH_ESTIMATION
            is_probe = trial_type == HIDDEN_SOURCE_PROBE

            # Hidden sources report NaN; visibility is read off the reports.
            visible_sources = jnp.isfinite(reports_raw)
            reports = jnp.nan_to_num(reports_raw)
            visible_pairs = (
                visible_sources[..., :, None]
                & visible_sources[..., None, :]
                & off_diagonal
            )

            # Observe: relative agreement a of the visible pairs.
            a = jnp.asarray(
                numpyro.deterministic(
                    f"~$a_{{t={trial_index}}}$",
                    _relative_agreement(reports, visible_pairs),
                )
            )

            # Update: c <- c + learningRate (a - c) on visible pairs of valid trials;
            # all other pairs are carried forward unchanged.
            update_pairs = visible_pairs & valid_trial[..., None, None]
            pair_scores = jnp.asarray(
                numpyro.deterministic(
                    f"~$c_{{t={trial_index}}}$",
                    pair_scores + update_pairs * learningRate * (a - pair_scores),
                )
            )

            # Respond, truth-estimation branch: redundancy discounting.
            η_y = jnp.asarray(
                numpyro.deterministic(
                    f"~$η^y_{{t={trial_index}}}$",
                    _truth_scores(pair_scores, visible_sources, off_diagonal),
                )
            )
            w = jnp.asarray(
                numpyro.deterministic(
                    f"~$w_{{t={trial_index}}}$",
                    _visible_softmax(η_y, g_y, visible_sources),
                )
            )
            μ = jnp.asarray(
                numpyro.deterministic(
                    f"~$μ_{{t={trial_index}}}$",
                    circular_mean(reports, w),
                )
            )

            # Respond, hidden-source branch: relatedness boosting.
            η_z = jnp.asarray(
                numpyro.deterministic(
                    f"~$η^z_{{t={trial_index}}}$",
                    _probe_scores(pair_scores, hidden_source, visible_sources),
                )
            )
            v = jnp.asarray(
                numpyro.deterministic(
                    f"~$v_{{t={trial_index}}}$",
                    _visible_softmax(η_z, g_z, visible_sources),
                )
            )
            ν = jnp.asarray(
                numpyro.deterministic(
                    f"~$ν_{{t={trial_index}}}$",
                    circular_mean(reports, v),
                )
            )

            # Likelihood: each valid trial contributes to exactly one branch.
            y_mask = valid_trial & is_truth
            z_mask = valid_trial & is_probe
            y_obs = centered_angle(
                jnp.where(y_mask, batch.truth_estimates[:, :, trial_index], 0.0)
            )
            z_obs = centered_angle(
                jnp.where(z_mask, batch.hidden_predictions[:, :, trial_index], 0.0)
            )
            with numpyro.handlers.mask(mask=y_mask):
                numpyro.sample(
                    f"$y_{{t={trial_index}}}$",
                    dist.VonMises(μ, vmConcentration),
                    obs=y_obs,
                )
            with numpyro.handlers.mask(mask=z_mask):
                numpyro.sample(
                    f"$z_{{t={trial_index}}}$",
                    dist.VonMises(ν, vmConcentration),
                    obs=z_obs,
                )
