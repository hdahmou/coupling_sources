"""Single active NumPyro model for learned coupling between sources.

The model is the preregistered Phase-2 learning model, written one
observe -> update -> respond step per trial. Every recorded site carries the
symbol used in the preregistration TeX:

    learningRate, g_y, g_z, vmConcentration   pooled parameters (sample sites)
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
  matrices with a zero diagonal. The preregistered neutral origin
  c_{ij,p,0} = 0 is a constant, not a site.
* Each trial is either a truth-estimation trial (observed at ``y_t*``) or a
  hidden-source-estimation trial (observed at ``z_t*``); the branch that does
  not apply, and invalid trials, are masked out of the likelihood.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from coupling_sources.core.constants import HIDDEN_SOURCE_PROBE
from coupling_sources.core.data import SimulationBatch, StimulusBatch
from coupling_sources.core.priors import CouplingPriors
from coupling_sources.utils.angles import circular_mean


def _split_batch(
    batch: StimulusBatch | SimulationBatch,
) -> tuple[StimulusBatch, jnp.ndarray | None, jnp.ndarray | None]:
    """Return ``(stimuli, truth_estimates, hidden_predictions)``.

    The response arrays are ``None`` when the batch carries stimuli only, in
    which case the model runs in prior-predictive mode.
    """

    if isinstance(batch, SimulationBatch):
        return batch.stimuli, batch.truth_estimates, batch.hidden_predictions
    return batch, None, None


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

    return jax.nn.softmax(gain * scores + jnp.log(visible_sources), axis=-1)


def model_coupling(batch: StimulusBatch | SimulationBatch) -> None:
    stimuli, truth_estimates, hidden_predictions = _split_batch(batch)
    n_participants, n_rounds, n_trials, n_sources = stimuli.visible_reports.shape
    off_diagonal = ~jnp.eye(n_sources, dtype=bool)

    priors = CouplingPriors.distributions()
    learningRate = numpyro.sample("learningRate", priors["learningRate"])
    g_y = numpyro.sample("g_y", priors["g_y"])
    g_z = numpyro.sample("g_z", priors["g_z"])
    vmConcentration = numpyro.sample("vmConcentration", priors["vmConcentration"])

    participants = numpyro.plate("participants", n_participants, dim=-2)
    rounds = numpyro.plate("rounds", n_rounds, dim=-1)
    with participants, rounds:
        # c_{ij,p,0} = 0: every pair starts at the preregistered neutral origin.
        pair_scores = jnp.zeros((n_participants, n_rounds, n_sources, n_sources))

        for trial_index in range(n_trials):
            t = f"t{trial_index + 1}"
            reports_raw = stimuli.visible_reports[:, :, trial_index, :]
            hidden_source = stimuli.hidden_source[:, :, trial_index]
            valid_trial = stimuli.valid_trials[:, :, trial_index]
            is_probe = stimuli.trial_type[:, :, trial_index] == HIDDEN_SOURCE_PROBE

            # Hidden sources report NaN; visibility is read off the reports.
            visible_sources = jnp.isfinite(reports_raw)
            reports = jnp.nan_to_num(reports_raw)
            visible_pairs = (
                visible_sources[..., :, None]
                & visible_sources[..., None, :]
                & off_diagonal
            )

            # Observe: relative agreement a of the visible pairs.
            a = numpyro.deterministic(
                f"~a_{t}", _relative_agreement(reports, visible_pairs)
            )

            # Update: c <- c + learningRate (a - c) on visible pairs of valid trials;
            # all other pairs are carried forward unchanged.
            update_pairs = visible_pairs & valid_trial[..., None, None]
            pair_scores = numpyro.deterministic(
                f"~c_{t}", pair_scores + update_pairs * learningRate * (a - pair_scores)
            )

            # Respond, truth-estimation branch: redundancy discounting.
            η_y = numpyro.deterministic(
                f"~η^y_{t}", _truth_scores(pair_scores, visible_sources, off_diagonal)
            )
            w = numpyro.deterministic(
                f"~w_{t}", _visible_softmax(η_y, g_y, visible_sources)
            )
            μ = numpyro.deterministic(f"~μ_{t}", circular_mean(reports, w))

            # Respond, hidden-source branch: relatedness boosting.
            η_z = numpyro.deterministic(
                f"~η^z_{t}", _probe_scores(pair_scores, hidden_source, visible_sources)
            )
            v = numpyro.deterministic(
                f"~v_{t}", _visible_softmax(η_z, g_z, visible_sources)
            )
            ν = numpyro.deterministic(f"~ν_{t}", circular_mean(reports, v))

            # Likelihood: each valid trial contributes to exactly one branch.
            y_mask = valid_trial & ~is_probe
            z_mask = valid_trial & is_probe
            y_obs = (
                None
                if truth_estimates is None
                else jnp.where(y_mask, truth_estimates[:, :, trial_index], 0.0)
            )
            z_obs = (
                None
                if hidden_predictions is None
                else jnp.where(z_mask, hidden_predictions[:, :, trial_index], 0.0)
            )
            with numpyro.handlers.mask(mask=y_mask):
                numpyro.sample(f"y_{t}", dist.VonMises(μ, vmConcentration), obs=y_obs)
            with numpyro.handlers.mask(mask=z_mask):
                numpyro.sample(f"z_{t}", dist.VonMises(ν, vmConcentration), obs=z_obs)