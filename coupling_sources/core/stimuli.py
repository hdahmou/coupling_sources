from dataclasses import dataclass

import jax
import jax.numpy as jnp

from coupling_sources.core.parameters.stimuli import StimuliParams


@dataclass(frozen=True)
class StimulusBatch:
    truth: jax.Array
    source_choices: jax.Array
    hidden_source: jax.Array
    complete_source_choices: jax.Array
    covariance: jax.Array


def _stimulus_covariance(
    n_sources: int, sigma: float | jax.Array, lam: float | jax.Array, coupled: tuple[int, int]
) -> jax.Array:
    corr = jnp.eye(n_sources)
    i, j = coupled
    corr = corr.at[i, j].set(lam).at[j, i].set(lam)
    return sigma**2 * corr


def _hide_random_source_round(
    key: jax.Array,
    source_choices: jax.Array,
    n_trials_full: int,
    n_trials_partial: int,
) -> tuple[jax.Array, jax.Array]:
    n_sources, n_trials = source_choices.shape
    if n_trials != n_trials_full + n_trials_partial:
        raise ValueError(
            f"source_choices has {n_trials} trials, expected "
            f"{n_trials_full} + {n_trials_partial}"
        )

    key_trial, key_source = jax.random.split(key)
    is_hidden = jax.random.permutation(key_trial, jnp.arange(n_trials) >= n_trials_full)
    hidden_idx = jax.random.randint(key_source, (n_trials,), 0, n_sources)
    trial_idx = jnp.arange(n_trials)
    hidden_source = jnp.where(is_hidden, hidden_idx.astype(float), jnp.nan)
    hidden_value = jnp.where(is_hidden, jnp.nan, source_choices[hidden_idx, trial_idx])
    return source_choices.at[hidden_idx, trial_idx].set(hidden_value), hidden_source


def _hide_random_sources(
    key: jax.Array,
    source_choices: jax.Array,
    n_trials_full: int,
    n_trials_partial: int,
) -> tuple[jax.Array, jax.Array]:
    source_choices = jnp.asarray(source_choices)
    if source_choices.ndim != 3:
        raise ValueError("source_choices must have shape (source, round, time)")
    choices_by_round = jnp.swapaxes(source_choices, 0, 1)
    keys = jax.random.split(key, choices_by_round.shape[0])
    masked_by_round, hidden_source = jax.vmap(
        lambda subkey, choices_i: _hide_random_source_round(
            subkey, choices_i, n_trials_full, n_trials_partial
        )
    )(keys, choices_by_round)
    return jnp.swapaxes(masked_by_round, 0, 1), hidden_source


def generate_stimuli(
    key: jax.Array,
    n_rounds: int,
    n_trials_full: int,
    n_trials_partial: int,
    n_sources: int = 4,
    sigma: float | jax.Array = 0.55,
    lam: float | jax.Array = 0.80,
    coupled: tuple[int, int] = (0, 1),
) -> StimulusBatch:
    if n_trials_full < 0 or n_trials_partial < 0:
        raise ValueError("trial counts must be non-negative")

    n_trials = n_trials_full + n_trials_partial
    key_theta, key_error, key_hide = jax.random.split(key, 3)
    cov = _stimulus_covariance(n_sources, sigma, lam, coupled)
    cov_root = jnp.linalg.cholesky(cov)
    truth = jax.random.uniform(key_theta, (n_rounds, n_trials), minval=0.0, maxval=2 * jnp.pi)
    error = jax.random.normal(key_error, (n_rounds, n_trials, n_sources)) @ cov_root.T
    reports = (truth[..., None] + error) % (2 * jnp.pi)
    complete_source_choices = jnp.moveaxis(reports, -1, 0)
    source_choices, hidden_source = _hide_random_sources(
        key_hide,
        complete_source_choices,
        n_trials_full=n_trials_full,
        n_trials_partial=n_trials_partial,
    )
    return StimulusBatch(
        truth=truth,
        source_choices=source_choices,
        hidden_source=hidden_source,
        complete_source_choices=complete_source_choices,
        covariance=cov,
    )


def generate_stimulus(params: StimuliParams) -> StimulusBatch:
    return generate_stimuli(
        jax.random.PRNGKey(params.seed),
        n_rounds=params.n_rounds,
        n_trials_full=params.n_trials_full,
        n_trials_partial=params.n_trials_partial,
        n_sources=params.n_sources,
        sigma=params.sigma,
        lam=params.lam,
        coupled=params.coupled_sources,
    )
