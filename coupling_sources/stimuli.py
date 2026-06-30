from dataclasses import dataclass

import jax
import jax.numpy as jnp

from coupling_sources.utils.shapes import source_round_to_round_source


@dataclass(frozen=True)
class StimulusBatch:
    truth: jax.Array
    source_choices: jax.Array
    hidden_source: jax.Array
    complete_source_choices: jax.Array
    covariance: jax.Array


def stimulus_covariance(
    n_sources: int, sigma: float | jax.Array, lam: float | jax.Array, coupled: tuple[int, int]
) -> jax.Array:
    corr = jnp.eye(n_sources)
    i, j = coupled
    corr = corr.at[i, j].set(lam).at[j, i].set(lam)
    return sigma**2 * corr


def hide_random_source_round(
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


def hide_random_sources(
    key: jax.Array,
    source_choices: jax.Array,
    n_trials_full: int,
    n_trials_partial: int,
) -> tuple[jax.Array, jax.Array]:
    choices_by_round = source_round_to_round_source(source_choices)
    keys = jax.random.split(key, choices_by_round.shape[0])
    masked_by_round, hidden_source = jax.vmap(
        lambda subkey, choices_i: hide_random_source_round(
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
    cov = stimulus_covariance(n_sources, sigma, lam, coupled)
    cov_root = jnp.linalg.cholesky(cov)
    truth = jax.random.uniform(key_theta, (n_rounds, n_trials), minval=0.0, maxval=2 * jnp.pi)
    error = jax.random.normal(key_error, (n_rounds, n_trials, n_sources)) @ cov_root.T
    reports = (truth[..., None] + error) % (2 * jnp.pi)
    complete_source_choices = jnp.moveaxis(reports, -1, 0)
    source_choices, hidden_source = hide_random_sources(
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
