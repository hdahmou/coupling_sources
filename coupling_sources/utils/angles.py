import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg


def circular_mean(angles: jax.Array, weights: jax.Array) -> jax.Array:
    x = jnp.sum(weights * jnp.cos(angles), axis=-1)
    y = jnp.sum(weights * jnp.sin(angles), axis=-1)
    return jnp.arctan2(y, x) % (2 * jnp.pi)


def centered_angle(x: jax.Array) -> jax.Array:
    return jnp.angle(jnp.exp(1j * x))


def mean_angle_difference(source_choices: jax.Array, weights: jax.Array) -> jax.Array:
    n_sources = source_choices.shape[-1]
    equal_weights = jnp.full(
        (n_sources,),
        1.0 / n_sources,
        dtype=source_choices.dtype,
    )
    equal_mean = circular_mean(source_choices, equal_weights)
    weighted_mean = circular_mean(source_choices, weights)
    return jnp.abs(centered_angle(equal_mean - weighted_mean))


def optimal_weights(cov: jax.Array) -> jax.Array:
    raw = jsp_linalg.solve(cov, jnp.ones(cov.shape[0]), assume_a="sym")
    return raw / raw.sum()
