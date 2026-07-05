import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg


def circular_mean(angles: jax.Array, weights: jax.Array) -> jax.Array:
    x = jnp.sum(weights * jnp.cos(angles), axis=-1)
    y = jnp.sum(weights * jnp.sin(angles), axis=-1)
    return jnp.arctan2(y, x) % (2 * jnp.pi)


def centered_angle(x: jax.Array) -> jax.Array:
    return jnp.angle(jnp.exp(1j * x))


def circular_difference_deg(a: jax.Array, b: jax.Array) -> jax.Array:
    return (a - b + 180) % 360 - 180


def optimal_weights(cov: jax.Array) -> jax.Array:
    raw = jsp_linalg.solve(cov, jnp.ones(cov.shape[0]), assume_a="sym")
    return raw / raw.sum()
