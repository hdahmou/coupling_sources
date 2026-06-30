import jax
import jax.numpy as jnp


def sample_vonmises(key: jax.Array, mu: jax.Array, kappa: jax.Array) -> jax.Array:
    def uniform(k: jax.Array) -> jax.Array:
        return jax.random.uniform(k, (), minval=0.0, maxval=2 * jnp.pi)

    def best_fisher(k: jax.Array) -> jax.Array:
        a = 1.0 + jnp.sqrt(1.0 + 4.0 * kappa**2)
        b = (a - jnp.sqrt(2.0 * a)) / (2.0 * kappa)
        r = (1.0 + b**2) / (2.0 * b)

        def cond(state: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
            return ~state[2]

        def body(state: tuple[jax.Array, jax.Array, jax.Array]):
            k, _, _ = state
            k, subkey = jax.random.split(k)
            u = jax.random.uniform(subkey, (3,))
            z = jnp.cos(jnp.pi * u[0])
            f = (1.0 + r * z) / (r + z)
            c = kappa * (r - f)
            accepted = (u[1] < c * (2.0 - c)) | (jnp.log(c / u[1]) + 1.0 - c >= 0.0)
            angle = jnp.where(u[2] < 0.5, -1.0, 1.0) * jnp.arccos(f)
            return k, angle, accepted

        _, angle, _ = jax.lax.while_loop(cond, body, (k, 0.0, False))
        return (mu + angle) % (2 * jnp.pi)

    return jax.lax.cond(kappa < 1e-6, uniform, best_fisher, key)
