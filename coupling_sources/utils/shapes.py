import jax
import jax.numpy as jnp


def as_source_round_choices(source_choices: jax.Array) -> jax.Array:
    source_choices = jnp.asarray(source_choices)
    if source_choices.ndim == 2:
        return source_choices[:, None, :]
    if source_choices.ndim != 3:
        raise ValueError("source_choices must have shape (source, time) or (source, round, time)")
    return source_choices


def as_round_hidden_source(hidden_source: jax.Array) -> jax.Array:
    hidden_source = jnp.asarray(hidden_source)
    if hidden_source.ndim == 1:
        return hidden_source[None, :]
    if hidden_source.ndim != 2:
        raise ValueError("hidden_source must have shape (time,) or (round, time)")
    return hidden_source


def as_round_choices(choices: jax.Array | None) -> jax.Array | None:
    if choices is None:
        return None
    choices = jnp.asarray(choices)
    if choices.ndim == 1:
        return choices[None, :]
    if choices.ndim != 2:
        raise ValueError("choices must have shape (time,) or (round, time)")
    return choices


def as_source_round_data(
    source_choices: jax.Array,
    hidden_source: jax.Array,
    choices: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    return (
        as_source_round_choices(source_choices),
        as_round_hidden_source(hidden_source),
        as_round_choices(choices),
    )


def source_round_to_round_source(source_choices: jax.Array) -> jax.Array:
    return jnp.swapaxes(as_source_round_choices(source_choices), 0, 1)


def source_round_to_round_time_source(source_choices: jax.Array) -> jax.Array:
    return jnp.moveaxis(as_source_round_choices(source_choices), 0, -1)
