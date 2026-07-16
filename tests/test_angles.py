import numpy as np
import jax.numpy as jnp

from coupling_sources.utils.angles import mean_angle_difference


def test_mean_angle_difference_matches_manual_circular_means() -> None:
    source_choices = np.asarray(
        [
            [0.0, 0.3, 1.2, 2.1],
            [5.9, 0.2, 0.7, 1.4],
        ],
        dtype=np.float32,
    )
    weights = np.asarray([0.12, 0.12, 0.38, 0.38], dtype=np.float32)
    equal_weights = np.full(source_choices.shape[-1], 1.0 / source_choices.shape[-1])

    equal_mean = _manual_circular_mean(source_choices, equal_weights)
    weighted_mean = _manual_circular_mean(source_choices, weights)
    expected = np.abs(np.angle(np.exp(1j * (equal_mean - weighted_mean))))

    np.testing.assert_allclose(
        np.asarray(mean_angle_difference(jnp.asarray(source_choices), jnp.asarray(weights))),
        expected,
        rtol=1e-6,
    )


def _manual_circular_mean(angles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.sum(weights * np.cos(angles), axis=-1)
    y = np.sum(weights * np.sin(angles), axis=-1)
    return np.mod(np.arctan2(y, x), 2.0 * np.pi)
