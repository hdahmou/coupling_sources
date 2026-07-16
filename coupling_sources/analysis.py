from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde, norm


def bf10(draws: np.ndarray, prior_sd: float = 1.0) -> float:
    """Return the Savage-Dickey BF10 for a zero-valued point null."""
    posterior_at_zero = gaussian_kde(np.asarray(draws).ravel())(0.0)[0]
    prior_at_zero = norm.pdf(0.0, scale=prior_sd)
    return float(prior_at_zero / posterior_at_zero)
