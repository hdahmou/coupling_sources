from __future__ import annotations

import numpy as np
import pytest

from coupling_sources.analysis import bf10


def test_bf10_uses_zero_ordinate_and_configurable_prior_sd() -> None:
    draws = np.linspace(-2.0, 2.0, 101)

    unit_prior = bf10(draws)
    narrow_prior = bf10(draws, prior_sd=0.5)

    assert unit_prior > 0.0
    assert narrow_prior == pytest.approx(2.0 * unit_prior)
