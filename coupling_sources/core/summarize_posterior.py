from __future__ import annotations

import numpy as np
import pandas as pd

from coupling_sources.core.parameters.simulation import SimParams


def parameter_values(params: SimParams) -> pd.Series:
    names = params.names()
    return pd.Series(
        [float(np.asarray(getattr(params, name))) for name in names],
        index=names,
    )


def summarize_posterior(frame: pd.DataFrame, true_values: pd.Series) -> pd.DataFrame:
    summary = frame.quantile([0.05, 0.50, 0.95]).T
    summary.columns = ["q05", "median", "q95"]
    summary.insert(0, "true", true_values)
    summary.insert(1, "mean", frame.mean())
    summary.insert(2, "sd", frame.std())
    return summary
