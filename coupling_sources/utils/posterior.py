from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def extract_parameter_samples(samples: dict[str, Any]) -> dict[str, np.ndarray]:
    from coupling_sources.model import PARAMETER_SITES

    return {
        parameter: np.asarray(samples[site])
        for parameter, site in PARAMETER_SITES.items()
    }


def posterior_frame(samples: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(extract_parameter_samples(samples))


def parameter_values(params: Any) -> pd.Series:
    return pd.Series({
        "alpha": float(np.asarray(params.alpha)),
        "gamma": float(np.asarray(params.gamma)),
        "kappa_p": float(np.asarray(params.kappa_p)),
        "kappa_a": float(np.asarray(params.kappa_a)),
    })


def summarize_posterior(frame: pd.DataFrame, true_values: pd.Series) -> pd.DataFrame:
    summary = frame.quantile([0.05, 0.50, 0.95]).T
    summary.columns = ["q05", "median", "q95"]
    summary.insert(0, "true", true_values)
    summary.insert(1, "mean", frame.mean())
    summary.insert(2, "sd", frame.std())
    return summary
