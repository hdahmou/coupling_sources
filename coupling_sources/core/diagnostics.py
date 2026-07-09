from __future__ import annotations

import numpy as np
import numpyro.diagnostics as diagnostics

from coupling_sources.core.priors import CouplingPriors


def mcmc_diagnostics(mcmc: object) -> dict[str, object]:
    grouped_samples = mcmc.get_samples(group_by_chain=True)
    parameter_names = CouplingPriors.parameter_sites()
    try:
        parameter_summary = diagnostics.summary(
            {name: grouped_samples[name] for name in parameter_names},
            group_by_chain=True,
        )
        r_hat = {
            parameter: float(np.asarray(stats["r_hat"]))
            for parameter, stats in parameter_summary.items()
        }
        ess = {
            parameter: float(np.asarray(stats["n_eff"]))
            for parameter, stats in parameter_summary.items()
        }
    except AssertionError:
        r_hat = {name: float("nan") for name in parameter_names}
        ess = {name: float("nan") for name in parameter_names}
    extra_fields = mcmc.get_extra_fields(group_by_chain=True)
    divergences = extra_fields.get("diverging")
    n_divergences = None if divergences is None else int(np.asarray(divergences).sum())
    return {
        "r_hat": r_hat,
        "ess": ess,
        "has_divergences": None if n_divergences is None else bool(n_divergences),
        "n_divergences": n_divergences,
    }
