from __future__ import annotations

import numpy as np
import numpyro
import pytest

from coupling_sources.core.constants import HIDDEN_SOURCE_PROBE, TRUTH_ESTIMATION
from coupling_sources.core.data import (
    GainConstants,
    MCMCSpecs,
    SimulationBatch,
    StimulusBatch,
    StimulusMetadata,
)
from coupling_sources.core.inference import fit_simulation
from coupling_sources.core.model import model_coupling
from coupling_sources.core.priors import CouplingPriors
from coupling_sources.core.simulator import simulate_coupling


def test_prior_parameter_sites_match_model_site_names() -> None:
    assert CouplingPriors.parameter_sites() == (
        "learningRate",
        "gamma_y",
        "gamma_z",
        "vmConcentration",
    )
    assert tuple(CouplingPriors.distributions()) == CouplingPriors.parameter_sites()
    priors = CouplingPriors.distributions()
    for name in ("gamma_y", "gamma_z"):
        assert float(priors[name].mean) == 0.0
        assert float(priors[name].variance) == 1.0


@pytest.mark.parametrize(
    "values",
    [
        {"g_star_y": 0.0, "g_star_z": 1.0},
        {"g_star_y": 1.0, "g_star_z": -1.0},
        {"g_star_y": np.nan, "g_star_z": 1.0},
        {"g_star_y": 1.0, "g_star_z": np.inf},
    ],
)
def test_gain_constants_must_be_finite_and_positive(values: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="g_star"):
        GainConstants(**values)


def test_model_uses_prior_parameter_site_names() -> None:
    stimuli = StimulusBatch(
        visible_reports=np.array([[[[0.1, 0.2], [0.3, np.nan]]]]),
        hidden_source=np.array([[[-1, 1]]]),
        trial_type=np.array([[[TRUTH_ESTIMATION, HIDDEN_SOURCE_PROBE]]]),
        valid_trials=np.array([[[True, True]]]),
        metadata=StimulusMetadata(
            complete_reports=np.array([[[[0.1, 0.2], [0.3, 0.4]]]]),
            true_direction=np.array([[[0.15, 0.35]]]),
            source_choices=np.array([0, 1]),
            stimulus_covariance=np.eye(2),
            resultant_length_full=np.ones((1, 1, 2)),
            resultant_length_leave_one_out=np.ones((1, 1, 2, 2)),
        ),
    )
    simulation = SimulationBatch(
        stimuli=stimuli,
        truth_estimates=np.array([[[0.15, np.nan]]]),
        hidden_predictions=np.array([[[np.nan, 0.4]]]),
        responses=np.array([[[0.15, 0.4]]]),
    )

    gains = GainConstants(g_star_y=19.946, g_star_z=28.121)
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model_coupling, rng_seed=0)
    ).get_trace(simulation, 0.25, gains)
    sample_sites = {name for name, site in trace.items() if site["type"] == "sample"}

    parameter_sites = set(CouplingPriors.parameter_sites())
    assert parameter_sites <= sample_sites
    assert {
        name for name in sample_sites if name in CouplingPriors.distributions()
    } == parameter_sites
    np.testing.assert_allclose(
        trace["~$g_y$"]["value"],
        trace["gamma_y"]["value"] * gains.g_star_y,
    )
    np.testing.assert_allclose(
        trace["~$g_z$"]["value"],
        trace["gamma_z"]["value"] * gains.g_star_z,
    )
    assert trace["~$c_0$"]["value"] == 0.25
    first_pair_scores = trace["~$c_{t=0}$"]["value"]
    assert 0.0 < first_pair_scores[0, 0, 0, 1] < 0.25
    assert first_pair_scores[0, 0, 0, 0] == 0.0

    graph_source = numpyro.render_model(
        model_coupling,
        model_args=(simulation, 0.25, gains),
        render_distributions=False,
    ).source
    assert '"~$c_0$" -> "~$c_{t=0}$"' in graph_source
    assert '"~$c_{t=0}$" -> "~$η^y_{t=0}$"' in graph_source
    assert 'learningRate -> "~$η^y_{t=0}$"' not in graph_source


def test_simulator_converts_gamma_to_effective_gains() -> None:
    stimuli = StimulusBatch(
        visible_reports=np.array([[[[0.1, 0.2]]]]),
        hidden_source=np.array([[[-1]]]),
        trial_type=np.array([[[TRUTH_ESTIMATION]]]),
        valid_trials=np.array([[[True]]]),
        metadata=StimulusMetadata(
            complete_reports=np.array([[[[0.1, 0.2]]]]),
            true_direction=np.array([[[0.15]]]),
            source_choices=np.array([0, 1]),
            stimulus_covariance=np.eye(2),
            resultant_length_full=np.ones((1, 1, 1)),
            resultant_length_leave_one_out=np.ones((1, 1, 1, 2)),
        ),
    )
    gains = GainConstants(g_star_y=19.946, g_star_z=28.121)
    simulation = simulate_coupling(
        seed=0,
        batch=stimuli,
        parameters={
            "learningRate": 0.2,
            "gamma_y": 0.5,
            "gamma_z": -1.0,
            "vmConcentration": 8.0,
            "c0": 0.0,
        },
        gains=gains,
    )

    assert simulation.participant_parameters is not None
    assert float(simulation.participant_parameters["g_y"]) == pytest.approx(
        0.5 * gains.g_star_y
    )
    assert float(simulation.participant_parameters["g_z"]) == pytest.approx(
        -gains.g_star_z
    )

    fit = fit_simulation(
        simulation=simulation,
        c0=0.0,
        gains=gains,
        mcmc=MCMCSpecs(
            seed=1,
            num_warmup=2,
            num_samples=2,
            num_chains=1,
            chain_method="sequential",
            target_accept_prob=0.8,
            max_tree_depth=3,
        ),
        progress_bar=False,
    )
    assert tuple(fit["samples"]) == CouplingPriors.parameter_sites()


def test_sample_parameter_draws_uses_model_site_names() -> None:
    samples = CouplingPriors.sample_parameter_draws(seed=1, draws=3, c0=0.25)

    assert tuple(samples) == (
        "learningRate",
        "gamma_y",
        "gamma_z",
        "vmConcentration",
        "c0",
    )
    assert samples["learningRate"].shape == (3,)
    np.testing.assert_array_equal(samples["c0"], np.full(3, 0.25))


def test_normalize_parameters_uses_model_site_names() -> None:
    params = CouplingPriors.normalize_parameters(
        {
            "learningRate": 0.2,
            "gamma_y": -1.0,
            "gamma_z": 2.0,
            "vmConcentration": 8.0,
            "c0": 0.25,
        }
    )

    assert set(params) == {
        "learningRate",
        "gamma_y",
        "gamma_z",
        "vmConcentration",
        "c0",
    }
    np.testing.assert_allclose(
        [
            params["learningRate"],
            params["gamma_y"],
            params["gamma_z"],
            params["vmConcentration"],
            params["c0"],
        ],
        [0.2, -1.0, 2.0, 8.0, 0.25],
    )


def test_normalize_parameters_requires_c0() -> None:
    with pytest.raises(KeyError, match="c0"):
        CouplingPriors.normalize_parameters(
            {
                "learningRate": 0.2,
                "gamma_y": 1.0,
                "gamma_z": 2.0,
                "vmConcentration": 8.0,
            }
        )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("learningRate", 0.0, "strictly between"),
        ("learningRate", 1.0, "strictly between"),
        ("learningRate", np.array([0.2]), "scalar"),
        ("gamma_y", np.nan, "finite"),
        ("gamma_z", np.inf, "finite"),
        ("vmConcentration", 0.0, "positive"),
        ("c0", np.nan, "finite"),
    ],
)
def test_normalize_parameters_rejects_invalid_simulation_values(
    name: str,
    value: object,
    message: str,
) -> None:
    parameters: dict[str, object] = {
        "learningRate": 0.2,
        "gamma_y": -1.0,
        "gamma_z": 2.0,
        "vmConcentration": 8.0,
        "c0": 0.25,
    }
    parameters[name] = value

    with pytest.raises(ValueError, match=message):
        CouplingPriors.normalize_parameters(parameters)
