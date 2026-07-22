from __future__ import annotations

from dataclasses import replace

import numpy as np
import numpyro
import pytest
from numpyro.infer.util import log_density

from coupling_sources.core.constants import (
    HIDDEN_SOURCE_PROBE,
    NO_HIDDEN_SOURCE,
    TRUTH_ESTIMATION,
)
from coupling_sources.core.data import (
    GainConstants,
    SimulationBatch,
    StimulusBatch,
    StimulusMetadata,
    prepare_simulation_for_fit,
)
from coupling_sources.core.model import model_coupling
from coupling_sources.core.simulator import simulate_coupling


def _metadata(complete_reports: np.ndarray) -> StimulusMetadata:
    trial_shape = complete_reports.shape[:-1]
    n_sources = complete_reports.shape[-1]
    return StimulusMetadata(
        complete_reports=complete_reports,
        true_direction=np.zeros(trial_shape),
        source_choices=complete_reports,
        stimulus_covariance=np.eye(n_sources),
        resultant_length_full=np.ones(trial_shape),
        resultant_length_leave_one_out=np.ones((*trial_shape, n_sources)),
    )


def _parity_stimuli() -> StimulusBatch:
    n_participants, n_rounds, n_trials, n_sources = 2, 2, 4, 4
    complete_reports = np.empty(
        (n_participants, n_rounds, n_trials, n_sources),
        dtype=np.float64,
    )
    for participant in range(n_participants):
        for round_index in range(n_rounds):
            offset = 0.17 * participant + 0.09 * round_index
            complete_reports[participant, round_index] = (
                np.array(
                    [
                        [0.10, 0.45, 0.90, 1.30],
                        [0.25, 0.60, 1.05, 1.55],
                        [0.35, 0.70, 1.20, 1.75],
                        [0.20, 0.55, 1.10, 1.65],
                    ]
                )
                + offset
            )

    visible_reports = complete_reports.copy()
    visible_reports[:, :, 1, 1] = np.nan
    visible_reports[:, :, 2, :] = np.nan
    visible_reports[:, :, 3, 3] = np.nan
    trial_type = np.broadcast_to(
        np.array(
            [
                TRUTH_ESTIMATION,
                HIDDEN_SOURCE_PROBE,
                TRUTH_ESTIMATION,
                HIDDEN_SOURCE_PROBE,
            ]
        ),
        (n_participants, n_rounds, n_trials),
    ).copy()
    hidden_source = np.broadcast_to(
        np.array([NO_HIDDEN_SOURCE, 1, NO_HIDDEN_SOURCE, 3]),
        trial_type.shape,
    ).copy()
    valid_trials = np.broadcast_to(
        np.array([True, True, False, True]),
        trial_type.shape,
    ).copy()
    return StimulusBatch(
        visible_reports=visible_reports,
        hidden_source=hidden_source,
        trial_type=trial_type,
        valid_trials=valid_trials,
        metadata=_metadata(complete_reports),
    )


def _stack_trial_sites(
    trace: dict[str, dict[str, object]],
    template: str,
    n_trials: int,
) -> np.ndarray:
    return np.stack(
        [
            np.asarray(trace[template.replace("*", str(trial))]["value"])
            for trial in range(n_trials)
        ],
        axis=2,
    )


def test_model_deterministic_sites_match_simulator_diagnostics() -> None:
    stimuli = _parity_stimuli()
    gains = GainConstants(g_star_y=19.946, g_star_z=28.121)
    parameters = {
        "learningRate": 0.35,
        "gamma_y": -0.6,
        "gamma_z": 1.2,
        "vmConcentration": 7.0,
        "c0": 0.15,
    }
    simulation = simulate_coupling(
        seed=17,
        batch=stimuli,
        parameters=parameters,
        gains=gains,
    )
    fit_data = prepare_simulation_for_fit(simulation)
    conditioned_model = numpyro.handlers.substitute(
        model_coupling,
        data={name: parameters[name] for name in parameters if name != "c0"},
    )
    trace = numpyro.handlers.trace(conditioned_model).get_trace(
        fit_data,
        parameters["c0"],
        gains,
    )

    assert simulation.diagnostics is not None
    n_trials = stimuli.visible_reports.shape[2]
    site_to_diagnostic = {
        "~$a_{t=*}$": "relative_agreement",
        "~$c_{t=*}$": "pair_scores",
        "~$η^y_{t=*}$": "truth_source_scores",
        "~$η^z_{t=*}$": "probe_source_scores",
        "~$w_{t=*}$": "truth_weights",
        "~$v_{t=*}$": "probe_weights",
        "~$μ_{t=*}$": "truth_mean_angle",
        "~$ν_{t=*}$": "hidden_mean_angle",
    }
    model_values: dict[str, np.ndarray] = {}
    for site_template, diagnostic_name in site_to_diagnostic.items():
        values = _stack_trial_sites(trace, site_template, n_trials)
        model_values[diagnostic_name] = values
        np.testing.assert_allclose(
            values,
            np.asarray(simulation.diagnostics[diagnostic_name]),
            rtol=1e-11,
            atol=1e-11,
        )

    expected_response_mean = np.where(
        np.asarray(stimuli.trial_type) == HIDDEN_SOURCE_PROBE,
        model_values["hidden_mean_angle"],
        model_values["truth_mean_angle"],
    )
    np.testing.assert_allclose(
        expected_response_mean,
        np.asarray(simulation.diagnostics.response_mean_angle),
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        np.asarray(simulation.diagnostics.pre_update_pair_scores)[:, :, 1:],
        np.asarray(simulation.diagnostics.pair_scores)[:, :, :-1],
    )
    np.testing.assert_allclose(
        np.asarray(simulation.diagnostics.pair_scores)[:, :, 2],
        np.asarray(simulation.diagnostics.pair_scores)[:, :, 1],
    )
    assert np.all(np.isfinite(model_values["truth_mean_angle"]))
    assert np.all(np.isfinite(model_values["hidden_mean_angle"]))
    assert float(trace["~$g_y$"]["value"]) == pytest.approx(
        parameters["gamma_y"] * gains.g_star_y
    )
    assert float(trace["~$g_z$"]["value"]) == pytest.approx(
        parameters["gamma_z"] * gains.g_star_z
    )
    log_joint, _ = log_density(
        model_coupling,
        (fit_data, parameters["c0"], gains),
        {},
        {name: parameters[name] for name in parameters if name != "c0"},
    )
    assert np.isfinite(float(log_joint))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("trial_code", "trial_type"),
        ("truth_hidden", "truth trials"),
        ("truth_nan", "finite on valid truth"),
        ("probe_missing", "one NaN"),
        ("probe_wrong_nan", "one NaN"),
    ],
)
def test_stimulus_batch_rejects_malformed_valid_trials(
    case: str,
    message: str,
) -> None:
    reports = np.array([[[[0.1, 0.4, 0.8, 1.2]]]])
    hidden_source = np.array([[[NO_HIDDEN_SOURCE]]])
    trial_type = np.array([[[TRUTH_ESTIMATION]]])
    if case == "trial_code":
        trial_type[0, 0, 0] = 2
    elif case == "truth_hidden":
        hidden_source[0, 0, 0] = 0
    elif case == "truth_nan":
        reports[0, 0, 0, 0] = np.nan
    elif case == "probe_missing":
        trial_type[0, 0, 0] = HIDDEN_SOURCE_PROBE
        hidden_source[0, 0, 0] = 2
    elif case == "probe_wrong_nan":
        trial_type[0, 0, 0] = HIDDEN_SOURCE_PROBE
        hidden_source[0, 0, 0] = 2
        reports[0, 0, 0, 1] = np.nan

    with pytest.raises(ValueError, match=message):
        StimulusBatch(
            visible_reports=reports,
            hidden_source=hidden_source,
            trial_type=trial_type,
            valid_trials=np.ones((1, 1, 1), dtype=bool),
            metadata=_metadata(np.nan_to_num(reports)),
        )


def test_simulation_batch_rejects_inconsistent_combined_responses() -> None:
    simulation = simulate_coupling(
        seed=23,
        batch=_parity_stimuli(),
        parameters={
            "learningRate": 0.2,
            "gamma_y": 1.0,
            "gamma_z": 1.0,
            "vmConcentration": 8.0,
            "c0": 0.0,
        },
        gains=GainConstants(g_star_y=19.946, g_star_z=28.121),
    )
    responses = np.asarray(simulation.responses).copy()
    responses[0, 0, 0] += 0.2

    with pytest.raises(ValueError, match="active response branch"):
        replace(simulation, responses=responses)


def test_model_wraps_simulator_responses_to_von_mises_support() -> None:
    parameters = {
        "learningRate": 0.2,
        "gamma_y": 1.0,
        "gamma_z": 1.0,
        "vmConcentration": 8.0,
        "c0": 0.0,
    }
    gains = GainConstants(g_star_y=19.946, g_star_z=28.121)
    simulation = simulate_coupling(
        seed=29,
        batch=_parity_stimuli(),
        parameters=parameters,
        gains=gains,
    )
    shifted = replace(
        simulation,
        truth_estimates=np.where(
            np.isfinite(simulation.truth_estimates),
            np.asarray(simulation.truth_estimates) + 2 * np.pi,
            np.nan,
        ),
        hidden_predictions=np.where(
            np.isfinite(simulation.hidden_predictions),
            np.asarray(simulation.hidden_predictions) + 2 * np.pi,
            np.nan,
        ),
        responses=np.where(
            np.isfinite(simulation.responses),
            np.asarray(simulation.responses) + 2 * np.pi,
            np.nan,
        ),
    )
    conditioned_model = numpyro.handlers.substitute(
        model_coupling,
        data={name: parameters[name] for name in parameters if name != "c0"},
    )

    with numpyro.validation_enabled():
        trace = numpyro.handlers.trace(conditioned_model).get_trace(
            prepare_simulation_for_fit(shifted),
            parameters["c0"],
            gains,
        )

    for trial_index in range(shifted.stimuli.trial_type.shape[-1]):
        for branch in ("y", "z"):
            observed = np.asarray(trace[f"${branch}_{{t={trial_index}}}$"]["value"])
            assert np.all(observed >= -np.pi)
            assert np.all(observed <= np.pi)
