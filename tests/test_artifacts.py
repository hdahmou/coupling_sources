from __future__ import annotations

import numpy as np
import pytest

from coupling_sources.core.data import (
    CouplingDiagnostics,
    FitData,
    SimulationBatch,
    StimulusBatch,
    StimulusMetadata,
    prepare_simulation_for_fit,
)
from coupling_sources.utils.artifacts import Artifacts


def _stimuli() -> StimulusBatch:
    return StimulusBatch(
        visible_reports=np.array([[[[np.nan, 0.2], [0.3, 0.4]]]]),
        hidden_source=np.array([[[0, -1]]]),
        trial_type=np.array([[[1, 0]]]),
        valid_trials=np.array([[[True, True]]]),
        metadata=StimulusMetadata(
            complete_reports=np.array([[[[0.1, 0.2], [0.3, 0.4]]]]),
            true_direction=np.array([[[0.5, 0.6]]]),
            source_choices=np.array([0, 1]),
            stimulus_covariance=np.eye(2),
            resultant_length_full=np.ones((1, 1, 2)),
            resultant_length_leave_one_out=np.ones((1, 1, 2, 2)),
        ),
    )


def _diagnostics() -> CouplingDiagnostics:
    values = np.array([[0.1, 0.2]])
    return CouplingDiagnostics(
        visible_source_mask=np.array([[True, False]]),
        relative_agreement=values,
        pre_update_pair_scores=values + 1.0,
        pair_scores=values + 2.0,
        truth_source_scores=values + 3.0,
        probe_source_scores=values + 4.0,
        truth_weights=values + 5.0,
        probe_weights=values + 6.0,
        truth_mean_angle=values + 7.0,
        hidden_mean_angle=values + 8.0,
        response_mean_angle=values + 9.0,
    )


def test_prior_and_posterior_artifacts_round_trip(tmp_path):
    prior_path = tmp_path / "prior.npz"
    posterior_path = tmp_path / "posterior.npz"

    Artifacts.write_dist_samples(
        prior_path,
        {
            "learningRate": np.array([0.1, 0.2]),
            "gamma_y": np.array([-0.5, 1.0]),
        },
    )
    prior = Artifacts.read_dist_samples(prior_path)
    np.testing.assert_allclose(prior["learningRate"], [0.1, 0.2], rtol=1e-3)
    np.testing.assert_allclose(prior["gamma_y"], [-0.5, 1.0], rtol=1e-3)

    diagnostics = {
        "r_hat": {"learningRate": 1.01},
        "ess": {"learningRate": 120.0},
        "has_divergences": False,
        "n_divergences": 0,
    }
    Artifacts.write_dist_samples(
        posterior_path,
        {"learningRate": np.array([[0.1, 0.2], [0.3, 0.4]])},
    )
    Artifacts.write_mcmc_diagnostics(posterior_path, diagnostics)
    posterior = Artifacts.read_dist_samples(posterior_path)
    np.testing.assert_allclose(
        posterior["learningRate"],
        [[0.1, 0.2], [0.3, 0.4]],
        rtol=1e-3,
    )
    assert Artifacts.read_mcmc_diagnostics(posterior_path) == diagnostics


def test_stimulus_and_simulation_artifacts_round_trip(tmp_path):
    stimuli_path = tmp_path / "stimuli.npz"
    simulation_path = tmp_path / "simulation.npz"

    stimuli = _stimuli()
    Artifacts.write_config(stimuli_path, stimuli.to_config())
    loaded_stimuli = StimulusBatch.from_config(Artifacts.read_config(stimuli_path))
    np.testing.assert_array_equal(loaded_stimuli.hidden_source, stimuli.hidden_source)
    np.testing.assert_array_equal(loaded_stimuli.valid_trials, stimuli.valid_trials)
    np.testing.assert_allclose(
        loaded_stimuli.metadata.stimulus_covariance,
        stimuli.metadata.stimulus_covariance,
    )

    simulation = SimulationBatch(
        stimuli=stimuli,
        truth_estimates=np.array([[[np.nan, 0.2]]]),
        hidden_predictions=np.array([[[0.3, np.nan]]]),
        responses=np.array([[[0.3, 0.2]]]),
        participant_parameters={
            "learningRate": 0.25,
            "gamma_y": 0.5,
            "gamma_z": 1.0,
            "g_y": 9.973,
            "g_z": 28.121,
            "vmConcentration": 6.0,
            "c0": 0.0,
        },
        diagnostics=_diagnostics(),
    )
    Artifacts.write_config(
        simulation_path,
        simulation.to_config(),
        float_dtype=np.float16,
    )
    loaded_simulation = SimulationBatch.from_config(
        Artifacts.read_config(simulation_path)
    )
    np.testing.assert_allclose(
        loaded_simulation.responses,
        simulation.responses,
        rtol=1e-3,
    )
    assert loaded_simulation.participant_parameters == {
        "learningRate": pytest.approx(0.25),
        "gamma_y": pytest.approx(0.5),
        "gamma_z": pytest.approx(1.0),
        "g_y": pytest.approx(9.973, rel=1e-3),
        "g_z": pytest.approx(28.121, rel=1e-3),
        "vmConcentration": pytest.approx(6.0),
        "c0": pytest.approx(0.0),
    }
    assert loaded_simulation.diagnostics is not None
    assert simulation.diagnostics is not None
    np.testing.assert_array_equal(
        loaded_simulation.diagnostics.visible_source_mask,
        simulation.diagnostics.visible_source_mask,
    )


def test_fit_data_artifact_is_full_precision_and_round_trips_exactly(tmp_path):
    simulation = SimulationBatch(
        stimuli=_stimuli(),
        truth_estimates=np.array([[[np.nan, 0.2]]], dtype=np.float64),
        hidden_predictions=np.array([[[0.3, np.nan]]], dtype=np.float64),
        responses=np.array([[[0.3, 0.2]]], dtype=np.float64),
    )
    fit_data = prepare_simulation_for_fit(simulation)
    path = tmp_path / "fit_data.npz"

    Artifacts.write_config(path, fit_data.to_config())
    stored = Artifacts.read_config(path)
    loaded = FitData.from_config(stored)

    assert set(stored) == {
        "visible_reports",
        "hidden_source",
        "trial_type",
        "valid_trials",
        "responses",
    }
    assert stored["visible_reports"].dtype == np.float64
    assert stored["responses"].dtype == np.float64
    for name in stored:
        np.testing.assert_array_equal(
            np.asarray(getattr(loaded, name)),
            np.asarray(getattr(fit_data, name)),
        )


def test_artifact_paths_must_be_npz(tmp_path):
    with pytest.raises(ValueError, match="must end with .npz"):
        Artifacts.write_dist_samples(
            tmp_path / "prior", {"learningRate": np.array([0.1])}
        )
