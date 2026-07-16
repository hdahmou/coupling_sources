from dataclasses import replace

import numpy as np
import pytest

from coupling_sources.core.constants import HIDDEN_SOURCE_PROBE, TRUTH_ESTIMATION
from coupling_sources.core.data import StimulusSpec
from coupling_sources.core.stimuli import _resultant_lengths, generate_stimulus
from coupling_sources.workflows.job_grid import _stimulus_parameters


def make_spec(n_participants: int) -> StimulusSpec:
    return StimulusSpec(
        seed=101,
        n_participants=n_participants,
        n_sources=4,
        n_rounds=2,
        n_trials_full=4,
        n_trials_partial=4,
        sigma=0.55,
        lam=0.8,
        coupled_sources=(0, 1),
        resultant_floor_full=0.0,
        resultant_floor_leave_one_out=0.0,
    )


def test_generate_stimulus_uses_requested_participant_count() -> None:
    stimuli = generate_stimulus(make_spec(n_participants=3))

    assert stimuli.visible_reports.shape == (3, 2, 8, 4)
    assert stimuli.hidden_source.shape == (3, 2, 8)
    assert stimuli.trial_type.shape == (3, 2, 8)
    assert stimuli.valid_trials.shape == (3, 2, 8)
    assert stimuli.metadata.complete_reports.shape == (3, 2, 8, 4)
    assert stimuli.metadata.true_direction.shape == (3, 2, 8)
    assert not np.array_equal(
        np.asarray(stimuli.metadata.true_direction[0]),
        np.asarray(stimuli.metadata.true_direction[1]),
    )


def test_first_participant_matches_single_participant_seed() -> None:
    single_participant = generate_stimulus(make_spec(n_participants=1))
    three_participants = generate_stimulus(
        replace(make_spec(n_participants=1), n_participants=3)
    )

    np.testing.assert_allclose(
        np.asarray(single_participant.visible_reports[0]),
        np.asarray(three_participants.visible_reports[0]),
        atol=1e-12,
        equal_nan=True,
    )


def test_stimulus_spec_rejects_empty_participant_count() -> None:
    with pytest.raises(ValueError, match="n_participants"):
        make_spec(n_participants=0)


def test_generate_stimulus_balances_hidden_source_counts() -> None:
    spec = replace(make_spec(n_participants=3), n_trials_partial=8)
    stimuli = generate_stimulus(spec)

    hidden_source = np.asarray(stimuli.hidden_source)
    trial_type = np.asarray(stimuli.trial_type)

    assert np.all(trial_type[hidden_source >= 0] == HIDDEN_SOURCE_PROBE)
    assert np.all(trial_type[hidden_source < 0] == TRUTH_ESTIMATION)

    for participant_hidden_sources in hidden_source.reshape(
        (-1, hidden_source.shape[-1])
    ):
        counts = np.bincount(
            participant_hidden_sources[participant_hidden_sources >= 0],
            minlength=spec.n_sources,
        )
        np.testing.assert_array_equal(
            counts,
            np.full(spec.n_sources, spec.n_trials_partial // spec.n_sources),
        )


def test_stimulus_spec_rejects_unbalanced_partial_trial_count() -> None:
    with pytest.raises(ValueError, match="n_trials_partial.*multiple of n_sources"):
        replace(make_spec(n_participants=1), n_trials_partial=2)


def test_generate_stimulus_respects_both_resultant_floors() -> None:
    threshold = 0.6
    spec = replace(
        make_spec(n_participants=2),
        resultant_floor_full=threshold,
        resultant_floor_leave_one_out=threshold,
    )

    stimuli = generate_stimulus(spec)

    resultant_length_full = np.asarray(stimuli.metadata.resultant_length_full)
    resultant_length_leave_one_out = np.asarray(
        stimuli.metadata.resultant_length_leave_one_out
    )

    assert np.min(resultant_length_full) >= threshold - 1e-6
    assert np.min(resultant_length_leave_one_out) >= threshold - 1e-6


def test_full_resultant_can_pass_while_leave_one_out_resultant_fails() -> None:
    reports = np.array([[0.0, 0.0, 0.0, np.pi]])

    full, leave_one_out = _resultant_lengths(reports)

    assert float(full[0]) == pytest.approx(0.5)
    assert float(np.min(leave_one_out[0])) == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize(
    "field",
    ["resultant_floor_full", "resultant_floor_leave_one_out"],
)
def test_stimulus_spec_rejects_invalid_resultant_floor(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        replace(make_spec(n_participants=1), **{field: 1.1})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_sources": 3}, "n_sources"),
        ({"n_rounds": 0}, "n_rounds"),
        ({"sigma": 0.0}, "sigma"),
        ({"lam": 1.0}, "lam"),
        ({"coupled_sources": (0, 0)}, "coupled_sources"),
        ({"coupled_sources": (0, 4)}, "coupled_sources"),
    ],
)
def test_stimulus_spec_enforces_preregistered_design(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(make_spec(n_participants=1), **changes)


def test_stimulus_grid_parameters_requires_both_resultant_floors() -> None:
    with pytest.raises(KeyError, match="resultant_floor_full"):
        _stimulus_parameters(
            {
                "seed": 101,
                "n_participants": 1,
                "n_sources": 4,
                "n_rounds": 2,
                "n_trials_full": 4,
                "n_trials_partial": 2,
                "sigma": 0.55,
                "lam": 0.8,
                "coupled_sources": [0, 1],
            }
        )


def test_stimulus_grid_parameters_rejects_removed_mean_angle_difference_band() -> None:
    with pytest.raises(ValueError, match="mean_angle_difference_band.*removed"):
        _stimulus_parameters(
            {
                "seed": 101,
                "n_participants": 1,
                "n_sources": 4,
                "n_rounds": 2,
                "n_trials_full": 4,
                "n_trials_partial": 2,
                "sigma": 0.55,
                "lam": 0.8,
                "coupled_sources": [0, 1],
                "mean_angle_difference_band": [0.05, np.pi],
            }
        )


def test_stimulus_grid_parameters_rejects_removed_resultant_floor() -> None:
    with pytest.raises(ValueError, match="resultant_floor.*removed"):
        _stimulus_parameters(
            {
                "seed": 101,
                "n_participants": 1,
                "n_sources": 4,
                "n_rounds": 2,
                "n_trials_full": 4,
                "n_trials_partial": 4,
                "sigma": 0.55,
                "lam": 0.8,
                "coupled_sources": [0, 1],
                "resultant_floor": 0.6,
            }
        )


def test_stimulus_grid_parameters_read_resultant_floors() -> None:
    spec = _stimulus_parameters(
        {
            "seed": 101,
            "n_participants": 1,
            "n_sources": 4,
            "n_rounds": 2,
            "n_trials_full": 4,
            "n_trials_partial": 4,
            "sigma": 0.55,
            "lam": 0.8,
            "coupled_sources": [0, 1],
            "resultant_floor_full": 0.6,
            "resultant_floor_leave_one_out": 0.5,
        }
    )

    assert spec.resultant_floor_full == 0.6
    assert spec.resultant_floor_leave_one_out == 0.5
