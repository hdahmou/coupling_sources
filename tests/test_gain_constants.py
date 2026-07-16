from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from coupling_sources.core.data import StimulusSpec
from coupling_sources.core import gain_calibration
from coupling_sources.workflows import job_grid
from scripts import compute_gain_constants


def _reports() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, np.pi / 2.0, np.pi],
            [np.nan, 0.0, np.pi / 2.0, np.pi],
        ]
    )


def _spec(**changes: object) -> StimulusSpec:
    spec = StimulusSpec(
        seed=1,
        n_participants=2,
        n_sources=4,
        n_rounds=1,
        n_trials_full=4,
        n_trials_partial=4,
        sigma=0.55,
        lam=0.8,
        coupled_sources=(0, 1),
        resultant_floor_full=0.6,
        resultant_floor_leave_one_out=0.6,
    )
    return replace(spec, **changes)


def test_mean_signal_by_role_uses_visible_pair_relative_agreement() -> None:
    means = gain_calibration._mean_signal_by_role(
        _reports()[None, ...],
        (0, 1),
    )

    np.testing.assert_allclose(means["coupled"], [7.0 / 12.0])
    np.testing.assert_allclose(means["mixed"], [-5.0 / 36.0])
    np.testing.assert_allclose(means["independent"], [1.0 / 8.0])


def test_calibrate_gain_constants_uses_branch_specific_score_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = np.repeat(_reports()[None, ...], 2, axis=0)
    monkeypatch.setattr(
        gain_calibration,
        "generate_stimulus",
        lambda spec: SimpleNamespace(visible_reports=reports),
    )
    spec = _spec(
        n_trials_full=1,
        n_trials_partial=0,
        resultant_floor_full=0.0,
        resultant_floor_leave_one_out=0.0,
    )

    result = gain_calibration.calibrate_gain_constants(spec)

    assert result.delta_y == pytest.approx(11.0 / 72.0)
    assert result.delta_z == pytest.approx(13.0 / 18.0)
    assert result.g_star_y == pytest.approx(np.log(1.8) / result.delta_y)
    assert result.g_star_z == pytest.approx(np.log(13.0) / result.delta_z)
    assert result.g_star_y_se == 0.0
    assert result.g_star_z_se == 0.0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_participants": 1}, "at least two Monte Carlo replicates"),
        ({"n_rounds": 2}, "n_rounds=1"),
        ({"n_trials_full": 0}, "at least one trial"),
    ],
)
def test_calibration_rejects_invalid_monte_carlo_design(
    changes: dict[str, int],
    message: str,
) -> None:
    spec = _spec(
        n_trials_full=1,
        n_trials_partial=0,
        resultant_floor_full=0.0,
        resultant_floor_leave_one_out=0.0,
    )

    with pytest.raises(ValueError, match=message):
        gain_calibration.calibrate_gain_constants(replace(spec, **changes))


def test_locked_gains_must_match_recorded_calibration() -> None:
    config = {
        "gain_constants": {"g_star_y": 19.946, "g_star_z": 28.121},
        "gain_calibration": {
            "mode": "fixed",
            "seed": 1,
            "n_monte_carlo_replicates": 2,
            "n_sources": 4,
            "n_trials_full": 4,
            "n_trials_partial": 4,
            "sigma": 0.55,
            "lam": 0.8,
            "coupled_sources": [0, 1],
            "resultant_floor_full": 0.6,
            "resultant_floor_leave_one_out": 0.6,
            "estimates": {
                "g_star_y": 19.94570023037356,
                "g_star_z": 28.1212114090911,
            }
        },
    }

    gains = job_grid._resolve_gains(config, [_spec()])[0].constants

    assert gains.g_star_y == 19.946
    assert gains.g_star_z == 28.121

    config["gain_constants"]["g_star_y"] = 19.0
    with pytest.raises(ValueError, match="g_star_y.*calibration"):
        job_grid._resolve_gains(config, [_spec()])


def test_fixed_calibration_must_match_stimulus_design() -> None:
    config = {
        "gain_constants": {"g_star_y": 1.0, "g_star_z": 1.0},
        "gain_calibration": {
            "mode": "fixed",
            "seed": 1,
            "n_monte_carlo_replicates": 2,
            "n_sources": 4,
            "n_trials_full": 8,
            "n_trials_partial": 4,
            "sigma": 0.55,
            "lam": 0.8,
            "coupled_sources": [0, 1],
            "resultant_floor_full": 0.6,
            "resultant_floor_leave_one_out": 0.6,
            "estimates": {"g_star_y": 1.0, "g_star_z": 1.0},
        },
    }

    with pytest.raises(ValueError, match="n_trials_full.*stimulus design"):
        job_grid._resolve_gains(config, [_spec()])


def _calibration(g_star_y: float, g_star_z: float) -> gain_calibration.GainCalibration:
    return gain_calibration.GainCalibration(
        s_coupled=0.2,
        s_coupled_se=0.01,
        s_mixed=0.0,
        s_mixed_se=0.01,
        s_independent=-0.1,
        s_independent_se=0.01,
        delta_y=0.1,
        delta_y_se=0.01,
        delta_z=0.2,
        delta_z_se=0.01,
        g_star_y=g_star_y,
        g_star_y_se=0.1,
        g_star_z=g_star_z,
        g_star_z_se=0.1,
    )


def test_auto_gains_are_computed_once_per_calibration_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _spec(
        n_rounds=3,
    )
    designs = [
        base,
        replace(base, n_trials_full=8),
        replace(base, seed=99, n_participants=5, n_rounds=7),
    ]
    calibrated_specs: list[StimulusSpec] = []

    def calibrate(spec: StimulusSpec) -> gain_calibration.GainCalibration:
        calibrated_specs.append(spec)
        return _calibration(float(spec.n_trials_full), float(spec.n_trials_partial))

    monkeypatch.setattr(job_grid, "calibrate_gain_constants", calibrate)
    resolved = job_grid._resolve_gains(
        {
            "gain_calibration": {
                "mode": "auto",
                "seed": 123,
                "n_monte_carlo_replicates": 50,
            }
        },
        designs,
    )

    assert len(calibrated_specs) == 2
    assert all(spec.seed == 123 for spec in calibrated_specs)
    assert all(spec.n_participants == 50 for spec in calibrated_specs)
    assert all(spec.n_rounds == 1 for spec in calibrated_specs)
    assert resolved[0].constants.g_star_y == 4.0
    assert resolved[1].constants.g_star_y == 8.0
    assert resolved[2] == resolved[0]
    assert resolved[0].calibration["mode"] == "auto"


def test_auto_and_fixed_gain_modes_cannot_be_combined() -> None:
    with pytest.raises(ValueError, match="cannot include gain_constants"):
        job_grid._resolve_gains(
            {
                "gain_constants": {"g_star_y": 1.0, "g_star_z": 1.0},
                "gain_calibration": {
                    "mode": "auto",
                    "estimates": {"g_star_y": 1.0, "g_star_z": 1.0},
                },
            },
            [_spec()],
        )


def test_gain_calibration_mode_is_required() -> None:
    with pytest.raises(ValueError, match="mode must be 'auto' or 'fixed'"):
        job_grid._resolve_gains(
            {
                "gain_constants": {"g_star_y": 1.0, "g_star_z": 1.0},
                "gain_calibration": {
                    "estimates": {"g_star_y": 1.0, "g_star_z": 1.0},
                },
            },
            [_spec()],
        )


def test_gain_calibration_provenance_is_required() -> None:
    with pytest.raises(ValueError, match="gain_calibration must be an object"):
        job_grid._resolve_gains(
            {"gain_constants": {"g_star_y": 1.0, "g_star_z": 1.0}},
            [_spec()],
        )


def test_standalone_calibration_reports_and_enforces_stability(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        compute_gain_constants,
        "calibrate_gain_constants",
        lambda spec: _calibration(20.0, 30.0),
    )

    result = compute_gain_constants.main(
        [
            "--sigma",
            "0.55",
            "--lam",
            "0.8",
            "--resultant-floor-full",
            "0.6",
            "--resultant-floor-leave-one-out",
            "0.6",
            "--n-trials-full",
            "28",
            "--n-trials-partial",
            "12",
            "--n-monte-carlo-replicates",
            "20",
            "--max-relative-mc-se",
            "0.004",
        ]
    )

    output = capsys.readouterr().out
    assert result == 1
    assert "relative MC se 0.50%" in output
    assert "approx. 95% MC interval" in output
    assert "stability: FAIL" in output
    assert '"stability"' in output
