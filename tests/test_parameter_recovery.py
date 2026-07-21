import json

from coupling_sources.core.data import GainConstants, StimulusSpec
from coupling_sources.workflows import parameter_recovery
from coupling_sources.workflows.job_grid import GridRun


def test_prepare_job_writes_worker_inputs(tmp_path) -> None:
    spec = StimulusSpec(
        seed=1,
        n_participants=2,
        n_sources=4,
        n_rounds=1,
        n_trials_full=4,
        n_trials_partial=4,
        sigma=0.3,
        lam=0.7,
        coupled_sources=(0, 1),
        resultant_floor_full=0.6,
        resultant_floor_leave_one_out=0.6,
    )
    job = parameter_recovery._prepare_job(
        run_dir=tmp_path,
        run=GridRun(
            "job_000000",
            spec,
            GainConstants(g_star_y=19.946, g_star_z=28.121),
            {"value": 1.0, "c0": 0.25},
            {
                "mode": "auto",
                "seed": 100,
                "n_monte_carlo_replicates": 20,
            },
        ),
        simulation_seed=10,
    )

    job_config = json.loads((job.job_dir / "job.json").read_text())
    assert job_config["stimulus"]["coupled_sources"] == [0, 1]
    assert job_config["stimulus"]["seed"] == 1
    assert job_config["stimulus"]["n_participants"] == 2
    assert job_config["gain_constants"] == {
        "g_star_y": 19.946,
        "g_star_z": 28.121,
    }
    assert job_config["gain_calibration"] == {
        "mode": "auto",
        "seed": 100,
        "n_monte_carlo_replicates": 20,
    }
    assert job_config["simulation"] == {
        "seed": 10,
        "parameters": {"value": 1.0, "c0": 0.25},
    }
    assert not (job.job_dir / "stimulus_parameters.json").exists()
    assert not (job.job_dir / "simulation_parameters.json").exists()
    assert not (job.job_dir / "stimuli.npz").exists()
