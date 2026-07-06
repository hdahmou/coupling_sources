from __future__ import annotations

import sys
from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp
import numpyro

from coupling_sources.core.constants import HIDDEN_SOURCE_PROBE, TRUTH_ESTIMATION
from coupling_sources.core.data import SimulationBatch, StimulusBatch, StimulusMetadata
from coupling_sources.core.model import model_coupling


def toy_inputs() -> tuple[SimulationBatch]:
    n_participants, n_rounds, n_trials, n_sources = 2, 2, 2, 4
    visible_reports = jnp.zeros((n_participants, n_rounds, n_trials, n_sources))
    hidden_source = -jnp.ones((n_participants, n_rounds, n_trials), dtype=jnp.int32)
    trial_type = TRUTH_ESTIMATION * jnp.ones((n_participants, n_rounds, n_trials), dtype=jnp.int32)
    hidden_source = hidden_source.at[:, :, -1].set(0)
    trial_type = trial_type.at[:, :, -1].set(HIDDEN_SOURCE_PROBE)
    visible_reports = visible_reports.at[:, :, -1, 0].set(jnp.nan)
    valid_trials = jnp.ones((n_participants, n_rounds, n_trials), dtype=bool)
    truth_estimates = jnp.zeros((n_participants, n_rounds, n_trials))
    hidden_predictions = jnp.zeros((n_participants, n_rounds, n_trials))
    stimuli = StimulusBatch(
        visible_reports=visible_reports,
        hidden_source=hidden_source,
        trial_type=trial_type,
        valid_trials=valid_trials,
        metadata=StimulusMetadata(
            complete_reports=jnp.zeros_like(visible_reports),
            true_direction=jnp.zeros((n_participants, n_rounds, n_trials)),
            source_choices=visible_reports,
            stimulus_covariance=jnp.eye(n_sources),
        ),
    )
    batch = SimulationBatch(
        stimuli=stimuli,
        truth_estimates=truth_estimates,
        hidden_predictions=hidden_predictions,
        responses=jnp.zeros((n_participants, n_rounds, n_trials)),
    )
    return (batch,)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the graph from coupling_sources.core.model.")
    parser.add_argument("--output-stem", type=Path, default=Path("artifacts/preregistration_model_graph_numpyro"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        numpyro.render_model(model_coupling, model_args=toy_inputs(), filename=args.output_stem.with_suffix(suffix))
    print(f"rendered {args.output_stem} (.pdf, .png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
