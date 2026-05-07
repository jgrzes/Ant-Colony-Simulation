from pathlib import Path
from typing import Any, Dict, List

import optuna

import numpy as np

from agent_model import build_step_metrics, run_demo
from dataset_utils import (
    compute_step_metrics_for_sequence,
    load_sequence,
    prepare_agent_df,
)

# Metrics that we try to fit to
DEFAULT_METRICS = [
    "mean_displacement",
    "mean_turning_angle",
    "mean_sinuosity",
    "space_coverage",
]


def _align_and_mse(real_df, sim_df, metrics: List[str]) -> float:
    merged = real_df.merge(sim_df, on="step", suffixes=("_real", "_sim"))
    normalized_errors = []

    for metric in metrics:
        real_col = f"{metric}_real"
        sim_col = f"{metric}_sim"
        if real_col in merged.columns and sim_col in merged.columns:
            real_vals = merged[real_col].to_numpy()
            sim_vals = merged[sim_col].to_numpy()
            diff = real_vals - sim_vals

            real_range = np.nanmax(real_vals) - np.nanmin(real_vals)
            if real_range < 1e-9:
                # If range is tiny, use max value as denominator
                real_range = np.nanmax(np.abs(real_vals)) + 1e-9

            normalized_mse = np.nanmean((diff / real_range) ** 2)
            normalized_errors.append(normalized_mse)

    if not normalized_errors:
        return float("inf")

    return float(np.nanmean(normalized_errors))


def _build_context(sequence_path: str | Path) -> Dict[str, Any]:
    sequence_path = Path(sequence_path)
    seqinfo, gt_df = load_sequence(sequence_path)
    agent_df_real = prepare_agent_df(gt_df)
    first_step = int(agent_df_real["step"].min())
    initial_positions = agent_df_real[agent_df_real["step"] == first_step][
        ["x", "y"]
    ].values.tolist()
    n_ants = len(initial_positions)
    width = int(seqinfo.get("imWidth") or 40)
    height = int(seqinfo.get("imHeight") or 40)
    real_metrics = compute_step_metrics_for_sequence(sequence_path)
    steps = int(real_metrics["step"].max())
    return {
        "sequence_path": sequence_path,
        "initial_positions": initial_positions,
        "n_ants": n_ants,
        "width": width,
        "height": height,
        "real_metrics": real_metrics,
        "steps": steps,
    }


def _evaluate_params(
    params: Dict[str, Any], context: Dict[str, Any], metrics: List[str]
):
    model, sim_agent_df = run_demo(
        steps=context["steps"],
        n_ants=context["n_ants"],
        width=context["width"],
        height=context["height"],
        initial_positions=context["initial_positions"],
        rng=None,
        **params,
    )
    sim_metrics = build_step_metrics(
        sim_agent_df,
        width=context["width"],
        height=context["height"],
        cell_size=10.0,
    )
    loss = _align_and_mse(context["real_metrics"], sim_metrics, metrics)
    return loss, sim_metrics


def optuna_fit(
    sequence_path: str | Path,
    n_iter: int = 40,
    metrics: List[str] | None = None,
    results_dir: str | Path | None = None,
):
    if metrics is None:
        metrics = DEFAULT_METRICS

    results_dir = (
        Path(results_dir) if results_dir else Path("Simulation Metrics Reports")
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    context = _build_context(sequence_path)

    def objective(trial):
        params = {
            "pheromone_deposit": trial.suggest_float("pheromone_deposit", 0.1, 2.0),
            "evaporation_rate": trial.suggest_float("evaporation_rate", 0.005, 0.1),
            "turn_strength": trial.suggest_float("turn_strength", 0.5, 2.0),
            "noise_strength": trial.suggest_float("noise_strength", 0.0, 0.6),
            "sensor_distance": trial.suggest_float("sensor_distance", 0.5, 8.0),
            "sensor_angle": trial.suggest_float("sensor_angle", 0.1, 1.5),
            "pheromone_delay": trial.suggest_int("pheromone_delay", 0, 6),
        }

        loss, sim_metrics = _evaluate_params(params, context, metrics)

        trial.set_user_attr("sim_metrics", sim_metrics)

        return loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_iter)

    best_trial = study.best_trial

    history_df = study.trials_dataframe()

    # clean up column names
    if "value" in history_df.columns:
        history_df.rename(columns={"value": "loss"}, inplace=True)

    if "number" in history_df.columns:
        history_df.rename(columns={"number": "iter"}, inplace=True)

    history_df.columns = [col.replace("params_", "") for col in history_df.columns]

    out_csv = results_dir / f"fit_optuna_{context['sequence_path'].name}_history.csv"
    history_df.to_csv(out_csv, index=False)

    return {
        "best": {
            "loss": best_trial.value,
            "params": best_trial.params,
            "sim_metrics": best_trial.user_attrs.get("sim_metrics"),
        },
        "history_path": out_csv,
        "real_metrics": context["real_metrics"],
    }
