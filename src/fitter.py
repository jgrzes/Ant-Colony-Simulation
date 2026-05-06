from pathlib import Path
import csv
import random
from typing import Any, Dict, List, Tuple

import numpy as np

from agent_model import build_step_metrics, run_demo
from dataset_utils import (
    compute_step_metrics_for_sequence,
    load_sequence,
    prepare_agent_df,
)

DEFAULT_METRICS = [
    "mean_distance",
    "dispersion",
    "mean_turning_angle",
    "mean_displacement",
    "mean_sinuosity",
    "space_coverage",
]

DEFAULT_PARAM_SPACE: Dict[str, Tuple[float, float]] = {
    "pheromone_deposit": (0.1, 2.0),
    "evaporation_rate": (0.005, 0.1),
    "turn_strength": (0.1, 1.2),
    "noise_strength": (0.0, 0.6),
    "sensor_distance": (1.0, 5.0),
    "sensor_angle": (0.1, 1.5),
    "pheromone_delay": (0, 6),
}


def _align_and_mse(real_df, sim_df, metrics: List[str]) -> float:
    merged = real_df.merge(sim_df, on="step", suffixes=("_real", "_sim"))
    errors = []
    for metric in metrics:
        real_col = f"{metric}_real"
        sim_col = f"{metric}_sim"
        if real_col in merged.columns and sim_col in merged.columns:
            diff = merged[real_col].to_numpy() - merged[sim_col].to_numpy()
            errors.append(np.nanmean(diff**2))
    if not errors:
        return float("inf")
    return float(np.nanmean(errors))


def _coerce_param_value(name: str, value: float) -> Any:
    if name == "pheromone_delay":
        return int(round(value))
    return float(value)


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
        nest_x=model.nest_x,
        nest_y=model.nest_y,
        width=context["width"],
        height=context["height"],
        cell_size=1.0,
    )
    loss = _align_and_mse(context["real_metrics"], sim_metrics, metrics)
    return loss, sim_metrics


def random_search_fit(
    sequence_path: str | Path,
    param_space: Dict[str, Tuple[float, float]] | None = None,
    n_iter: int = 40,
    metrics: List[str] | None = None,
    results_dir: str | Path | None = None,
):
    if param_space is None:
        param_space = DEFAULT_PARAM_SPACE
    if metrics is None:
        metrics = DEFAULT_METRICS
    if results_dir is None:
        results_dir = Path("Simulation Metrics Reports")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    context = _build_context(sequence_path)
    history: List[Dict[str, Any]] = []
    best = {"loss": float("inf"), "params": None, "sim_metrics": None}

    for i in range(n_iter):
        params = {}
        for name, bounds in param_space.items():
            value = random.uniform(*bounds)
            params[name] = _coerce_param_value(name, value)
        loss, sim_metrics = _evaluate_params(params, context, metrics)
        history.append({"iter": i, "loss": loss, **params})
        if loss < best["loss"]:
            best = {"loss": loss, "params": params, "sim_metrics": sim_metrics}

    out_csv = results_dir / f"fit_{context['sequence_path'].name}_history.csv"
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    return {
        "best": best,
        "history_path": out_csv,
        "real_metrics": context["real_metrics"],
    }
