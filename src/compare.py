from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from agent_model import build_step_metrics, run_demo
from dataset_utils import (
    compute_step_metrics_for_sequence,
    load_sequence,
    prepare_agent_df,
)
from plots import (
    plot_ant_step_by_step_gif,
    plot_fit_history,
    plot_metric_comparison,
)


def _coerce_value(value: str) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    text = str(value).strip()
    if text == "":
        return text
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in text or "e" in lowered:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _load_best_params(history_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    history_df = pd.read_csv(history_path)
    if history_df.empty:
        raise ValueError(f"History file is empty: {history_path}")

    best_row = history_df.loc[history_df["loss"].idxmin()].to_dict()
    params = {
        key: _coerce_value(value)
        for key, value in best_row.items()
        if key not in {"iter", "loss"}
    }
    if "pheromone_delay" in params and params["pheromone_delay"] is not None:
        params["pheromone_delay"] = int(params["pheromone_delay"])
    return history_df, params


def compare_sequence(sequence_path: str | Path, history_path: str | Path | None = None):
    project_root = Path(__file__).resolve().parents[1]
    sequence_path = Path(sequence_path)
    if not sequence_path.is_absolute():
        sequence_path = (project_root / sequence_path).resolve()

    if history_path is None:
        history_path = (
            project_root
            / "Simulation Metrics Reports"
            / f"fit_{sequence_path.name}_history.csv"
        )
    history_path = Path(history_path)

    if not sequence_path.exists():
        raise FileNotFoundError(f"Sequence path does not exist: {sequence_path}")
    if not history_path.exists():
        raise FileNotFoundError(f"History path does not exist: {history_path}")

    seqinfo, gt_df = load_sequence(sequence_path)
    real_agent_df = prepare_agent_df(gt_df)
    first_step = int(real_agent_df["step"].min())
    initial_positions = real_agent_df[real_agent_df["step"] == first_step][
        ["x", "y"]
    ].values.tolist()
    n_ants = len(initial_positions)
    width = int(seqinfo.get("imWidth") or 40)
    height = int(seqinfo.get("imHeight") or 40)
    steps = int(real_agent_df["step"].max())

    real_metrics = compute_step_metrics_for_sequence(sequence_path)
    history_df, best_params = _load_best_params(history_path)

    model, sim_agent_df = run_demo(
        steps=steps,
        n_ants=n_ants,
        width=width,
        height=height,
        initial_positions=initial_positions,
        rng=42,
        **best_params,
    )
    sim_metrics = build_step_metrics(
        sim_agent_df,
        nest_x=model.nest_x,
        nest_y=model.nest_y,
        width=width,
        height=height,
        cell_size=1.0,
    )

    output_dir = project_root / "Generated Plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_fit_history(
        history_df, save_path=output_dir / f"{sequence_path.name}_fit_history.png"
    )

    for metric in [
        "mean_distance",
        "dispersion",
        "mean_turning_angle",
        "mean_displacement",
        "mean_sinuosity",
        "space_coverage",
    ]:
        if metric in real_metrics.columns and metric in sim_metrics.columns:
            plot_metric_comparison(
                real_metrics[["step", metric]],
                sim_metrics[["step", metric]],
                metric=metric,
                save_path=output_dir / f"{sequence_path.name}_{metric}_real_vs_sim.png",
            )

    plot_ant_step_by_step_gif(
        real_agent_df,
        width=width,
        height=height,
        title=f"{sequence_path.name}_real",
        save_path=output_dir / f"{sequence_path.name}_real.gif",
    )
    plot_ant_step_by_step_gif(
        sim_agent_df,
        width=width,
        height=height,
        pheromone_grid=model.pheromone_grid,
        title=f"{sequence_path.name}_sim",
        save_path=output_dir / f"{sequence_path.name}_sim.gif",
    )

    return {
        "history_path": history_path,
        "best_params": best_params,
        "output_dir": output_dir,
    }
