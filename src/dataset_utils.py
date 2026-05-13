"""
Scripts used for importing and parsing files from the dataset, computing metrics and generating reports.
"""

from configparser import ConfigParser
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from InquirerPy import inquirer

from metrics import (
    compute_colony_dispersion,
    compute_mean_turning_angle,
    compute_mean_displacement,
    compute_sinuosity,
)

from plots import (
    plot_ant_trajectories,
    plot_space_coverage,
    plot_colony_dispersion,
    plot_mean_turning_angle,
    plot_mean_displacement,
    plot_sinuosity,
)


def load_seqinfo(seq_dir: str | Path) -> dict[str, Any]:
    seq_dir = Path(seq_dir)
    seqinfo_path = seq_dir / "seqinfo.ini"

    parser = ConfigParser()
    parser.read(seqinfo_path)

    seq = parser["Sequence"]

    return {
        "name": seq.get("name", seq_dir.name),
        "imDir": seq.get("imDir", "img"),
        "frameRate": seq.getint("frameRate", fallback=None),
        "seqLength": seq.getint("seqLength", fallback=None),
        "imWidth": seq.getint("imWidth", fallback=None),
        "imHeight": seq.getint("imHeight", fallback=None),
        "imExt": seq.get("imExt", ".jpg"),
    }


def load_gt(seq_dir: str | Path) -> pd.DataFrame:
    seq_dir = Path(seq_dir)
    gt_path = seq_dir / "gt" / "gt.txt"

    df = pd.read_csv(gt_path, header=None)

    # frame, id, bb_left, bb_top, bb_width, bb_height, conf
    base_cols = ["frame", "ant_id", "bb_left", "bb_top", "bb_width", "bb_height"]
    extra_cols = [f"extra_{i}" for i in range(df.shape[1] - 6)]
    df.columns = base_cols + extra_cols

    # we calculate the center of the bounding box as the position of the ant
    df["x"] = df["bb_left"] + df["bb_width"] / 2.0
    df["y"] = df["bb_top"] + df["bb_height"] / 2.0

    df = df.sort_values(["ant_id", "frame"]).reset_index(drop=True)
    return df


def load_sequence(seq_dir: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    seqinfo = load_seqinfo(seq_dir)
    gt_df = load_gt(seq_dir)
    return seqinfo, gt_df


def prepare_agent_df(gt_df: pd.DataFrame) -> pd.DataFrame:
    agent_df = gt_df.rename(columns={"frame": "step", "ant_id": "agent_id"}).copy()
    required_columns = ["step", "agent_id", "x", "y"]
    return agent_df[required_columns]


def compute_step_metrics_for_sequence(
    seq_dir: str | Path, cell_size: float = 10.0
) -> pd.DataFrame:
    seqinfo, gt_df = load_sequence(seq_dir)
    agent_df = prepare_agent_df(gt_df)

    per_step_df = compute_colony_dispersion(agent_df)

    turning_df = compute_mean_turning_angle(agent_df)
    displacement_df = compute_mean_displacement(agent_df)
    sinuosity_df = compute_sinuosity(agent_df)
    for extra in (turning_df, displacement_df, sinuosity_df):
        per_step_df = per_step_df.merge(extra, on="step", how="left")

    width = float(seqinfo["imWidth"] or 0)
    height = float(seqinfo["imHeight"] or 0)
    total_cells = int((width // cell_size) * (height // cell_size))

    cells_df = agent_df.copy()
    cells_df["cell_x"] = (cells_df["x"] // cell_size).astype(int)
    cells_df["cell_y"] = (cells_df["y"] // cell_size).astype(int)

    steps = sorted(cells_df["step"].unique())
    visited_cells: set[tuple[int, int]] = set()
    coverage_by_step: dict[int, float] = {}

    for step in steps:
        step_cells = cells_df.loc[
            cells_df["step"] == step, ["cell_x", "cell_y"]
        ].drop_duplicates()
        visited_cells.update(map(tuple, step_cells.to_numpy()))
        if total_cells == 0:
            coverage_by_step[int(step)] = 0.0
        else:
            coverage_by_step[int(step)] = len(visited_cells) / total_cells

    per_step_df["space_coverage"] = (
        per_step_df["step"].map(coverage_by_step).fillna(0.0)
    )
    per_step_df["ants"] = (
        agent_df.groupby("step")["agent_id"]
        .nunique()
        .reindex(per_step_df["step"])
        .to_numpy()
    )

    return per_step_df.sort_values("step").reset_index(drop=True)


def compute_metrics_for_seq(
    sequence_path: str | Path,
    report_dir: str | Path | None = None,
    cell_size: float = 10.0,
) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    sequence_path = Path(sequence_path)
    if not sequence_path.is_absolute():
        sequence_path = project_root / sequence_path
    sequence_path = sequence_path.resolve()

    if not sequence_path.exists():
        raise FileNotFoundError(f"Sequence path does not exist: {sequence_path}")

    if report_dir is None:
        report_dir = project_root / "Dataset Metrics Reports"
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_per_step_df = compute_step_metrics_for_sequence(
        sequence_path, cell_size=cell_size
    )
    step_report_path = report_dir / f"{sequence_path.name}_metrics_per_step.csv"
    metrics_per_step_df.to_csv(step_report_path, index=False)

    seqinfo, gt_df = load_sequence(sequence_path)
    gt_df = gt_df[gt_df["frame"] < 100]
    agent_df = prepare_agent_df(gt_df)

    try:
        plot_ant_trajectories(
            agent_df=agent_df,
            width=seqinfo["imWidth"],
            height=seqinfo["imHeight"],
            title=f"Ant Trajectories - {seqinfo['name']}",
        )
        plot_space_coverage(
            metrics_per_step_df[["step", "space_coverage"]], is_simulation=False
        )
        plot_colony_dispersion(
            metrics_per_step_df[["step", "dispersion"]], is_simulation=False
        )
        plot_mean_turning_angle(
            metrics_per_step_df[["step", "mean_turning_angle"]], is_simulation=False
        )
        plot_mean_displacement(
            metrics_per_step_df[["step", "mean_displacement"]], is_simulation=False
        )
        plot_sinuosity(
            metrics_per_step_df[["step", "mean_sinuosity"]], is_simulation=False
        )
    except Exception as e:
        print(f"Failed to generate plots: {e}")

    return step_report_path


# For testing purposes, TODO - move to separate script and add compratator with simulation results
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    if len(sys.argv) > 1:
        sequence_path = Path(sys.argv[1])
    else:
        default_path = (
            project_root / "dataset" / "IndoorDataset" / "Seq0001Object10Image94"
        )
        sequence_path = inquirer.text(
            message="Path to sequence directory:",
            default=str(default_path),
        ).execute()

    step_report_path = compute_metrics_for_seq(sequence_path, cell_size=10.0)
    print("Metrics computation and plotting completed.")
