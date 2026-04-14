"""
Scripts used for importing and parsing files from the dataset, computing metrics and generating reports.
"""

from configparser import ConfigParser
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from metrics import (
    compute_colony_dispersion,
    compute_distance_from_nest,
)

from plots import (
    plot_ant_trajectories, 
    plot_mean_distance, 
    plot_space_coverage
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

# we have to change it probably
def infer_nest_position(agent_df: pd.DataFrame) -> tuple[float, float]:
    first_step = agent_df["step"].min()
    first_positions = agent_df[agent_df["step"] == first_step]
    return float(first_positions["x"].mean()), float(first_positions["y"].mean())


def compute_step_metrics_for_sequence(seq_dir: str | Path, cell_size: float = 10.0) -> pd.DataFrame:
    seqinfo, gt_df = load_sequence(seq_dir)
    agent_df = prepare_agent_df(gt_df)

    nest_x, nest_y = infer_nest_position(agent_df)

    _, mean_distance_per_step, _ = compute_distance_from_nest(
        agent_df,
        nest_x=nest_x,
        nest_y=nest_y,
    )
    dispersion_per_step = compute_colony_dispersion(
        agent_df,
        nest_x=nest_x,
        nest_y=nest_y,
    )

    per_step_df = mean_distance_per_step.merge(dispersion_per_step, on="step", how="left")

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
        step_cells = cells_df.loc[cells_df["step"] == step, ["cell_x", "cell_y"]].drop_duplicates()
        visited_cells.update(map(tuple, step_cells.to_numpy()))
        if total_cells == 0:
            coverage_by_step[int(step)] = 0.0
        else:
            coverage_by_step[int(step)] = len(visited_cells) / total_cells

    per_step_df["space_coverage"] = per_step_df["step"].map(coverage_by_step).fillna(0.0)
    per_step_df["ants"] = (
        agent_df.groupby("step")["agent_id"]
        .nunique()
        .reindex(per_step_df["step"])
        .to_numpy()
    )

    return per_step_df.sort_values("step").reset_index(drop=True)


# For testing purposes
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    report_dir = project_root / "Metric Raports"
    report_dir.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        sequence_path = Path(sys.argv[1])
        if not sequence_path.is_absolute():
            sequence_path = project_root / sequence_path
    else:
        sequence_path = project_root / "dataset" / "IndoorDataset" / "Seq0001Object10Image94"

    sequence_path = sequence_path.resolve()
    if not sequence_path.exists():
        raise FileNotFoundError(f"Nie znaleziono ścieżki sekwencji: {sequence_path}")

    metrics_per_step_df = compute_step_metrics_for_sequence(sequence_path, cell_size=10.0)
    step_report_path = report_dir / f"{sequence_path.name}_metrics_per_step.csv"

    metrics_per_step_df.to_csv(step_report_path, index=False)

    print(f"Step report saved to: {step_report_path}")

    seqinfo, gt_df = load_sequence(sequence_path)
    gt_df = gt_df[gt_df["frame"] < 100]
    agent_df = prepare_agent_df(gt_df)

    try:
        plot_ant_trajectories(agent_df=agent_df, width=seqinfo["imWidth"], height=seqinfo["imHeight"], title=f"Trajektorie mrówek - {seqinfo['name']}")
        plot_mean_distance(metrics_per_step_df[["step", "mean_distance"]])
        plot_space_coverage(metrics_per_step_df[["step", "space_coverage"]])
    except Exception as e:
        print(f"Failed to generate plots: {e}")