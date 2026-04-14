from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt

import pandas as pd

from metrics import (
    compute_colony_dispersion,
    compute_distance_from_nest,
    compute_space_coverage,
)

from plots import plot_ant_trajectories


def load_seqinfo(seq_dir: str | Path) -> dict[str, Any]:
    seq_dir = Path(seq_dir)
    seqinfo_path = seq_dir / "seqinfo.ini"

    if not seqinfo_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {seqinfo_path}")

    parser = ConfigParser()
    parser.read(seqinfo_path)

    if "Sequence" not in parser:
        raise ValueError(f"Brak sekcji [Sequence] w pliku {seqinfo_path}")

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

    if not gt_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {gt_path}")

    # Typowy przypadek: CSV bez nagłówka
    df = pd.read_csv(gt_path, header=None)

    # frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
    if df.shape[1] >= 6:
        base_cols = ["frame", "ant_id", "bb_left", "bb_top", "bb_width", "bb_height"]
        extra_cols = [f"extra_{i}" for i in range(df.shape[1] - 6)]
        df.columns = base_cols + extra_cols

        df["x"] = df["bb_left"] + df["bb_width"] / 2.0
        df["y"] = df["bb_top"] + df["bb_height"] / 2.0

    elif df.shape[1] == 4:
        df.columns = ["frame", "ant_id", "x", "y"]

    else:
        raise ValueError(
            f"Nieobsługiwany format gt.txt. Liczba kolumn: {df.shape[1]}"
        )

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


def infer_nest_position(agent_df: pd.DataFrame) -> tuple[float, float]:
    first_step = agent_df["step"].min()
    first_positions = agent_df[agent_df["step"] == first_step]
    return float(first_positions["x"].mean()), float(first_positions["y"].mean())


def compute_metrics_for_sequence(seq_dir: str | Path, cell_size: float = 10.0) -> dict[str, Any]:
    seqinfo, gt_df = load_sequence(seq_dir)
    agent_df = prepare_agent_df(gt_df)

    nest_x, nest_y = infer_nest_position(agent_df)

    _, mean_distance_per_step, final_mean_distance = compute_distance_from_nest(
        agent_df,
        nest_x=nest_x,
        nest_y=nest_y,
    )
    dispersion_per_step = compute_colony_dispersion(
        agent_df,
        nest_x=nest_x,
        nest_y=nest_y,
    )
    coverage = compute_space_coverage(
        agent_df,
        width=float(seqinfo["imWidth"] or 0),
        height=float(seqinfo["imHeight"] or 0),
        cell_size=cell_size,
    )

    return {
        "sequence": seqinfo["name"],
        "frames": int(agent_df["step"].nunique()),
        "ants": int(agent_df["agent_id"].nunique()),
        "nest_x": nest_x,
        "nest_y": nest_y,
        "mean_distance_final": float(final_mean_distance),
        "mean_distance_overall": float(mean_distance_per_step["mean_distance"].mean()),
        "dispersion_final": float(dispersion_per_step["dispersion"].iloc[-1]),
        "dispersion_overall": float(dispersion_per_step["dispersion"].mean()),
        "space_coverage": float(coverage),
    }


def list_sequence_dirs(dataset_root: str | Path) -> list[Path]:
    dataset_root = Path(dataset_root)
    return sorted(dataset_root.glob("*Dataset/Seq*"))


def run_metrics_for_dataset(dataset_root: str | Path, cell_size: float = 10.0) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for seq_dir in list_sequence_dirs(dataset_root):
        rows.append(compute_metrics_for_sequence(seq_dir, cell_size=cell_size))

    if not rows:
        raise FileNotFoundError(f"Nie znaleziono sekwencji pod: {dataset_root}")

    return pd.DataFrame(rows).sort_values("sequence").reset_index(drop=True)

if __name__ == "__main__":
    dataset_root = Path(__file__).resolve().parent / "dataset"
    report_dir = Path(__file__).resolve().parent / "Metric Raports"
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = run_metrics_for_dataset(dataset_root, cell_size=10.0)
    report_path = report_dir / "dataset_metrics.csv"
    metrics_df.to_csv(report_path, index=False)

    print("Policzono metryki dla sekwencji:")
    print(metrics_df.to_string(index=False))
    print(f"\nZapisano raport CSV: {report_path}")

    seqinfo, gt_df = load_sequence("dataset/OutdoorDataset/Seq0006Object21Image64")
    gt_df = gt_df[gt_df["frame"] < 100]
    agent_df = prepare_agent_df(gt_df)
    plot_ant_trajectories(agent_df, title=seqinfo["name"])