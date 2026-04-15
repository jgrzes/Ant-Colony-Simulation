import numpy as np


def compute_distance_from_nest(agent_df, nest_x, nest_y):
    df = agent_df.copy()
    df["distance_from_nest"] = np.sqrt(
        (df["x"] - nest_x) ** 2 + (df["y"] - nest_y) ** 2
    )

    mean_distance_per_step = (
        df.groupby("step")["distance_from_nest"]
        .mean()
        .reset_index(name="mean_distance")
    )

    final_mean_distance = mean_distance_per_step["mean_distance"].iloc[-1]
    return df, mean_distance_per_step, final_mean_distance


def compute_colony_dispersion(agent_df, nest_x, nest_y):
    df = agent_df.copy()
    df["distance_from_nest"] = np.sqrt(
        (df["x"] - nest_x) ** 2 + (df["y"] - nest_y) ** 2
    )

    dispersion_per_step = (
        df.groupby("step")["distance_from_nest"].var().reset_index(name="dispersion")
    )

    return dispersion_per_step


def compute_space_coverage(agent_df, width, height, cell_size=1.0):
    df = agent_df.copy()

    df["cell_x"] = (df["x"] // cell_size).astype(int)
    df["cell_y"] = (df["y"] // cell_size).astype(int)

    visited_cells = df[["cell_x", "cell_y"]].drop_duplicates().shape[0]
    total_cells = int((width // cell_size) * (height // cell_size))

    if total_cells == 0:
        return 0.0

    return visited_cells / total_cells
