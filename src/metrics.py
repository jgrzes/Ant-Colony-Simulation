import numpy as np


_EPS = 10


def compute_colony_dispersion(agent_df):
    df = agent_df.copy()
    centroids = (
        df.groupby("step")[["x", "y"]]
        .mean()
        .rename(columns={"x": "cx", "y": "cy"})
        .reset_index()
    )
    df = df.merge(centroids, on="step", how="left")
    df["sq_dist_to_centroid"] = (df["x"] - df["cx"]) ** 2 + (df["y"] - df["cy"]) ** 2
    dispersion_per_step = (
        df.groupby("step")["sq_dist_to_centroid"].mean().reset_index(name="dispersion")
    )

    return dispersion_per_step


def compute_mean_turning_angle(agent_df):
    """
    Średnia zmiana kierunku Δθ = angle(v_t, v_{t+1}) na krok,
    gdzie v_t = x(t) - x(t-1). Zwracana jest średnia |Δθ| po agentach.
    """
    df = agent_df.sort_values(["agent_id", "step"]).copy()
    df["dx"] = df.groupby("agent_id")["x"].diff()
    df["dy"] = df.groupby("agent_id")["y"].diff()
    df["dx_next"] = df.groupby("agent_id")["dx"].shift(-1)
    df["dy_next"] = df.groupby("agent_id")["dy"].shift(-1)

    dot = df["dx"] * df["dx_next"] + df["dy"] * df["dy_next"]
    cross = df["dx"] * df["dy_next"] - df["dy"] * df["dx_next"]
    df["delta_theta"] = np.abs(np.arctan2(cross, dot))

    return (
        df.dropna(subset=["delta_theta"])
        .groupby("step")["delta_theta"]
        .mean()
        .reset_index(name="mean_turning_angle")
    )


def compute_mean_displacement(agent_df):
    """
    Średnie przemieszczenie agenta d̂(t) = (1/N) Σ ||x_i(t) - x_i(0)||.
    Pozycja startowa to pierwsza zarejestrowana klatka danego agenta.
    """
    df = agent_df.sort_values(["agent_id", "step"]).copy()
    starts = (
        df.groupby("agent_id")
        .first()[["x", "y"]]
        .rename(columns={"x": "x0", "y": "y0"})
        .reset_index()
    )
    df = df.merge(starts, on="agent_id", how="left")
    df["displacement"] = np.sqrt((df["x"] - df["x0"]) ** 2 + (df["y"] - df["y0"]) ** 2)
    return (
        df.groupby("step")["displacement"].mean().reset_index(name="mean_displacement")
    )


def compute_sinuosity(agent_df):
    """
    Krętość trajektorii S_i(t) = długość_drogi / dystans_euklidesowy
    od pozycji startowej. Zwracana jest średnia po agentach na krok.
    Krok 0 i agenci nieprzemieszczeni (euclid ≈ 0) są pomijani.
    """
    df = agent_df.sort_values(["agent_id", "step"]).copy()
    df["dx"] = df.groupby("agent_id")["x"].diff().fillna(0.0)
    df["dy"] = df.groupby("agent_id")["y"].diff().fillna(0.0)
    df["step_length"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    df["path_length"] = df.groupby("agent_id")["step_length"].cumsum()

    starts = (
        df.groupby("agent_id")
        .first()[["x", "y"]]
        .rename(columns={"x": "x0", "y": "y0"})
        .reset_index()
    )
    df = df.merge(starts, on="agent_id", how="left")
    df["euclid"] = np.sqrt((df["x"] - df["x0"]) ** 2 + (df["y"] - df["y0"]) ** 2)

    df = df[df["euclid"] > _EPS]
    df["sinuosity"] = df["path_length"] / df["euclid"]
    return df.groupby("step")["sinuosity"].mean().reset_index(name="mean_sinuosity")


def compute_space_coverage(agent_df, width, height, cell_size=1.0):
    df = agent_df.copy()

    df["cell_x"] = (df["x"] // cell_size).astype(int)
    df["cell_y"] = (df["y"] // cell_size).astype(int)

    visited_cells = df[["cell_x", "cell_y"]].drop_duplicates().shape[0]
    total_cells = int((width // cell_size) * (height // cell_size))

    if total_cells == 0:
        return 0.0

    return visited_cells / total_cells
