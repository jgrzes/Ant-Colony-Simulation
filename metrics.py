import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def plot_mean_distance(mean_distance_per_step):
    plt.figure(figsize=(8, 5))
    plt.plot(
        mean_distance_per_step["step"],
        mean_distance_per_step["mean_distance"],
        marker="o",
        markersize=3,
    )
    plt.xlabel("Krok symulacji")
    plt.ylabel("Średnia odległość od mrowiska")
    plt.title("Zmiana średniej odległości agentów od mrowiska")
    plt.grid(True)
    plt.show()