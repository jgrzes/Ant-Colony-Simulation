import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent / "Generated Plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def plot_mean_distance(mean_distance_per_step, is_simulation=True):
    plt.figure(figsize=(8, 5))
    plt.plot(
        mean_distance_per_step["step"],
        mean_distance_per_step["mean_distance"],
        marker="o",
        markersize=3,
    )
    plt.xlabel("Krok symulacji" if is_simulation else "Krok sekwencji")
    plt.ylabel("Średnia odległość od mrowiska")
    plt.title("Zmiana średniej odległości agentów od mrowiska")
    plt.grid(True)
    plt.savefig(PLOT_DIR / "mean_distance_over_time.png")


def plot_ant_trajectories(
    agent_df=None,
    width=None,
    height=None,
    pheromone_grid=None,
    title="Ants Trajectories",
):
    plt.figure(figsize=(8, 8))

    if pheromone_grid is not None:
        plt.imshow(
            pheromone_grid,
            origin="lower",
            extent=(0, width, 0, height),
            cmap="inferno",
            alpha=0.75,
            aspect="auto",
        )
        plt.colorbar(label="Poziom feromonu")

    if agent_df is not None:
        for agent_id, group in agent_df.groupby("agent_id"):
            plt.plot(
                group["x"],
                group["y"],
                marker="o",
                markersize=2,
                label=f"Ant {agent_id}",
            )

    plt.scatter([width / 2], [height / 2], marker="x", s=120, label="Mrowisko")
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True)
    if agent_df is not None or pheromone_grid is not None:
        plt.legend()
    plt.savefig(PLOT_DIR / f"{title.replace(' ', '_')}.png")


def plot_space_coverage(coverage_per_step, is_simulation=True):
    plt.figure(figsize=(8, 5))
    plt.plot(
        coverage_per_step["step"],
        coverage_per_step["space_coverage"],
        marker="o",
        markersize=3,
    )
    plt.xlabel("Krok symulacji" if is_simulation else "Krok sekwencji")
    plt.ylabel("Pokrycie przestrzeni")
    plt.title("Zmiana pokrycia przestrzeni przez agentów")
    plt.grid(True)
    plt.savefig(PLOT_DIR / "space_coverage_over_time.png")


def plot_colony_dispersion(dispersion_per_step, is_simulation=True):
    plt.figure(figsize=(8, 5))
    plt.plot(
        dispersion_per_step["step"],
        dispersion_per_step["dispersion"],
        marker="o",
        markersize=3,
    )
    plt.xlabel("Krok symulacji" if is_simulation else "Krok sekwencji")
    plt.ylabel("Dyspersja kolonii")
    plt.title("Zmiana dyspersji kolonii w czasie")
    plt.grid(True)
    plt.savefig(PLOT_DIR / "colony_dispersion_over_time.png")
