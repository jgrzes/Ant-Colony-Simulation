import matplotlib.pyplot as plt
from pathlib import Path

# Keep generated plots in the project root, outside src.
PLOT_DIR = Path(__file__).resolve().parents[1] / "Generated Plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def plot_mean_distance(mean_distance_per_step, is_simulation=True, save_path=None):
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
    if save_path is None:
        save_path = PLOT_DIR / "mean_distance_over_time.png"
    plt.savefig(save_path)
    plt.close()


def plot_ant_trajectories(
    agent_df=None,
    width=None,
    height=None,
    pheromone_grid=None,
    title="Ants Trajectories",
    save_path=None,
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

    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True)
    if agent_df is not None or pheromone_grid is not None:
        if len(agent_df["agent_id"].unique()) <= 25:
            plt.legend()
        else:
            plt.title(f"{title} (legend hidden for >25 ants)")
    if save_path is None:
        save_path = PLOT_DIR / f"{title.replace(' ', '_')}.png"
    plt.savefig(save_path)
    plt.close()


def plot_mean_turning_angle(turning_per_step, is_simulation=True, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(
        turning_per_step["step"],
        turning_per_step["mean_turning_angle"],
        marker="o",
        markersize=3,
    )
    plt.xlabel("Krok symulacji" if is_simulation else "Krok sekwencji")
    plt.ylabel("Δθ [rad]")
    plt.title("Średnia zmiana kierunku ruchu")
    plt.grid(True)
    if save_path is None:
        save_path = PLOT_DIR / "mean_turning_angle_over_time.png"
    plt.savefig(save_path)
    plt.close()


def plot_mean_displacement(displacement_per_step, is_simulation=True, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(
        displacement_per_step["step"],
        displacement_per_step["mean_displacement"],
        marker="o",
        markersize=3,
        color="tab:orange",
    )
    plt.xlabel("Krok symulacji" if is_simulation else "Krok sekwencji")
    plt.ylabel("Średnie przemieszczenie")
    plt.title("Średnie przemieszczenie agenta od pozycji startowej")
    plt.grid(True)
    if save_path is None:
        save_path = PLOT_DIR / "mean_displacement_over_time.png"
    plt.savefig(save_path)
    plt.close()


def plot_sinuosity(sinuosity_per_step, is_simulation=True, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(
        sinuosity_per_step["step"],
        sinuosity_per_step["mean_sinuosity"],
        marker="o",
        markersize=3,
        color="tab:green",
    )
    plt.xlabel("Krok symulacji" if is_simulation else "Krok sekwencji")
    plt.ylabel("Krętość S")
    plt.title("Średnia krętość trajektorii agentów")
    plt.grid(True)
    if save_path is None:
        save_path = PLOT_DIR / "sinuosity_over_time.png"
    plt.savefig(save_path)
    plt.close()


def plot_space_coverage(coverage_per_step, is_simulation=True, save_path=None):
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
    if save_path is None:
        save_path = PLOT_DIR / "space_coverage_over_time.png"
    plt.savefig(save_path)
    plt.close()


def plot_colony_dispersion(dispersion_per_step, is_simulation=True, save_path=None):
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
    if save_path is None:
        save_path = PLOT_DIR / "colony_dispersion_over_time.png"
    plt.savefig(save_path)
    plt.close()
