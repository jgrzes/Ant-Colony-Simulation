from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Keep generated plots in the project root, outside src.
PLOT_DIR = Path(__file__).resolve().parents[1] / "Generated Plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _auto_bucket_size(step_count, target_bins=35):
    if step_count <= 0:
        return 1
    return max(1, int(np.ceil(step_count / target_bins)))


def _bucket_metric_df(metric_df, metric_col, bucket_size):
    df = metric_df[["step", metric_col]].dropna().sort_values("step").copy()
    if df.empty:
        return df, bucket_size

    if bucket_size <= 1:
        df["metric_mean"] = df[metric_col]
        df["metric_std"] = 0.0
        return df[["step", "metric_mean", "metric_std"]], 1

    first_step = float(df["step"].min())
    df["bucket_id"] = ((df["step"] - first_step) // bucket_size).astype(int)
    bucketed = (
        df.groupby("bucket_id")
        .agg(
            step=("step", "mean"),
            metric_mean=(metric_col, "mean"),
            metric_std=(metric_col, "std"),
        )
        .reset_index(drop=True)
    )
    bucketed["metric_std"] = bucketed["metric_std"].fillna(0.0)
    return bucketed, bucket_size


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


def plot_ant_step_by_step_gif(
    agent_df,
    width,
    height,
    pheromone_grid=None,
    title="Ants Step by Step",
    save_path=None,
    frame_interval=180,
    max_frames=120,
):
    """Create an animated GIF that shows how the colony evolves step by step."""

    if save_path is None:
        save_path = PLOT_DIR / f"{title.replace(' ', '_')}.gif"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = agent_df.sort_values(["step", "agent_id"]).copy()
    steps = list(dict.fromkeys(df["step"].tolist()))
    if not steps:
        raise ValueError("agent_df is empty, cannot create animation")

    if max_frames and len(steps) > max_frames:
        step_indices = np.linspace(0, len(steps) - 1, num=max_frames, dtype=int)
        steps = [steps[index] for index in step_indices]

    agent_ids = list(dict.fromkeys(df["agent_id"].tolist()))
    color_map = {
        agent_id: plt.cm.tab20(index % 20) for index, agent_id in enumerate(agent_ids)
    }

    fig, ax = plt.subplots(figsize=(8, 8))

    if pheromone_grid is not None:
        ax.imshow(
            pheromone_grid,
            origin="lower",
            extent=(0, width, 0, height),
            cmap="inferno",
            alpha=0.65,
            aspect="auto",
        )
        fig.colorbar(ax.images[0], ax=ax, label="Poziom feromonu")

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)

    lines = {}
    for agent_id in agent_ids:
        (line,) = ax.plot([], [], lw=1.4, alpha=0.85, color=color_map[agent_id])
        lines[agent_id] = line

    scatter = ax.scatter([], [], s=28)
    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    def init():
        for line in lines.values():
            line.set_data([], [])
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_facecolors([])
        status_text.set_text("")
        return list(lines.values()) + [scatter, status_text]

    def update(step):
        current = df[df["step"] == step]
        past = df[df["step"] <= step]

        current_positions = []
        current_colors = []
        for agent_id, line in lines.items():
            agent_history = past[past["agent_id"] == agent_id]
            line.set_data(agent_history["x"], agent_history["y"])

        for _, row in current.iterrows():
            current_positions.append((row["x"], row["y"]))
            current_colors.append(color_map[row["agent_id"]])

        scatter.set_offsets(np.array(current_positions))
        scatter.set_facecolors(current_colors)
        scatter.set_edgecolors("black")
        status_text.set_text(f"Step: {int(step)} | Ants: {len(current)}")
        ax.set_title(title)
        return list(lines.values()) + [scatter, status_text]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=steps,
        init_func=init,
        interval=frame_interval,
        blit=False,
        repeat=False,
    )
    writer = animation.PillowWriter(fps=max(1, int(1000 / frame_interval)))
    anim.save(save_path, writer=writer)
    plt.close(fig)
    return save_path


def plot_metric_comparison(real_df, sim_df, metric, is_simulation=True, save_path=None):
    """Plot real vs simulated metric values aggregated into step buckets."""
    step_count = max(real_df["step"].nunique(), sim_df["step"].nunique())
    bucket_size = _auto_bucket_size(step_count)
    real_bucketed, bucket_size = _bucket_metric_df(real_df, metric, bucket_size)
    sim_bucketed, _ = _bucket_metric_df(sim_df, metric, bucket_size)
    bar_width = max(0.2, float(bucket_size) * 0.4)

    plt.figure(figsize=(8, 5))

    plt.bar(
        real_bucketed["step"] - bar_width / 2,
        real_bucketed["metric_mean"],
        width=bar_width,
        yerr=real_bucketed["metric_std"] if bucket_size > 1 else None,
        capsize=3 if bucket_size > 1 else 0,
        alpha=0.8,
        label="real",
    )
    plt.bar(
        sim_bucketed["step"] + bar_width / 2,
        sim_bucketed["metric_mean"],
        width=bar_width,
        yerr=sim_bucketed["metric_std"] if bucket_size > 1 else None,
        capsize=3 if bucket_size > 1 else 0,
        alpha=0.8,
        label="sim",
    )

    plt.xlabel("Krok symulacji" if is_simulation else "Krok sekwencji")
    plt.ylabel(metric)
    title = f"Real vs Simulated - {metric}"
    if bucket_size > 1:
        title += f" (bucket={bucket_size} krokow)"
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path is None:
        save_path = PLOT_DIR / f"{metric}_real_vs_sim.png"
    plt.savefig(save_path)
    plt.close()


def plot_fit_history(history_df, save_path=None):
    plt.figure(figsize=(8, 5))
    df = history_df.copy()
    if "iter" in df.columns:
        df = df.sort_values("iter").reset_index(drop=True)
        x = df["iter"]
    else:
        x = df.index
    plt.plot(x, df["loss"], marker="o", markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Fit history")
    plt.grid(True)

    if save_path is None:
        save_path = PLOT_DIR / "fit_history.png"
    plt.savefig(save_path)
    plt.close()


def plot_mean_turning_angle(turning_per_step, is_simulation=True, save_path=None):
    bucket_size = _auto_bucket_size(turning_per_step["step"].nunique())
    turning_bucketed, bucket_size = _bucket_metric_df(
        turning_per_step,
        "mean_turning_angle",
        bucket_size,
    )

    bar_width = max(0.2, float(bucket_size) * 0.8)
    plt.figure(figsize=(8, 5))
    plt.bar(
        turning_bucketed["step"],
        turning_bucketed["metric_mean"],
        width=bar_width,
        yerr=turning_bucketed["metric_std"] if bucket_size > 1 else None,
        capsize=3 if bucket_size > 1 else 0,
        alpha=0.85,
    )
    plt.xlabel("Krok symulacji" if is_simulation else "Krok sekwencji")
    plt.ylabel("Δθ [rad]")
    title = "Średnia zmiana kierunku ruchu"
    if bucket_size > 1:
        title += f" (bucket={bucket_size} krokow)"
    plt.title(title)
    plt.grid(True)
    if save_path is None:
        save_path = PLOT_DIR / "mean_turning_angle_over_time.png"
    plt.savefig(save_path)
    plt.close()


def plot_mean_displacement(displacement_per_step, is_simulation=True, save_path=None):
    bucket_size = _auto_bucket_size(displacement_per_step["step"].nunique())
    displacement_bucketed, bucket_size = _bucket_metric_df(
        displacement_per_step,
        "mean_displacement",
        bucket_size,
    )

    bar_width = max(0.2, float(bucket_size) * 0.8)
    plt.figure(figsize=(8, 5))
    plt.bar(
        displacement_bucketed["step"],
        displacement_bucketed["metric_mean"],
        width=bar_width,
        yerr=displacement_bucketed["metric_std"] if bucket_size > 1 else None,
        capsize=3 if bucket_size > 1 else 0,
        color="tab:orange",
        alpha=0.85,
    )
    plt.xlabel("Krok symulacji" if is_simulation else "Krok sekwencji")
    plt.ylabel("Średnie przemieszczenie")
    title = "Średnie przemieszczenie agenta od pozycji startowej"
    if bucket_size > 1:
        title += f" (bucket={bucket_size} krokow)"
    plt.title(title)
    plt.grid(True)
    if save_path is None:
        save_path = PLOT_DIR / "mean_displacement_over_time.png"
    plt.savefig(save_path)
    plt.close()


def plot_sinuosity(sinuosity_per_step, is_simulation=True, save_path=None):
    bucket_size = _auto_bucket_size(sinuosity_per_step["step"].nunique())
    sinuosity_bucketed, bucket_size = _bucket_metric_df(
        sinuosity_per_step,
        "mean_sinuosity",
        bucket_size,
    )

    bar_width = max(0.2, float(bucket_size) * 0.8)
    plt.figure(figsize=(8, 5))
    plt.bar(
        sinuosity_bucketed["step"],
        sinuosity_bucketed["metric_mean"],
        width=bar_width,
        yerr=sinuosity_bucketed["metric_std"] if bucket_size > 1 else None,
        capsize=3 if bucket_size > 1 else 0,
        color="tab:green",
        alpha=0.85,
    )
    plt.xlabel("Krok symulacji" if is_simulation else "Krok sekwencji")
    plt.ylabel("Krętość S")
    title = "Średnia krętość trajektorii agentów"
    if bucket_size > 1:
        title += f" (bucket={bucket_size} krokow)"
    plt.title(title)
    plt.grid(True)
    if save_path is None:
        save_path = PLOT_DIR / "sinuosity_over_time.png"
    plt.savefig(save_path)
    plt.close()


def plot_space_coverage(coverage_per_step, is_simulation=True, save_path=None):
    bucket_size = _auto_bucket_size(coverage_per_step["step"].nunique())
    coverage_bucketed, bucket_size = _bucket_metric_df(
        coverage_per_step,
        "space_coverage",
        bucket_size,
    )

    bar_width = max(0.2, float(bucket_size) * 0.8)
    plt.figure(figsize=(8, 5))
    plt.bar(
        coverage_bucketed["step"],
        coverage_bucketed["metric_mean"],
        width=bar_width,
        yerr=coverage_bucketed["metric_std"] if bucket_size > 1 else None,
        capsize=3 if bucket_size > 1 else 0,
        alpha=0.85,
    )
    plt.xlabel("Krok symulacji" if is_simulation else "Krok sekwencji")
    plt.ylabel("Pokrycie przestrzeni")
    title = "Zmiana pokrycia przestrzeni przez agentów"
    if bucket_size > 1:
        title += f" (bucket={bucket_size} krokow)"
    plt.title(title)
    plt.grid(True)
    if save_path is None:
        save_path = PLOT_DIR / "space_coverage_over_time.png"
    plt.savefig(save_path)
    plt.close()


def plot_colony_dispersion(dispersion_per_step, is_simulation=True, save_path=None):
    bucket_size = _auto_bucket_size(dispersion_per_step["step"].nunique())
    dispersion_bucketed, bucket_size = _bucket_metric_df(
        dispersion_per_step,
        "dispersion",
        bucket_size,
    )

    bar_width = max(0.2, float(bucket_size) * 0.8)
    plt.figure(figsize=(8, 5))
    plt.bar(
        dispersion_bucketed["step"],
        dispersion_bucketed["metric_mean"],
        width=bar_width,
        yerr=dispersion_bucketed["metric_std"] if bucket_size > 1 else None,
        capsize=3 if bucket_size > 1 else 0,
        alpha=0.85,
    )
    plt.xlabel("Krok symulacji" if is_simulation else "Krok sekwencji")
    plt.ylabel("Dyspersja kolonii")
    title = "Zmiana dyspersji kolonii w czasie"
    if bucket_size > 1:
        title += f" (bucket={bucket_size} krokow)"
    plt.title(title)
    plt.grid(True)
    if save_path is None:
        save_path = PLOT_DIR / "colony_dispersion_over_time.png"
    plt.savefig(save_path)
    plt.close()
