import matplotlib.pyplot as plt


def _resolve_plot_bounds(agent_df=None, width=None, height=None, pheromone_grid=None):
    if width is None:
        if pheromone_grid is not None:
            width = pheromone_grid.shape[1]
        elif agent_df is not None and not agent_df.empty:
            width = max(float(agent_df["x"].max()) + 1.0, 1.0)
        else:
            width = 40

    if height is None:
        if pheromone_grid is not None:
            height = pheromone_grid.shape[0]
        elif agent_df is not None and not agent_df.empty:
            height = max(float(agent_df["y"].max()) + 1.0, 1.0)
        else:
            height = 40

    return width, height


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


def plot_ant_trajectories(
    agent_df=None,
    width=None,
    height=None,
    pheromone_grid=None,
    title="Trajektorie mrówek - model feromonowy",
):
    width, height = _resolve_plot_bounds(agent_df=agent_df, width=width, height=height, pheromone_grid=pheromone_grid)
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
    plt.show()


def plot_pheromone_map(pheromone_grid, width=None, height=None, title="Mapa feromonów"):
    width, height = _resolve_plot_bounds(width=width, height=height, pheromone_grid=pheromone_grid)

    plot_ant_trajectories(
        agent_df=None,
        width=width,
        height=height,
        pheromone_grid=pheromone_grid,
        title=title,
    )


def plot_trajectories(
    agent_df,
    width=40,
    height=40,
    pheromone_grid=None,
    title="Trajektorie mrówek - model feromonowy",
):
    plot_ant_trajectories(
        agent_df=agent_df,
        width=width,
        height=height,
        pheromone_grid=pheromone_grid,
        title=title,
    )