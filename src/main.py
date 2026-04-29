from pathlib import Path

from agent_model import run_demo, build_step_metrics
from plots import (
    plot_mean_distance,
    plot_ant_trajectories,
    plot_space_coverage,
    plot_colony_dispersion,
    plot_mean_turning_angle,
    plot_mean_displacement,
    plot_sinuosity,
)

from InquirerPy import inquirer


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    n_ants = int(
        inquirer.number(
            message="Number of ants:",
            default=20,
            min_allowed=1,
        ).execute()
    )
    steps = int(
        inquirer.number(
            message="Number of steps:",
            default=100,
            min_allowed=1,
        ).execute()
    )

    model, agent_df = run_demo(steps=steps, n_ants=n_ants)

    step_metrics_df = build_step_metrics(
        agent_df,
        nest_x=model.nest_x,
        nest_y=model.nest_y,
        width=model.width,
        height=model.height,
        cell_size=1.0,
    )

    report_dir = project_root / "Simulation Metrics Reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"sim_s{steps}_a{n_ants}"

    step_report_path = report_dir / f"{run_tag}_metrics_per_step.csv"
    step_metrics_df.to_csv(step_report_path, index=False)

    try:
        plot_ant_trajectories(
            agent_df,
            width=model.width,
            height=model.height,
            pheromone_grid=model.pheromone_grid,
        )
        plot_mean_distance(step_metrics_df[["step", "mean_distance"]])
        plot_colony_dispersion(step_metrics_df[["step", "dispersion"]])
        plot_space_coverage(step_metrics_df[["step", "space_coverage"]])
        plot_mean_turning_angle(step_metrics_df[["step", "mean_turning_angle"]])
        plot_mean_displacement(step_metrics_df[["step", "mean_displacement"]])
        plot_sinuosity(step_metrics_df[["step", "mean_sinuosity"]])
    except Exception as e:
        print(f"Failed to generate plots: {e}")
    print("Simulation and plotting completed.")
