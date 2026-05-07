from pathlib import Path

from agent_model import run_demo, build_step_metrics
from fitter import optuna_fit
from compare import compare_sequence
from plots import (
    plot_ant_trajectories,
    plot_ant_step_by_step_gif,
    plot_space_coverage,
    plot_colony_dispersion,
    plot_mean_turning_angle,
    plot_mean_displacement,
    plot_sinuosity,
)

from InquirerPy import inquirer


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    compare = inquirer.confirm(
        message="Fit a simulation to a real dataset sequence and compare it? (if No, runs a demo simulation with default parameters)",
        default=True,
    ).execute()

    if compare:
        sequence_path = inquirer.filepath(
            message="Path to dataset sequence folder (with gt/gt.txt):",
            only_directories=True,
        ).execute()

        n_trials = inquirer.number(
            message="Number of optuna trials:",
            default=50,
            min_allowed=1,
        ).execute()

        print("\nRunning optuna fit...")
        fit_result = optuna_fit(sequence_path, n_iter=int(n_trials))
        print(f"Best loss: {fit_result['best']['loss']}")
        print(f"History saved: {fit_result['history_path']}")

        print("\nGenerating comparison plots and GIFs...")
        compare_result = compare_sequence(sequence_path, fit_result["history_path"])
        print("Comparison complete.")
        print(f"Output directory: {compare_result['output_dir']}")
        exit(0)

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
    plot_gif = inquirer.confirm(
        message="Generate step-by-step GIF of ant trajectories?",
        default=True,
    ).execute()

    model, agent_df = run_demo(steps=steps, n_ants=n_ants)

    step_metrics_df = build_step_metrics(
        agent_df,
        width=model.width,
        height=model.height,
        cell_size=10.0,
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
        if plot_gif:
            plot_ant_step_by_step_gif(
                agent_df,
                width=model.width,
                height=model.height,
                pheromone_grid=model.pheromone_grid,
            )
        plot_colony_dispersion(step_metrics_df[["step", "dispersion"]])
        plot_space_coverage(step_metrics_df[["step", "space_coverage"]])
        plot_mean_turning_angle(step_metrics_df[["step", "mean_turning_angle"]])
        plot_mean_displacement(step_metrics_df[["step", "mean_displacement"]])
        plot_sinuosity(step_metrics_df[["step", "mean_sinuosity"]])
    except Exception as e:
        print(f"Failed to generate plots: {e}")
    print("Simulation and plotting completed.")
