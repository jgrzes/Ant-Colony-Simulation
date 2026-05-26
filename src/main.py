from datetime import datetime
from pathlib import Path
import csv

from agent_model import run_demo, build_step_metrics
from fitter import optuna_fit
from compare import compare_sequence
from plots import (
    plot_ant_trajectories,
    plot_ant_step_by_step_gif,
    plot_space_coverage,
    plot_colony_dispersion,
    plot_mean_displacement,
    plot_sinuosity,
)

from InquirerPy import inquirer


REPORT_METRICS = [
    "dispersion",
    "mean_displacement",
    "space_coverage",
    "mean_sinuosity",
]


def _list_sequences(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        return []

    return [
        path
        for path in sorted(dataset_root.iterdir())
        if path.is_dir() and (path / "seqinfo.ini").exists()
    ]


def _metric_gaps(real_metrics, sim_metrics) -> dict[str, float]:
    merged = real_metrics.merge(sim_metrics, on="step", suffixes=("_real", "_sim"))
    metric_names = [
        metric
        for metric in REPORT_METRICS
        if f"{metric}_real" in merged.columns and f"{metric}_sim" in merged.columns
    ]

    gaps: dict[str, float] = {}
    for metric in metric_names:
        real_col = f"{metric}_real"
        sim_col = f"{metric}_sim"
        gaps[f"mae_{metric}"] = float((merged[real_col] - merged[sim_col]).abs().mean())
        gaps[f"final_real_{metric}"] = float(merged[real_col].iloc[-1])
        gaps[f"final_sim_{metric}"] = float(merged[sim_col].iloc[-1])
    return gaps


def _run_batch_experiment(
    project_root: Path,
    dataset_names: list[str],
    n_trials: int,
    metrics: list[str],
):
    dataset_root = project_root / "dataset"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_tag = "-".join(metrics)
    report_root = (
        project_root
        / "Simulation Metrics Reports"
        / "experiments"
        / f"{run_tag}_{metrics_tag}"
    )
    plots_root = (
        project_root / "Generated Plots" / "experiments" / f"{run_tag}_{metrics_tag}"
    )
    report_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    for dataset_name in dataset_names:
        for sequence_path in _list_sequences(dataset_root / dataset_name):
            sequence_report_dir = report_root / dataset_name / sequence_path.name
            sequence_plot_dir = plots_root / dataset_name / sequence_path.name
            sequence_report_dir.mkdir(parents=True, exist_ok=True)
            sequence_plot_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"\nFitting {dataset_name}/{sequence_path.name} with {n_trials} trials..."
            )
            fit_result = optuna_fit(
                sequence_path,
                n_iter=n_trials,
                metrics=metrics,
                results_dir=sequence_report_dir,
            )
            compare_result = compare_sequence(
                sequence_path,
                fit_result["history_path"],
                output_dir=sequence_plot_dir,
            )

            baseline_loss = fit_result["baseline"]["loss"]
            best_loss = fit_result["best"]["loss"]

            improvement_abs = baseline_loss - best_loss
            improvement_pct = (
                100.0 * improvement_abs / baseline_loss
                if baseline_loss and baseline_loss > 0
                else 0.0
            )

            best_params = fit_result["best"]["params"]
            metric_gaps = _metric_gaps(
                compare_result["real_metrics"], compare_result["sim_metrics"]
            )

            summary_row = {
                "dataset": dataset_name,
                "sequence": sequence_path.name,
                "n_trials": n_trials,
                "metrics": ",".join(metrics),
                "baseline_loss": baseline_loss,
                "best_loss": best_loss,
                "improvement_abs": improvement_abs,
                "improvement_pct": improvement_pct,
                "history_path": str(fit_result["history_path"]),
                "output_dir": str(compare_result["output_dir"]),
                "report_dir": str(sequence_report_dir),
            }

            for param_name, param_value in best_params.items():
                summary_row[f"best_{param_name}"] = param_value

            summary_row.update(metric_gaps)

            summary_rows.append(summary_row)

    summary_path = report_root / "batch_experiment_summary.csv"

    if not summary_rows:
        print("No sequences found for selected dataset(s).")
        return summary_path

    summary_rows = sorted(
        summary_rows,
        key=lambda row: (
            str(row.get("dataset", "")),
            float(row.get("best_loss", float("inf"))),
        ),
    )

    base_fieldnames = [
        "dataset",
        "sequence",
        "n_trials",
        "metrics",
        "baseline_loss",
        "best_loss",
        "improvement_abs",
        "improvement_pct",
        "history_path",
        "output_dir",
        "report_dir",
    ]
    extra_fieldnames = sorted(
        {
            key
            for row in summary_rows
            for key in row.keys()
            if key not in base_fieldnames
        }
    )
    fieldnames = base_fieldnames + extra_fieldnames

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    mode = inquirer.select(
        message="What do you want to run?",
        choices=[
            "Fit a single dataset sequence",
            "Run a batch experiment on all sequences",
            "Run a demo simulation",
        ],
        default="Fit a single dataset sequence",
    ).execute()

    if mode == "Fit a single dataset sequence":
        sequence_path = inquirer.filepath(
            message="Path to dataset sequence folder (with gt/gt.txt):",
            only_directories=True,
        ).execute()

        n_trials = inquirer.number(
            message="Number of optuna trials:",
            default=100,
            min_allowed=1,
        ).execute()

        metrics = [
            "dispersion",
            "mean_displacement",
            "mean_sinuosity",
            "space_coverage",
        ]

        print("\nRunning optuna fit...")
        fit_result = optuna_fit(sequence_path, n_iter=int(n_trials), metrics=metrics)
        baseline_loss = fit_result["baseline"]["loss"]
        best_loss = fit_result["best"]["loss"]
        improvement_abs = baseline_loss - best_loss
        improvement_pct = (
            100.0 * improvement_abs / baseline_loss
            if baseline_loss and baseline_loss > 0
            else 0.0
        )

        print(f"Baseline loss: {baseline_loss}")
        print(f"Best loss: {best_loss}")
        print(f"Improvement: {improvement_abs} ({improvement_pct:.2f}%)")
        print(f"History saved: {fit_result['history_path']}")

        print("\nGenerating comparison plots and GIFs...")
        compare_result = compare_sequence(sequence_path, fit_result["history_path"])
        print("Comparison complete.")
        print(f"Output directory: {compare_result['output_dir']}")
        exit(0)

    if mode == "Run a batch experiment on all sequences":
        dataset_choice = inquirer.select(
            message="Which dataset(s) should be included?",
            choices=["IndoorDataset", "OutdoorDataset", "Both"],
            default="Both",
        ).execute()
        selected_datasets = (
            ["IndoorDataset", "OutdoorDataset"]
            if dataset_choice == "Both"
            else [dataset_choice]
        )

        n_trials = int(
            inquirer.number(
                message="Number of optuna trials per sequence:",
                default=100,
                min_allowed=1,
            ).execute()
        )

        metrics = [
            "dispersion",
            "mean_displacement",
            "mean_sinuosity",
            "space_coverage",
        ]

        print("\nRunning batch experiment...")
        summary_path = _run_batch_experiment(
            project_root,
            selected_datasets,
            n_trials,
            metrics,
        )
        print(f"Batch experiment completed. Summary saved to: {summary_path}")
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
        plot_mean_displacement(step_metrics_df[["step", "mean_displacement"]])
        plot_sinuosity(step_metrics_df[["step", "mean_sinuosity"]])
    except Exception as e:
        print(f"Failed to generate plots: {e}")
    print("Simulation and plotting completed.")
