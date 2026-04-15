#!/bin/bash

PLOT_DIR="Generated Plots"
DATASET_REPORT_DIR="Dataset Metrics Reports"
SIMULATION_REPORT_DIR="Simulation Metrics Reports"

if [ -d "$DATASET_REPORT_DIR" ]; then
    rm -rf "$DATASET_REPORT_DIR"
    echo "Dataset metric raports have been removed."
else
    echo "Directory '$DATASET_REPORT_DIR' does not exist. No dataset metric raports to remove."
fi

if [ -d "$SIMULATION_REPORT_DIR" ]; then
    rm -rf "$SIMULATION_REPORT_DIR"
    echo "Simulation metric raports have been removed."
else
    echo "Directory '$SIMULATION_REPORT_DIR' does not exist. No simulation metric raports to remove."
fi

if [ -d "$PLOT_DIR" ]; then
    rm -rf "$PLOT_DIR"
    echo "Generated plots have been removed."
else
    echo "Directory '$PLOT_DIR' does not exist. No generated plots to remove."
fi

echo "Generated plots and metric raports have been removed."
