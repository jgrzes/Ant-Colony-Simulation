# Ant Colony Simulation
A Python-based simulation visualizing the complex emergent behaviors of ant colonies.

Getting Started
1. Environment Setup
Prepare your local environment and install necessary dependencies by running the setup script:

```bash
./setup.sh
```

2. Running the Simulation
Once the environment is ready, launch the main simulation using:

```python
python3 src/main.py
```

Then choose one of three modes:

- fit a single dataset sequence and compare the simulation against the tracked data,
- run a batch experiment across all sequences in `IndoorDataset`, `OutdoorDataset`, or both,
- run a demo simulation with randomly initialized ants.

The fitting mode now defaults to 100 Optuna trials so you can explore a larger search budget without changing the code.
