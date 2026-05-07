import math
from collections import deque

import numpy as np
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace

from metrics import (
    compute_colony_dispersion,
    compute_mean_turning_angle,
    compute_mean_displacement,
    compute_sinuosity,
)


class AntAgent(Agent):
    def __init__(self, model, step_size=1.0):
        super().__init__(model)
        self.step_size = step_size
        self.heading = self.random.uniform(0, 2 * math.pi)
        # Bufor pozycji do opóźnionej depozycji feromonu - mrówka nie podąża
        # za własnym świeżym śladem (anty-self-trapping).
        self._deposit_queue: deque = deque(maxlen=model.pheromone_delay + 1)

    def _reflect_position(self, x, y):
        # Odbicie od ścian z odpowiednim odwróceniem kierunku zamiast
        # przycinania, które powodowało "klejenie się" mrówek do brzegu.
        w = self.model.width - 1e-6
        h = self.model.height - 1e-6

        if x < 0.0:
            x = -x
            self.heading = math.pi - self.heading
        elif x > w:
            x = 2.0 * w - x
            self.heading = math.pi - self.heading

        if y < 0.0:
            y = -y
            self.heading = -self.heading
        elif y > h:
            y = 2.0 * h - y
            self.heading = -self.heading

        # Zabezpieczenie na wypadek bardzo dużego kroku.
        x = min(max(x, 0.0), w)
        y = min(max(y, 0.0), h)
        return x, y

    def _sample_pheromone(self, angle_offset):
        """
        Odczyt feromonu w punkcie przesuniętym względem aktualnego kierunku.
        angle_offset < 0 -> prawa strona
        angle_offset > 0 -> lewa strona
        Punkty próbkowania poza planszą zwracają 0.
        """
        distance = self.model.sensor_distance
        sample_angle = self.heading + angle_offset
        sx = self.pos[0] + math.cos(sample_angle) * distance
        sy = self.pos[1] + math.sin(sample_angle) * distance

        if not (0.0 <= sx < self.model.width and 0.0 <= sy < self.model.height):
            return 0.0

        cell_x = int(sx)
        cell_y = int(sy)
        return self.model.pheromone_grid[cell_y, cell_x]

    def _deposit_pheromone(self):
        # Depozycja jest opóźniona o `pheromone_delay` kroków - dopiero gdy bufor
        # zapełni się, najstarsza zapamiętana pozycja zasila siatkę feromonu.
        self._deposit_queue.append(self.pos)
        if len(self._deposit_queue) <= self.model.pheromone_delay:
            return

        x, y = self._deposit_queue[0]
        cell_x = int(x)
        cell_y = int(y)
        self.model.pheromone_grid[cell_y, cell_x] += self.model.pheromone_deposit

    def step(self):
        # Zostaw feromon w aktualnej pozycji
        self._deposit_pheromone()

        sensor_angle = self.model.sensor_angle
        left_pheromone = self._sample_pheromone(angle_offset=sensor_angle)
        right_pheromone = self._sample_pheromone(angle_offset=-sensor_angle)

        # Reguła Webera: skręt zależy od (L - R) / (L + R).
        denom = left_pheromone + right_pheromone + 1e-6
        pheromone_bias = self.model.turn_strength * (
            (left_pheromone - right_pheromone) / denom
        )

        # Mały losowy szum, żeby ruch nie był całkiem deterministyczny
        noise = self.random.uniform(
            -self.model.noise_strength, self.model.noise_strength
        )

        # Aktualizacja kierunku
        self.heading += pheromone_bias + noise

        # Ruch
        dx = math.cos(self.heading) * self.step_size
        dy = math.sin(self.heading) * self.step_size

        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy
        new_x, new_y = self._reflect_position(new_x, new_y)

        self.model.space.move_agent(self, (new_x, new_y))


class AntModel(Model):
    def __init__(
        self,
        n_ants=8,
        width=40,
        height=40,
        rng=None,
        pheromone_deposit=1.0,
        evaporation_rate=0.02,
        turn_strength=0.6,
        noise_strength=0.25,
        sensor_distance=2.0,
        sensor_angle=math.pi / 4,
        pheromone_delay=3,
        initial_positions=None,
        **kwargs,
    ):
        super().__init__(rng=rng)

        self.width = width
        self.height = height
        self.space = ContinuousSpace(width, height, torus=False)

        self.pheromone_deposit = pheromone_deposit
        self.evaporation_rate = evaporation_rate
        self.turn_strength = turn_strength
        self.noise_strength = noise_strength
        self.sensor_distance = sensor_distance
        self.sensor_angle = sensor_angle
        self.pheromone_delay = pheromone_delay

        # Siatka feromonu: [y, x]
        self.pheromone_grid = np.zeros((height, width), dtype=float)

        self.datacollector = DataCollector(
            model_reporters={
                "num_ants": lambda m: len(m.agents),
            },
            agent_reporters={
                "x": lambda a: a.pos[0],
                "y": lambda a: a.pos[1],
                "heading": lambda a: a.heading,
                "step_size": lambda a: a.step_size,
            },
        )

        # If `initial_positions` provided, use them (one per ant, truncated/padded as needed).
        init_pos_list = []
        if initial_positions is not None:
            init_pos_list = list(initial_positions)

        for i in range(n_ants):
            ant = AntAgent(self, step_size=self.random.uniform(0.5, 1.5))

            if i < len(init_pos_list):
                start_x, start_y = float(init_pos_list[i][0]), float(
                    init_pos_list[i][1]
                )
            else:
                start_x = self.random.uniform(0.0, width - 1e-6)
                start_y = self.random.uniform(0.0, height - 1e-6)

            start_x = min(max(start_x, 0.0), width - 1e-6)
            start_y = min(max(start_y, 0.0), height - 1e-6)

            self.space.place_agent(ant, (start_x, start_y))

        self.datacollector.collect(self)

    def evaporate_pheromones(self):
        self.pheromone_grid *= 1.0 - self.evaporation_rate

    def step(self):
        self.agents.shuffle_do("step")
        self.evaporate_pheromones()
        self.datacollector.collect(self)


def run_demo(
    steps=80,
    n_ants=20,
    rng=42,
    width=40,
    height=40,
    initial_positions=None,
    **model_kwargs,
):

    model = AntModel(
        n_ants=n_ants,
        width=width,
        height=height,
        rng=rng,
        initial_positions=initial_positions,
        **model_kwargs,
    )

    for _ in range(steps):
        model.step()

    agent_df = model.datacollector.get_agent_vars_dataframe().reset_index()
    agent_df.columns = ["step", "agent_id", "x", "y", "heading", "step_size"]

    return model, agent_df


def build_step_metrics(agent_df, width, height, cell_size=1.0):
    dispersion_df = compute_colony_dispersion(agent_df)

    step_metrics_df = dispersion_df.copy()

    turning_df = compute_mean_turning_angle(agent_df)
    displacement_df = compute_mean_displacement(agent_df)
    sinuosity_df = compute_sinuosity(agent_df)
    for extra in (turning_df, displacement_df, sinuosity_df):
        step_metrics_df = step_metrics_df.merge(extra, on="step", how="left")

    cells_df = agent_df.copy()
    cells_df["cell_x"] = (cells_df["x"] // cell_size).astype(int)
    cells_df["cell_y"] = (cells_df["y"] // cell_size).astype(int)

    total_cells = int((width // cell_size) * (height // cell_size))
    visited_cells = set()
    coverage_by_step = {}

    for step in sorted(cells_df["step"].unique()):
        step_cells = cells_df.loc[
            cells_df["step"] == step, ["cell_x", "cell_y"]
        ].drop_duplicates()
        visited_cells.update(map(tuple, step_cells.to_numpy()))
        coverage_by_step[int(step)] = (
            0.0 if total_cells == 0 else len(visited_cells) / total_cells
        )

    step_metrics_df["space_coverage"] = (
        step_metrics_df["step"].map(coverage_by_step).fillna(0.0)
    )
    step_metrics_df["ants"] = (
        agent_df.groupby("step")["agent_id"]
        .nunique()
        .reindex(step_metrics_df["step"])
        .to_numpy()
    )
    step_metrics_df = step_metrics_df.sort_values("step").reset_index(drop=True)

    return step_metrics_df
