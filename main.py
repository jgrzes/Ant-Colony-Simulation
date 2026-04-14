import math

import matplotlib.pyplot as plt
import numpy as np
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace

from metrics import (
    compute_distance_from_nest,
    compute_colony_dispersion,
    compute_space_coverage,
)

from plots import plot_mean_distance, plot_trajectories

class AntAgent(Agent):
    def __init__(self, model, step_size=1.0):
        super().__init__(model)
        self.step_size = step_size
        self.heading = self.random.uniform(0, 2 * math.pi)

    def _clip_position(self, x, y):
        x = min(max(x, 0.0), self.model.width - 1e-6)
        y = min(max(y, 0.0), self.model.height - 1e-6)
        return x, y

    def _sample_pheromone(self, angle_offset, distance=2.0):
        """
        Odczyt feromonu w punkcie przesuniętym względem aktualnego kierunku.
        angle_offset < 0 -> prawa strona
        angle_offset > 0 -> lewa strona
        """
        sample_angle = self.heading + angle_offset
        sx = self.pos[0] + math.cos(sample_angle) * distance
        sy = self.pos[1] + math.sin(sample_angle) * distance

        sx, sy = self._clip_position(sx, sy)

        cell_x = int(sx)
        cell_y = int(sy)
        return self.model.pheromone_grid[cell_y, cell_x]

    def _deposit_pheromone(self):
        x, y = self.pos
        cell_x = int(x)
        cell_y = int(y)
        self.model.pheromone_grid[cell_y, cell_x] += self.model.pheromone_deposit

    def step(self):
        # Zostaw feromon w aktualnej pozycji
        self._deposit_pheromone()

        # Odczyt feromonu po lewej i prawej stronie
        left_pheromone = self._sample_pheromone(angle_offset=math.pi / 4)
        right_pheromone = self._sample_pheromone(angle_offset=-math.pi / 4)

        # Reguła podobna do Webera:
        # skręt zależy od różnicy / sumy
        denom = left_pheromone + right_pheromone + 1e-6
        pheromone_bias = self.model.turn_strength * (
            (left_pheromone - right_pheromone) / denom
        )

        # Mały losowy szum, żeby ruch nie był całkiem deterministyczny
        noise = self.random.uniform(-self.model.noise_strength, self.model.noise_strength)

        # Aktualizacja kierunku
        self.heading += pheromone_bias + noise

        # Ruch
        dx = math.cos(self.heading) * self.step_size
        dy = math.sin(self.heading) * self.step_size

        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy
        new_x, new_y = self._clip_position(new_x, new_y)

        self.model.space.move_agent(self, (new_x, new_y))


class AntModel(Model):
    def __init__(
        self,
        n_ants=8,
        width=40,
        height=40,
        seed=None,
        pheromone_deposit=1.0,
        evaporation_rate=0.02,
        turn_strength=0.6,
        noise_strength=0.25,
    ):
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.space = ContinuousSpace(width, height, torus=False)

        self.nest_x = width / 2
        self.nest_y = height / 2

        self.pheromone_deposit = pheromone_deposit
        self.evaporation_rate = evaporation_rate
        self.turn_strength = turn_strength
        self.noise_strength = noise_strength

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

        for _ in range(n_ants):
            ant = AntAgent(self, step_size=self.random.uniform(0.5, 1.5))

            start_x = self.nest_x + self.random.uniform(-2, 2)
            start_y = self.nest_y + self.random.uniform(-2, 2)

            start_x = min(max(start_x, 0.0), width - 1e-6)
            start_y = min(max(start_y, 0.0), height - 1e-6)

            self.space.place_agent(ant, (start_x, start_y))

        self.datacollector.collect(self)

    def evaporate_pheromones(self):
        self.pheromone_grid *= (1.0 - self.evaporation_rate)

    def step(self):
        self.agents.shuffle_do("step")
        self.evaporate_pheromones()
        self.datacollector.collect(self)


def run_demo(steps=80, n_ants=20, seed=42):
    model = AntModel(
        n_ants=n_ants,
        width=40,
        height=40,
        seed=seed,
        pheromone_deposit=1.0,
        evaporation_rate=0.02,
        turn_strength=0.6,
        noise_strength=0.25,
    )

    for _ in range(steps):
        model.step()

    agent_df = model.datacollector.get_agent_vars_dataframe().reset_index()
    agent_df.columns = ["step", "agent_id", "x", "y", "heading", "step_size"]

    return model, agent_df


if __name__ == "__main__":
    model, agent_df = run_demo(steps=100, n_ants=20, seed=123)

    plot_trajectories(
        agent_df,
        width=model.width,
        height=model.height,
        pheromone_grid=model.pheromone_grid,
    )

    _, mean_distance_per_step, final_mean_distance = compute_distance_from_nest(
        agent_df,
        nest_x=model.nest_x,
        nest_y=model.nest_y,
    )
    plot_mean_distance(mean_distance_per_step)

    dispersion_df = compute_colony_dispersion(
        agent_df,
        nest_x=model.nest_x,
        nest_y=model.nest_y,
    )
    print(dispersion_df.head())

    coverage = compute_space_coverage(agent_df, width=model.width, height=model.height, cell_size=1.0)
    print(f"Pokrycie przestrzeni: {coverage:.3f}")