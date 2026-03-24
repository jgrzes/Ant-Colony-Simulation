import math

import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector

from metrics import compute_distance_from_nest, plot_mean_distance


class AntAgent(Agent):
    def __init__(self, model, step_size=1.0):
        super().__init__(model)
        self.step_size = step_size

    def step(self):
        x, y = self.pos

        angle = self.random.uniform(0, 2 * math.pi)
        dx = math.cos(angle) * self.step_size
        dy = math.sin(angle) * self.step_size

        new_x = x + dx
        new_y = y + dy

        self.model.space.move_agent(self, (new_x, new_y))


class AntModel(Model):
    def __init__(self, n_ants=8, width=40, height=40, seed=None):
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.space = ContinuousSpace(width, height, torus=False)

        self.datacollector = DataCollector(
            model_reporters={
                "num_ants": lambda m: len(m.agents)
            },
            agent_reporters={
                "x": lambda a: a.pos[0],
                "y": lambda a: a.pos[1],
            }
        )

        nest_x = width / 2
        nest_y = height / 2

        for _ in range(n_ants):
            ant = AntAgent(self, step_size=self.random.uniform(0.5, 1.5))

            start_x = nest_x + self.random.uniform(-2, 2)
            start_y = nest_y + self.random.uniform(-2, 2)

            self.space.place_agent(ant, (start_x, start_y))

        self.datacollector.collect(self)

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


def run_demo(steps=50, n_ants=8, seed=42):
    model = AntModel(n_ants=n_ants, width=40, height=40, seed=seed)

    for _ in range(steps):
        model.step()

    agent_df = model.datacollector.get_agent_vars_dataframe().reset_index()
    agent_df.columns = ["step", "agent_id", "x", "y"]

    return model, agent_df


def plot_trajectories(agent_df, width=40, height=40):
    plt.figure(figsize=(8, 8))

    for agent_id, group in agent_df.groupby("agent_id"):
        plt.plot(group["x"], group["y"], marker="o", markersize=2, label=f"Ant {agent_id}")

    plt.scatter([width / 2], [height / 2], marker="x", s=120)
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajektorie mrówek - demo")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    model, agent_df = run_demo(steps=60, n_ants=6, seed=123)

    plot_trajectories(agent_df, width=model.width, height=model.height)
    _, mean_distance_per_step, final_mean_distance = compute_distance_from_nest(
        agent_df, nest_x=model.width / 2, nest_y=model.height / 2
    )

    plot_mean_distance(mean_distance_per_step)