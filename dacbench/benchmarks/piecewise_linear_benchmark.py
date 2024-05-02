import numpy as np
from dacbench import AbstractBenchmark
from dacbench.abstract_benchmark import objdict
from dacbench.envs import PiecewiseLinearEnv
from matplotlib import pyplot as plt
import logging

DEFAULT_DIM = 5


# build random but seeded default instance set
# for our experiments we used a fixed instance, generating train instances with seed 0 and test instances with seed 1
DEFAULT_INSTANCE_SET = {}
instance_set_rng = np.random.RandomState(0)
for i in range(300):
    inter_x = instance_set_rng.random() * 9
    inter_y = instance_set_rng.random()
    grow = instance_set_rng.choice([True, False])
    DEFAULT_INSTANCE_SET[i] = (inter_x, inter_y, grow)

PIECEWISE_LINEAR_DEFAULTS = {
    "action_space_class": "MultiDiscrete",
    "action_space_args": [[3 for _ in range(DEFAULT_DIM)]],
    "observation_space_class": "Box",
    "observation_space_type": np.float32,
    "observation_space_args": [np.full(DEFAULT_DIM + 4, -np.inf),
                               np.full(DEFAULT_DIM + 4, np.inf)],
    "reward_range": (0, 1),
    "cutoff": 10,
    "action_values": [3 for _ in range(DEFAULT_DIM)],
    "seed": 0,
    "multi_agent": False,
    "default_action": [1 for _ in range(DEFAULT_DIM)],
    "reward_shape": "exponential",
    "exp_reward": 4.6,
    "dim_importances": [0.5**i for i in range(DEFAULT_DIM)],
    "reverse_agents": False,
    # "instance_set": DEFAULT_INSTANCE_SET,
    "benchmark_info": "Piecewise Linear Benchmark",
}


class PiecewiseLinearBenchmark(AbstractBenchmark):
    def __init__(self, config_path=None, config: objdict = None):
        """
        Specifies prediction task on piecewise linear functions.
        Problem instances are defined as tuples of the form (inter_x, inter_y, grow), where inter_x, inter_y are x- and y-coordinate
        of the intermediate point, and grow is a boolean indicating whether the function grows or shrinks through the intermediate point.
        """
        super(PiecewiseLinearBenchmark, self).__init__(config_path, config)
        if not self.config:
            self.config = objdict(PIECEWISE_LINEAR_DEFAULTS.copy())

        if not hasattr(self.config, "instance_set"):
            self.read_instance_set()
        if not hasattr(self.config, "test_instance_set"):
            self.read_instance_set(test=True)

    def set_action_values(self, values, dim_importances: list=None) -> None:
        """
        Set action values for the environment. Note that the number of action values will determine action space dimensionality.
        Updates the action values and related configurations (e.g. relating to action and observation space and the dimension importances)
        Parameters
        ----------
        values : list
            List of action values per dimension.    
        """
        self.config.action_values = values
        self.config.action_space_args = [values]
        self.config.observation_space_args = [
            np.full(4 + len(values), -np.inf),
            np.full(4 + len(values), np.inf)
        ]
        self.config.default_action = [v // 2 for v in values]
        if dim_importances is not None:
            if len(dim_importances) != len(values):
                raise ValueError("Dimension importances must have same length as action values.")
            self.config.dim_importances = dim_importances
        else:
            self.config.dim_importances = [self.config.dim_importances[0]**i for i in range(len(values))]

        return

    def get_environment(self) -> PiecewiseLinearEnv:
        """
        Return a new instance of the PiecewiseLinearEnv environment, according to current config.

        Returns
        -------
        PiecewiseLinearEnv
            Environment instance
        """
        # check whether an instance_set is part of the config
        if not hasattr(self.config, "instance_set"):
            raise ValueError("No instance set provided in config and loading an instance set is not implemented yet.")
        return PiecewiseLinearEnv(self.config)

    def render_instances(self):
        """
        Renders all the problem instances of the benchmark.
        """
        cols = 20
        rows = len(self.config.instance_set) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, rows))

        for i, (inter_x, inter_y, grow) in self.config.instance_set.items():
            y_start, y_end = (0, 1) if grow else (1, 0) 
            ax = axes[i // cols, i % cols]
            ax.plot([0, inter_x, 9], [y_start, inter_y, y_end], "k-")
            # ax.set_title(f"Instance {i}")
            ax.set_xlim(0, 9)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_aspect("equal")
        fig.suptitle("Visualization of Piecewise Linear Benchmark Instances", fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.98))

    def read_instance_set(self, test: bool = False) -> None:
        """
        Read instance set from csv of shape (instance_id, inter_x, inter_y, grow).
        """

        if not test:
            if not hasattr(self.config, "instance_set_path"):
                raise ValueError("No instance set path provided in config.")
                return
            path = self.config.instance_set_path
        else:
            if not hasattr(self.config, "test_instance_set_path"):
                # warn that there is no test instance set path
                logging.warning("No test instance set path provided in config.")
                return
            path = self.config.test_instance_set_path

        instance_set = {}

        with open(path, "r") as f:
            for line in f:
                instance_id, inter_x, inter_y, grow = line.strip().split(",")
                # parse the grow strings to boolean
                grow = grow.lower() == "true"
                instance_set[int(instance_id)] = (float(inter_x), float(inter_y), grow)

        if not test:
            self.config.instance_set = instance_set
        else:
            self.config.test_instance_set = instance_set

        return
