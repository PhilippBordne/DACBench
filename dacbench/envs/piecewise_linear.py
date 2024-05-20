from dacbench import AbstractMADACEnv
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class PiecewiseLinearEnv(AbstractMADACEnv):
    def __init__(self, config):
        super(PiecewiseLinearEnv, self).__init__(config)

        self.n_actions = len(self.action_space.nvec)

        self.last_action = None

        if "reward_function" in config.keys():
            self.get_reward = config["reward_function"]
        else:
            self.get_reward = self.get_default_reward

        if self.config.get("reward_shape") == "exponential":
            self.get_reward = self.get_exponential_reward
            self.exp_reward = config["exp_reward"]

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        else:
            self.get_state = self.get_default_state

        self.dim_importances = config.dim_importances
        self.reverse_agents = config.reverse_agents

    def step(self, action: np.ndarray):
        """
        Execute environment step.

        Parameters
        ----------
        action : np.ndarray
            action to execute

        Returns
        -------
        np.array, float, bool, bool, dict
            state, reward, terminated, truncated, info
        """

        self.last_action = action
        self.episode_actions[:, self.c_step] = action
        reward = self.get_reward()
        self.done = super(PiecewiseLinearEnv, self).step_()
        next_state = self.get_state(self)

        return next_state, reward, False, self.done, {}

    def get_default_state(self, _):
        """
        Determine the current state of the environment.
        """
        remaining_budget = np.array([self.n_steps - self.c_step])
        next_state = np.concatenate([remaining_budget, np.array(self.instance), self.last_action])

        return next_state

    def reset(self, seed=None, instance_id=None, options={}) -> Tuple[np.ndarray, dict]:
        """
        Resets env.

        Returns
        -------
        numpy.array
            Environment state

        """
        super(PiecewiseLinearEnv, self).reset_(seed, instance_id=instance_id)

        self.done = False
        self.inter_x = self.instance[0]
        self.inter_y = self.instance[1]
        self.grow = self.instance[2]
        self.left_y, self.right_y = (0, 1) if self.grow else (1, 0)
        self.last_action = - np.ones(self.n_actions, dtype=int)

        # track all actions taken within an episode
        self.episode_actions = np.full((self.n_actions, self.n_steps), np.nan)

        self._prev_state = None
        return self.get_state(self), {}

    def get_default_reward(self):
        target = self._get_target()

        pred = self.compute_pred_from_actions(np.array(self.last_action))
        reward = 1 - np.abs(target - pred)
        return reward

    def get_exponential_reward(self):
        c = self.exp_reward
        target = self._get_target()
        pred = self.compute_pred_from_actions(np.array(self.last_action))
        return np.exp(-c * np.abs(target - pred))

    def compute_pred_from_actions(self, actions: np.ndarray, max_dim: int = None) -> np.ndarray:
        """
        Aggregates the actions up to level in the action vector(s) into a joint prediction, using a weighted sum induced by the importance of each action.

        Parameters
        ----------
        actions : np.ndarray
            Of shape (action_dim,) Action vector(s) to aggregate into the prediction.
        level : int
            Max action dimension to consider for prediction.
        """
        if max_dim is None:
            max_dim = len(actions - 1)  # start counting from dim 0
        if len(actions.shape) == 1:
            importances = self.dim_importances
        else:
            importances = self.dim_importances.reshape(-1, 1)

        if self.reverse_agents:
            actions = actions[::-1]

        pred = actions[0] / (self.action_space.nvec[0] - 1)
        # finetune +/- 0.5 around prior action dimension predictions
        unweighted_preds = actions[1:max_dim + 1] / (self.action_space.nvec[1:max_dim + 1] - 1) - 0.5
        pred += np.sum(unweighted_preds * importances[1:max_dim + 1], axis=0)

        return pred

    def _get_target(self, t: int = None):
        t = self.c_step if t is None else t
        if self.c_step <= self.inter_x:
            target = self.left_y + (self.inter_y - self.left_y) / self.inter_x * t
        else:
            target = self.inter_y + (self.right_y - self.inter_y) / (self.n_steps - 1 - self.inter_x) * (t - self.inter_x)
        return target

    def plot_predictions(self, ax: plt.Axes = None):
        """
        Plot the aggregated predictions of all actions per time step of the current episode.
        """
        if ax is None:
            fig, ax = plt.subplots()
        # Plot the current instance function
        x_left = np.arange(self.inter_x)
        x_right = np.arange(start=np.ceil(self.inter_x), stop=self.n_steps)
        x = np.concatenate([x_left, np.array([self.inter_x]), x_right])
        y = np.piecewise(x=x,
                         condlist=[x <= self.inter_x, x > self.inter_x],
                         funclist=[lambda x: self.left_y + (self.inter_y - self.left_y) / self.inter_x * x,
                                   lambda x: self.inter_y + (self.right_y - self.inter_y) / (self.n_steps - 1 - self.inter_x) * (x - self.inter_x)])
        ax.plot(x, y, zorder=0)
        episode_actions = self.episode_actions[:, :self.c_step + 1]
        predictions = np.full((self.n_actions, self.c_step), np.nan)
        for dim in range(self.n_actions):
            for t in range(self.c_step):
                predictions[dim] = self.compute_pred_from_actions(episode_actions[:, t], max_dim=dim)
        # create viridis color map, with one color per action dimension
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_actions))
        for dim, pred in enumerate(predictions):
            ax.scatter(range(self.c_step), pred, marker="o", color=colors[dim], zorder=dim + 1)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Aggregated Prediction")
        ax.set_title("Aggregated Predictions per Time Step")
        ax.set_title("Current Instance Function")

        return ax

        plt.show()
