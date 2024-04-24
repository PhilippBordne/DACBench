from dacbench import AbstractMADACEnv
import numpy as np
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

        self._prev_state = None
        return self.get_state(self), {}

    def get_default_reward(self):
        if self.c_step <= self.inter_x:
            target = self.left_y + (self.inter_y - self.left_y) / self.inter_x * self.c_step
        else:
            target = self.inter_y + (self.right_y - self.inter_y) / (self.n_steps - 1 - self.inter_x) * (self.c_step - self.inter_x)

        pred = self.compute_pred_from_actions(np.array(self.last_action))
        reward = 1 - np.abs(target - pred)
        return reward

    def get_exponential_reward(self):
        c = self.exp_reward
        return np.exp(-c * np.abs(1 - self.get_default_reward()))

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
            max_dim = len(actions)
        importances = self.dim_importances

        if self.reverse_agents:
            actions = actions[::-1]

        pred = actions[0] / (self.action_space.nvec[0] - 1)
        # finetune +/- 0.5 around prior action dimension predictions
        unweighted_preds = actions[1:max_dim] / (self.action_space.nvec[1:max_dim] - 1) - 0.5
        pred += np.sum(unweighted_preds * importances[1:max_dim])

        return pred
