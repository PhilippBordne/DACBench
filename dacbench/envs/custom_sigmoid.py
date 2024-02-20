import numpy as np
from typing import List

from dacbench.envs import SigmoidEnv
from matplotlib import pyplot as plt


# this is partly a hack to predict on a single sigmoid whithout touching mapping from dim to instance set
# TODO: map to the 1D instance set for this class of environments
# TODO: making this an own benchmark should be more appropriate
# TODO: for this benchmark class the action values must be positive and negative. actions should be uneven to allow for 0
class DiffImportanceSigmoidEnv(SigmoidEnv):
    def __init__(self, config) -> None:
        """
        Initialize Sigmoid Env.

        Parameters
        ----------
        config : objdict
            Environment configuration
        """
        super(DiffImportanceSigmoidEnv, self).__init__(config)

        # only predicting on a single sigmoid
        self.slopes = [-1]
        self.shifts = [self.n_steps / 2]

        # for plotting we keep track of the actions seen throughout the episode
        self.episode_actions = np.full((self.n_actions, self.n_steps), np.nan)
        self.episode_rewards = np.full((self.n_steps), np.nan)

        self.get_reward = self.get_default_reward

    def reset(self, seed=None, options={}) -> List[int]:
        # only prereset of parents to update the instance
        super().reset_(seed)
        self.episode_actions = np.full((self.n_actions, self.n_steps), np.nan)
        self.episode_rewards = np.full((self.n_steps), np.nan)
        self.shifts = [self.instance[0]]
        self.slopes = [self.instance[1]]
        self._prev_state = None
        return self.get_state(self), {}

    def step(self, action: int):
        # # update action history
        # if self.multi_agent:
        #     self.action_history[self.current_agent, self.c_step] = action
        # else:
        self.episode_actions[:, self.c_step] = action
        # update reward history
        res = super().step(action)
        self.episode_rewards[self.c_step - 1] = res[1]
        return res

    def get_default_reward(self, _):
        """
        Adapted reward function where the prediction is on a single sigmoid and the action is a composure 
        """
        raise NotImplementedError("Has to be implemented by a subclass: Importance via arities or via prediction fine-tuning.")

    def render(self, mode: str) -> None:
        raise NotImplementedError("Has to be implemented by a subclass: Importance via arities or via prediction fine-tuning.")

    def get_default_state(self, _):
        # TODO: remove method as soon as dim is mapped to 1D sigmoid instance set
        remaining_budget = self.n_steps - self.c_step
        next_state = [remaining_budget]
        next_state.append(self.shifts[0])
        next_state.append(self.slopes[0])
        if self.c_step == 0:
            next_state += [-1 for _ in range(self.n_actions)]
        else:
            next_state = np.array(list(next_state) + list(self.last_action))
        return np.array(next_state)


class DiffImportanceFineTuneSigmoidEnv(DiffImportanceSigmoidEnv):
    # TODO: implement me
    def __init__(self, config, reward_shape: str = 'linear', exp_reward: float = 4.6) -> None:
        """
        Parameters
        ----------
        reward_shape : str
            Shape of the reward function. if 'linear' the reward is 1 minus the absolute difference between the target
            sigmoid and the prediction. If 'exponential' the reward decays exponentially with the difference between
            the target sigmoid and the prediction.
        """
        super().__init__(config)
        # see whether dimension importances are specified in config else fall back to default
        if "dim_importances" in config.keys():
            self.dim_importances = config["dim_importances"]
        else:
            self.dim_importances = np.array([0.3**i for i in range(self.n_actions)])

        if reward_shape == 'linear':
            self.get_reward = self.get_default_reward
        elif reward_shape == 'exponential':
            self.exp_reward = exp_reward
            self.get_reward = self.get_exponential_reward

    def get_default_reward(self, _):
        """
        Reward that computes how close the current weighted sum of actions is to the target sigmoid.
        """
        # aggregate actions into prediction on single sigmoid and compute reward by comparing to target sigmoid
        pred = self.compute_pred_from_actions(np.array(self.last_action).reshape(-1, 1))
        reward = 1 - np.abs(self._sig(self.c_step, self.slopes[0], self.shifts[0]) - pred)

        # clip into reward range
        reward = max(self.reward_range[0], min(self.reward_range[1], reward))
        return reward

    def get_exponential_reward(self, _):
        """
        Reward that computes how close the current weighted sum of actions is to the target sigmoid.
        """
        c = self.exp_reward

        # aggregate actions into prediction on single sigmoid and compute reward by comparing to target sigmoid
        pred = self.compute_pred_from_actions(np.array(self.last_action).reshape(-1, 1))
        reward = np.exp(-c * np.abs(self._sig(self.c_step, self.slopes[0], self.shifts[0]) - pred))

        return reward

    def compute_pred_from_actions(self, actions: np.ndarray, level: int = None) -> np.ndarray:
        """
        Aggregates the action vector(s) into the prediction on a single sigmoid.
        First action determines central value of prediction, other actions fine tune around it according to their importance.

        Parameters
        ----------
        actions : np.ndarray
            Of shape (action_dim, 1) Action vector(s) to aggregate into the prediction.

        Returns
        -------
        np.ndarray
            Of shape (num_actions) Predictions on the single sigmoid.
        """
        # if importances is None:
        if level is None:
            level = len(actions)
        importances = self.dim_importances
        pred = actions[0] / (self.action_space.nvec[0] - 1)
        # the other actions fine tune around first action according to their importance
        # pred += np.sum(((2 * actions[1:level] / (self.action_space.nvec[1:level].reshape(-1, 1) - 1) - 1)
        #                 * np.array(importances[1:level].reshape(-1, 1))), axis=0)
        pred += np.sum(((actions[1:level] / (self.action_space.nvec[1:level].reshape(-1, 1) - 1) - 0.5)
                        * np.array(importances[1:level]).reshape(-1, 1)), axis=0)
        return pred

    def render(self, mode: str) -> None:
        """
        Render Env.
        """
        if mode == "human":
            plt.ion()
            plt.show()
            plt.cla()
            steps = np.arange(1, self.n_steps + 1)  # +1 because 
            sigmoid = self._sig(steps, self.slopes[0], self.shifts[0])

            reward_should = 1 - np.abs(sigmoid - self.compute_pred_from_actions(self.episode_actions))

            plt.plot(steps, sigmoid, label="target", color="red")
            plt.plot(steps, reward_should, label="reward should", color="orange")
            plt.plot(steps, self.episode_rewards, label="reward", color="green")
            for i in range(self.n_actions):
                preds = self.compute_pred_from_actions(self.episode_actions[:i + 1], level=i + 1)
                plt.plot(steps, preds,
                         label=f"pred {i}", color="blue", alpha=self.dim_importances[i])
            plt.legend()
            plt.title(f"Running on instance {self.instance_index}")
            plt.pause(2)
