from dacbench import AbstractMADACEnv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
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

        pred = self.compute_pred_from_actions()
        reward = 1 - np.abs(target - pred)
        return reward

    def get_exponential_reward(self):
        c = self.exp_reward
        target = self._get_target()
        pred = self._compute_pred_from_actions()
        return np.exp(-c * np.abs(target - pred))

    def _get_target(self, t: int = None):
        t = self.c_step if t is None else t
        if t <= self.inter_x:
            target = self.left_y + (self.inter_y - self.left_y) / self.inter_x * t
        else:
            target = self.inter_y + (self.right_y - self.inter_y) / (self.n_steps - 1 - self.inter_x) * (t - self.inter_x)
        return target

    def _compute_pred_from_actions(self) -> np.ndarray:
        actions = self.last_action if not self.reverse_agents else self.last_action[::-1]
        pred = actions[0] / (self.action_space.nvec[0] - 1)
        unweighted_preds = actions[1:] / (self.action_space.nvec[1:] - 1) - 0.5
        pred += np.sum(unweighted_preds * self.dim_importances[1:])
        return pred

    """
    These functions can be used for visualization and analysis. They isolate from functionality of the environment itself
    at the cost of some boilerplate to make the env mechanics better understandable and in the first place to avoid stupid
    bugs when running experiments :-)
    """

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
        importances = self.dim_importances[1:max_dim + 1]
        n_act = self.action_space.nvec[1:max_dim + 1]

        if len(actions.shape) > 1:
            importances = importances.reshape(-1, 1)
            n_act = n_act.reshape(-1, 1)

        if self.reverse_agents:
            actions = actions[::-1]

        pred = actions[0] / (self.action_space.nvec[0] - 1)

        # finetune +/- 0.5 around prior action dimension predictions
        unweighted_preds = actions[1:max_dim + 1] / (n_act - 1) - 0.5
        pred += np.sum(unweighted_preds * importances, axis=0)

        return pred

    def plot_predictions(self, ax: plt.Axes = None, add_legend: bool = False):
        """
        Plot the aggregated predictions of all actions per time step of the current episode.
        """
        fill_external_ax = ax is not None
        if ax is None:
            fig, ax = plt.subplots()
        # Plot the current instance function
        ax.set_xticks(range(0, self.n_steps, 1))
        x_left = np.arange(self.inter_x)
        x_right = np.arange(start=np.ceil(self.inter_x), stop=self.n_steps)
        x = np.concatenate([x_left, np.array([self.inter_x]), x_right])
        y = np.piecewise(x=x,
                         condlist=[x <= self.inter_x, x > self.inter_x],
                         funclist=[lambda x: self.left_y + (self.inter_y - self.left_y) / self.inter_x * x,
                                   lambda x: self.inter_y + (self.right_y - self.inter_y) / (self.n_steps - 1 - self.inter_x) * (x - self.inter_x)])
        ax.plot(x, y, zorder=0, color="tab:orange", label="target function")
        episode_actions = self.episode_actions[:, :self.c_step + 1]
        predictions = np.full((self.n_actions, self.c_step), np.nan)
        for i in range(self.n_actions):
            predictions[i] = self.compute_pred_from_actions(episode_actions, max_dim=i)

        # create viridis color map, with one color per action dimension
        colors = plt.cm.viridis_r(np.linspace(0, 1, self.n_actions))
        for i, pred in enumerate(predictions):
            ax.scatter(range(self.c_step), pred, marker="o", color=colors[i], zorder=i + 1, label=f"actions aggregated up to dim {i}",
                       alpha=0.3 + 0.7 * (i + 1) / self.n_actions, edgecolors="none")
        # plot a line that connects the highest level of aggregation
        ax.plot(range(self.c_step), predictions[-1], color=colors[-1], zorder=self.n_actions + 1, label="aggregated prediction",
                linestyle=":")

        if fill_external_ax and add_legend:
            custom_lines = [Line2D([0], [0], color="tab:orange", linestyle="-", label="target"),
                            Line2D([0], [0], linestyle="None", marker="o", color="gray", label="aggregated predictions")]
            ax.legend(loc='center left', bbox_to_anchor=(1, -0.2), handles=custom_lines)

        if not fill_external_ax:
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Aggregated Prediction")
            ax.set_title("Aggregated Predictions per Time Step")
            ax.set_title("Current Instance Function")
            return

        return ax

    def render_action_selection(self, t: int = None, ax: plt.Axes = None):
        """
        Renders how the several action dimensions are aggregated into the final prediction.
        If t is None will render for the last time step taken. (e.g. the last action submitted to the environment)
        """
        fill_ax = ax is not None
        if ax is None:
            fig, ax = plt.subplots()

        t = self.c_step - 1 if t is None else t
        action_to_plot = self.episode_actions[:, t]

        predictions = np.full(self.n_actions, np.nan)
        for dim in range(self.n_actions):
            predictions[dim] = self.compute_pred_from_actions(action_to_plot, max_dim=dim)

        possible_predictions = np.full((self.n_actions, self.action_space.nvec[0]), np.nan)

        for dim in range(self.n_actions):
            if dim == 0:
                possible_predictions[dim] = np.linspace(0, 1, self.action_space.nvec[dim])
            else:
                possible_predictions[dim] = predictions[dim - 1] + np.linspace(-0.5, 0.5, self.action_space.nvec[dim]) * self.dim_importances[dim]

        # plot the prediction target
        target = self._get_target(t)
        ax.axhline(y=target, color="tab:orange", label="target", zorder=0)

        # create viridis color map, with one color per action dimension
        colors = plt.cm.viridis_r(np.linspace(0, 1, self.n_actions))

        for dim, preds in enumerate(possible_predictions):
            ax.vlines(dim, ymin=preds[0], ymax=preds[-1], color=colors[dim], alpha=0.5)
            ax.scatter([dim] * len(preds), preds, color=colors[dim], zorder=dim + 1, marker="_", label=f"dim {dim} possible predictions")
            ax.scatter(dim, predictions[dim], color=colors[dim], zorder=10, marker="o", label=f"dim {dim} predictions")

        custom_lines = [Line2D([0], [0], color="tab:orange", linestyle="-", label="target"),
                        Line2D([0], [0], linestyle="None", marker="_", color="gray", label="possible predictions"),
                        Line2D([0], [0], linestyle="None", marker="o", color="gray", label="predictions")]

        # set up a legend for the symbols
        ax.legend(handles=custom_lines, labels=['target function value',
                                                'possible prediction',
                                                'actual prediction'], loc="lower right")

        x_min, x_max = ax.get_xlim()
        shrink = (self.n_actions - 1) / (x_max - x_min)
        cax, _ = matplotlib.colorbar.make_axes(ax, location="bottom", pad=0.15, shrink=shrink)
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=plt.cm.viridis_r, norm=matplotlib.colors.Normalize(vmin=0, vmax=dim),
                                                ticks=[], orientation="horizontal")
        # cbar.set_label('aggregated action dimensions', pad=0.1)
        ax.set_xlabel('aggregated action dimensions')
        # remove the numbers on the x-axis but keep the vertical lines
        ax.set_xticks(np.arange(self.n_actions))
        ax.set_xticklabels(np.arange(self.n_actions))

        if fill_ax:
            return ax

        else:
            return
