from gymnasium.spaces import Box, MultiDiscrete
import numpy as np
from typing import List

from dacbench.envs import SigmoidEnv


# TODO: Consider making this a wrapper function instead, as we simply mimic the MADAC SigmoidEnv
class LeaderFollowerSigmoidEnv(SigmoidEnv):
    def __init__(self, config) -> None:
        """
        Initialize Sigmoid Env.

        Parameters
        ----------
        config : objdict
            Environment configuration

        """
        super(LeaderFollowerSigmoidEnv, self).__init__(config)
        # define observation spaces per agent (follower observations include leader observations)
        # TODO: check how observations are define in MADAC paper (do agents observe complete state or only their sigmoid?)
        self.observation_spaces = {}
        for a in self.possible_agents:
            self.observation_spaces[a] = self.get_observation_space(a)

    def get_observation_space(self, agent_id: int):
        """Get observation space for agent. As of now assuming that agent id induces the order of importance of the agents.
        NOTE: Assuming that the agent-id start counting at 0."""
        # TODO: enable processing of partial orders (e.g. for every agent specify its parents and build the action space accordingly).
        # This will possibly require adaptating the multi-agent step where that has to ensure that agents take actions in this order
        # and it needs to be more fine-grained to only adding the actions from predecessor agents to the observations.
        # IDEA: store action selection per agent_id (possibly done anyways).
        if isinstance(self.action_space, MultiDiscrete) and isinstance(self.observation_space, Box):
            return Box(low=np.array([-np.inf for _ in range(1 + len(self.action_space.nvec) * 3 + agent_id)]),
                       high=np.array([np.inf for _ in range(1 + len(self.action_space.nvec) * 3 + agent_id)]),
                       dtype=np.float32)
        elif isinstance(self.observation_space, Box):
            raise TypeError("Only MultiDiscrete action spaces are supported.")
        # actually this shouldn't be triggered as the sigmoid benchmark only has Box observation spaces
        else:
            raise TypeError("Only Box observation spaces are supported.")

    def last(self, agent_id: int = None):
        """Get last observation, reward, terminated, truncated, info for agent.
        Appends actions chosen by previous agents to the observation of the current agent if agent_id provided."""
        observation, reward, termination, truncation, info = super().last()
        if agent_id is not None:
            # modified observation will look like:
            # [remaining_budget, shift_0, slope_0, ..., shift_n, slope_n, last_action_0, ..., last_action_n,
            # next_action_0, ..., next_action_agent_id-1]
            actions_to_append = np.array([self.action[a] for a in range(agent_id)])
            observation = np.append(observation, np.full(agent_id, actions_to_append))
        return observation, reward, termination, truncation, info

    def multi_agent_step(self, action):
        return super().multi_agent_step(action)


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

        self.get_reward = self.get_default_reward

    def reset(self, seed=None, options={}) -> List[int]:
        super().reset(seed, options)
        self.shifts = [self.instance[0]]
        self.slopes = [self.instance[self.n_actions]]
        self._prev_state = None
        return self.get_state(self), {}

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
    def __init__(self, config) -> None:
        super().__init__(config)
        # see whether dimension importances are specified in config else fall back to default
        if "dim_importances" in config.keys():
            self.dim_importances = config["dim_importances"]
        else:
            self.dim_importances = [0.3**i for i in range(self.n_actions)]

    def get_default_reward(self, _):
        """
        Reward that computes how close the current weighted sum of actions is to the target sigmoid.
        """
        # 1st action defines the step in the sigmoid. All other actions fine tune around it with different precisions
        pred = self.last_action[0] / self.action_space.nvec[0]

        # the other actions fine tune around first action according to their importance
        # offset to -0.5 to have a range of [-0.5, 0.5] for all actions
        pred += np.sum((self.last_action[1:] / self.action_space.nvec[1:] - 0.5) * np.array(self.dim_importances[1:]))

        reward = 1 - np.abs(self._sig(self.c_step, self.slopes[0], self.shifts[0]) - pred)
        # clip into reward range
        reward = max(self.reward_range[0], min(self.reward_range[1], reward))
        return reward


class DiffImportanceLeaderFollowerSigmoidEnv(LeaderFollowerSigmoidEnv):
    def __init__(self, config) -> None:
        """
        Initialize Sigmoid Env.

        Parameters
        ----------
        config : objdict
            Environment configuration

        """
        super(DiffImportanceLeaderFollowerSigmoidEnv, self).__init__(config)
        self.get_reward = self.get_default_reward

    def get_default_reward(self, _):
        """
        Adapted reward function where the prediction is on a single sigmoid and the action is a composure 
        """
        pass
