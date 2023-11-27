from dacbench.abstract_env import AbstractMADACEnv

from gymnasium.spaces import Box
import numpy as np


class LeaderFollowerWrapper:
    def __init__(self, env: AbstractMADACEnv) -> None:
        """
        Class that emulates a leader follower observations for a MADAC environment. E.g. it can be used to observe and act in a MADAC
        environment in a leader follower manner. However it is not intended to replace the full API of a MADAC environment but only
        those parts that are relevant for leader follower observations and actions.
        NOTE: Implemented under assumption that each agent takes over one action and agent_ids induce a total order of importance.
        """
        # TODO: generalize for partial importance orders and for multiple actions per agent
        self.env = env
        self.action_vector = [None for _ in range(self.env.n_agents)]

    def get_observation_space(self, agent_id: int):
        """Get observation space for agent. As of now assuming that agent id induces the order of importance of the agents.
        NOTE: Assuming that the agent-id start counting at 0."""
        # we only need to add a dimension to the default observation space for every predecessor agent, which corresponds to agent_id
        # assuming total order of agents
        if isinstance(self.env.observation_space, Box):
            low = self.env.observation_space.low
            high = self.env.observation_space.high
            # assuming no bound on actions
            low = np.append(low, np.full(agent_id, -np.inf))
            high = np.append(high, np.full(agent_id, np.inf))
            return Box(low=low, high=high, dtype=np.float32)
        else:
            raise TypeError("Can only wrap Box observation spaces.")

    def get_wrapped_observation(self, agent_id: int):
        """Wrap the observation of the agent with the given agent_id to include the actions of the predecessor agents.
        NOTE: Assure that the agents take their in order of their importance to ensure that the environment holds all actions 
        of the predecessor agents for the current state."""
        obs = self.env.last()
        obs = np.append(obs, self.env.action[:agent_id])
        return obs
