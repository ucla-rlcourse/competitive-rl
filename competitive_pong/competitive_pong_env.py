import random

import numpy as np

from competitive_pong.builtin_policies import get_compute_action_function, \
    get_builtin_agent_names


class TournamentEnvWrapper:
    def __init__(self, env, num_envs):
        self.env = env
        self.agents = {
            agent_name: get_compute_action_function(agent_name, num_envs)
            for agent_name in get_builtin_agent_names()
            if agent_name != "ALPHA_PONG"
        }
        self.agent_names = list(self.agents)
        self.current_agent = random.choice(self.agent_names)
        self.prev_opponent_obs = None

    def step(self, action):
        tuple_action = np.stack([
            np.asarray(action).reshape(-1),
            np.asarray(
                self.agents[self.current_agent](self.prev_opponent_obs)
            ).reshape(-1)
        ], axis=1)

        obs, rew, done, info = self.env.step(tuple_action)
        self.prev_opponent_obs = obs[1]
        return obs[0], rew[:, 0], done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_opponent_obs = obs[1]
        return obs[0]
