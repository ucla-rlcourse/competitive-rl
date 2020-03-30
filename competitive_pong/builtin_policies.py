"""
This file defines a set of built-in agents that can be used as opponent in
training and evaluation.

Usages:
1. Get a list of provided agent names: get_builtin_agent_names()
2. Get a policy (compute_action_function) by
    get_compute_action_function(agent_name)
3. Visualize builtin agents by calling this file:
    python load_agents.py --left STRONG --right MEDIUM

Nothing you need to implement in this file, unless you wish to implement a
custom policy or introduce a custom agent as opponent in training and
evaluation.

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import os.path as osp

import gym
import numpy as np

from competitive_pong.base_pong_env import CHEAT_CODES
from competitive_pong.utils.policy_serving import Policy

BUILTIN_AGENT_NAMES = [
    "RANDOM",
    "WEAK",
    "MEDIUM",
    "STRONG",
    "RULE_BASED",
    "ALPHA_PONG"  # Boss-level agent, not used for training
]

# Hard-coded
single_obs_space = gym.spaces.Box(0, 255, (1, 42, 42))
single_act_space = gym.spaces.Discrete(3)


def get_builtin_agent_names():
    return BUILTIN_AGENT_NAMES


def get_rule_based_policy(num_envs=1):
    if num_envs == 1:
        return lambda _: CHEAT_CODES
    else:
        return lambda _: [CHEAT_CODES] * num_envs


def get_random_policy(num_envs=1):
    if num_envs == 1:
        return lambda obs: np.random.randint(3)
    else:
        return lambda obs: [np.random.randint(3) for _ in range(num_envs)]


def get_compute_action_function(agent_name, num_envs=1):
    if agent_name == "STRONG":
        return Policy(
            single_obs_space, single_act_space, num_envs,
            osp.join(osp.dirname(__file__), "resources",
                     "checkpoint-strong.pkl"),
            use_light_model=False
        )
    if agent_name == "MEDIUM":
        return Policy(
            single_obs_space, single_act_space, num_envs,
            osp.join(osp.dirname(__file__), "resources",
                     "checkpoint-medium.pkl"),
            use_light_model=True
        )
    if agent_name == "ALPHA_PONG":
        return Policy(
            single_obs_space, single_act_space, num_envs,
            osp.join(osp.dirname(__file__), "resources",
                     "checkpoint-alphapong.pkl"),
            use_light_model=False
        )
    if agent_name == "WEAK":
        return Policy(
            single_obs_space, single_act_space, num_envs,
            osp.join(osp.dirname(__file__), "resources", "checkpoint-weak.pkl"),
            use_light_model=True
        )
    if agent_name == "RANDOM":
        return get_random_policy(num_envs)
    if agent_name == "RULE_BASED":
        return get_rule_based_policy(num_envs)
    raise ValueError("Unknown agent name: {}".format(agent_name))
