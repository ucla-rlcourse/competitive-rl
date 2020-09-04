import gym
from gym.envs.registration import register

from competitive_rl.pong.base_pong_env import PongSinglePlayerEnv, \
    PongDoublePlayerEnv


def register_competitive_envs():
    try:
        register(
            id="cPong-v0",
            entry_point=PongSinglePlayerEnv,
            kwargs=dict(
                max_num_rounds=21
            )
        )
        register(
            id="cPongDouble-v0",
            entry_point=PongDoublePlayerEnv,
            kwargs=dict(
                max_num_rounds=21
            )
        )
        print(
            "Register cPong-v0 and cPongDouble-v0 environments.")
    except gym.error.Error:
        pass
