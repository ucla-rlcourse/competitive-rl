import gym

from competitive_rl.car_racing.register import register_competitive_envs
from competitive_rl.utils.atari_wrappers import WrapPyTorch


def make_car_racing(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = WrapPyTorch(env)
        return env

    return _thunk
