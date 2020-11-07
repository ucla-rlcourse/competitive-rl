import gym
from gym.envs.registration import register

from competitive_rl.car_racing.car_racing_multi_players import CarRacing
from competitive_rl.utils.atari_wrappers import WrapPyTorch


def register_car_racing():
    try:
        register(
            id="cCarRacing-v0",
            entry_point=CarRacing,
            kwargs=dict(verbose=0),
            max_episode_steps=1000,
            reward_threshold=900
        )
        print("Register car_racing_multiple_players environments.")
    except gym.error.Error:
        pass


def make_car_racing(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = WrapPyTorch(env)
        return env

    return _thunk
