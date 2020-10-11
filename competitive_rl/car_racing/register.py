import gym
from gym.envs.registration import register

from competitive_rl.car_racing.car_racing_multi_players import CarRacing


def register_competitive_envs():
    register(
        id="cCarRacing-v0",
        entry_point=CarRacing,
        max_episode_steps=1000,
        reward_threshold=900
    )
    print(
        "Register car_racing_multiple_players environments.")