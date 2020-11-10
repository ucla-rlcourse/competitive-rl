import gym
from gym.envs.registration import register

from competitive_rl.car_racing.car_racing_multi_players import CarRacing
from competitive_rl.utils.atari_wrappers import WrapPyTorch, FrameStack


def register_car_racing():
    try:
        register(
            id="cCarRacing-v0",
            entry_point=CarRacing,
            kwargs=dict(verbose=0),
            max_episode_steps=1000,
            reward_threshold=900
        )
        register(
            id="cCarRacingDouble-v0",
            entry_point=CarRacing,
            kwargs=dict(verbose=0, num_player=2),
            max_episode_steps=1000,
            reward_threshold=900
        )
        print("Register cCarRacing-v0, cCarRacingDouble-v0 environments.")
    except gym.error.Error:
        pass


def make_car_racing(env_id, seed, rank, frame_stack=None, action_repeat=None):
    assert "CarRacing" in env_id

    def _thunk():
        env = gym.make(env_id, action_repeat=action_repeat)
        env.seed(seed + rank)
        if frame_stack is not None:
            env = FrameStack(env, frame_stack)
        env = WrapPyTorch(env)
        return env

    return _thunk
