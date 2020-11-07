import gym
from gym.envs.registration import register

from competitive_rl.car_racing.car_racing_multi_players import CarRacing


def register_competitive_envs():
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


def make_car_racing(env_id, seed, rank, resized_dim=84):
    def _thunk():
        env = gym.make(env_id)
        # env = make_atari(env_id)
        env.seed(seed + rank)
        # if log_dir is not None:
        #     env = Monitor(env, os.path.join(log_dir, str(rank)))
        env = wrap_deepmind(env, resized_dim)
        env = WrapPyTorch(env)
        return env

    return _thunk
