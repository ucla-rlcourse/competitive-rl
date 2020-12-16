from competitive_rl.car_racing import register_car_racing
from competitive_rl.car_racing.custom_PPO2 import PPO2

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common import make_vec_env

import gym

register_car_racing()


def run_self_play(seed=100, rank=1):
    e = gym.make('cCarRacingDouble-v0', action_repeat=1)
    # own_model = PPO2(CnnPolicy, e)
    # rival_model = PPO2(CnnPolicy, e)
    s = e.reset()
    episode_count = 0
    while True:
        print(episode_count)
        episode_count += 1
        step_count = 0
        while True:
            step_count += 1
            own_a = [0, 0.3]  # own_a = own_model.predict(s)
            rival_a = [0, 0.1]  # rival_a = rival_model.predict(s)
            obs, rewards, done, _ = e.step([own_a, rival_a])
            e.render()
            if done[0] or done[1]:
                break

        print(f"Episode {episode_count} ends with {step_count}")
        e.reset()


if __name__ == '__main__':
    run_self_play()
