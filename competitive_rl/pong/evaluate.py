import time

import numpy as np


def evaluate_two_policies(compute_action0, compute_action1, env, num_episode,
                          render=False, print_console=None, env_name="",
                          render_interval=0.05):
    gameResult0 = [0] * 4  # [0] Win [1] Draw [2] Lose [3] Cumulative Reward
    gameResult1 = [0] * 4  # [0] Win [1] Draw [2] Lose [3] Cumulative Reward
    reward_list = []

    for episode in range(num_episode):
        matchTotalReward = [0.0, 0.0]
        obs = env.reset()
        done = False
        if hasattr(compute_action0, "reset"):
            compute_action0.reset()
        elif hasattr(compute_action1, "reset"):
            compute_action1.reset()
        while not done:
            action = [compute_action0(obs[0]), compute_action1(obs[1])]
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            matchTotalReward[0] += reward[0]
            matchTotalReward[1] += reward[1]
            if render:
                time.sleep(render_interval)
                env.render(mode="human")
        if matchTotalReward[0] > 0.0:
            gameResult0[0] += 1
            gameResult1[2] += 1
        elif matchTotalReward[0] == 0.0:
            gameResult0[1] += 1
            gameResult1[1] += 1
        else:
            gameResult0[2] += 1
            gameResult1[0] += 1
        gameResult0[3] += matchTotalReward[0]
        gameResult1[3] += matchTotalReward[1]
        reward_list.append(matchTotalReward[0])

        if print_console is None:
            continue

        # Print match result
        print_console.printMatchInfo(
            env_name, episode, matchTotalReward[0]
        )
    return gameResult0, gameResult1


def evaluate_two_policies_in_batch(compute_action0, compute_action1, envs, num_episodes):
    gameResult0 = [0] * 4  # [0] Win [1] Draw [2] Lose [3] Cumulative Reward
    gameResult1 = [0] * 4  # [0] Win [1] Draw [2] Lose [3] Cumulative Reward
    episode_rewards = np.zeros([envs.num_envs, 2], dtype=np.float)
    total_episodes = 0
    obs = envs.reset()
    while True:
        actions = np.stack(
            [np.asarray(compute_action0(obs[0])).reshape(-1),
             np.asarray(compute_action1(obs[1])).reshape(-1)],
            axis=1
        )  # [num_envs, 2]

        obs, reward, done, info = envs.step(actions)
        if not np.isscalar(done[0]):
            done = np.all(done, axis=1)
        episode_rewards += reward
        for idx, d in enumerate(done):
            if d:  # the episode is done
                if episode_rewards[idx, 0] > 0.0:
                    gameResult0[0] += 1
                    gameResult1[2] += 1
                elif episode_rewards[idx, 0] == 0.0:
                    gameResult0[1] += 1
                    gameResult1[1] += 1
                else:
                    gameResult0[2] += 1
                    gameResult1[0] += 1
                gameResult0[3] += episode_rewards[idx, 0]
                gameResult1[3] += episode_rewards[idx, 1]
                total_episodes += 1
        masks = 1. - done.astype(np.float32)
        episode_rewards *= masks.reshape(-1, 1)
        if total_episodes >= num_episodes:
            break
    return gameResult0, gameResult1
