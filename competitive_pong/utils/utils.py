"""
This file implements the some helper functions.

You need to finish `step_envs` function.

Note that many codes in this file is not written by us. The credits go to the
original writers.

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import copy
import os
import time

import numpy as np
import pandas as pd
import torch


def step_envs(cpu_actions, envs, episode_rewards, frame_stack_tensor,
              reward_recorder, length_recorder, total_steps, total_episodes,
              device, test):
    """Step the vectorized environments for one step. Process the reward
    recording and terminal states."""
    obs, reward, done, info = envs.step(cpu_actions)
    episode_rewards += reward.reshape(episode_rewards.shape)
    episode_rewards_old_shape = episode_rewards.shape
    if not np.isscalar(done[0]):
        done = np.all(done, axis=1)
    for idx, d in enumerate(done):
        if d:  # the episode is done
            # Record the reward of the terminated episode to
            reward_recorder.append(episode_rewards[idx].copy())

            # For CartPole-v0 environment, the length of episodes is not
            # recorded.
            if "num_steps" in info[idx]:
                length_recorder.append(info[idx]["num_steps"])
            total_episodes += 1
    masks = 1. - done.astype(np.float32)
    episode_rewards *= masks.reshape(-1, 1)

    assert episode_rewards.shape == episode_rewards_old_shape

    total_steps += obs[0].shape[0] if isinstance(obs, tuple) else obs.shape[0]
    masks = torch.from_numpy(masks).to(device).view(-1, 1)
    # frame_stack_tensor is refreshed in-place if done happen.
    if test:
        frame_stack_masks = masks.view(-1, 1)
    else:
        frame_stack_masks = masks.view(-1, 1, 1, 1)
    # If in multiple pong mode, we suppose only the first observation is used to
    # train agent.
    frame_stack_tensor.update(obs[0] if isinstance(obs, tuple) else obs,
                              frame_stack_masks)
    return obs, reward, done, info, masks, total_episodes, total_steps, \
           episode_rewards


def save_progress(log_dir, progress):
    path = os.path.join(log_dir, "progress.pkl")
    torch.save(progress, path)
    return path


def load_progress(log_dir):
    progress = torch.load(os.path.join(log_dir, "progress.pkl"))
    progress = [flatten_dict(d) for d in progress]
    return pd.DataFrame(progress)


def flatten_dict(dt, delimiter="/"):
    dt = copy.deepcopy(dt)
    while any(isinstance(v, dict) for v in dt.values()):
        remove = []
        add = {}
        for key, value in dt.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    add[delimiter.join([key, subkey])] = v
                remove.append(key)
        dt.update(add)
        for k in remove:
            del dt[k]
    return dt


def summary(array, name, extra_dict=None):
    ret = {
        "{}_mean".format(name): float(np.mean(array)) if len(array) else np.nan,
        "{}_min".format(name): float(np.min(array)) if len(array) else np.nan,
        "{}_max".format(name): float(np.max(array)) if len(array) else np.nan,
    }
    if extra_dict:
        ret.update(extra_dict)
    return ret


def evaluate(trainer, eval_envs, frame_stack, num_episodes=10, seed=0):
    """This function evaluate the given policy and return the mean episode
    reward.
    :param policy: a function whose input is the observation
    :param env: an environment instance
    :param num_episodes: number of episodes you wish to run
    :param seed: the random seed
    :return: the averaged episode reward of the given policy.
    """

    frame_stack_tensor = FrameStackTensor(
        eval_envs.num_envs, eval_envs.observation_space.shape, frame_stack,
        trainer.device
    )

    def get_action(frame_stack_tensor):
        obs = frame_stack_tensor.get()
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(trainer.device)
        with torch.no_grad():
            act = trainer.compute_action(obs, deterministic=True)[1]
        act = act.view(-1).cpu().numpy()
        return act

    reward_recorder = []
    episode_length_recorder = []
    episode_rewards = np.zeros([eval_envs.num_envs, 1], dtype=np.float)
    total_steps = 0
    total_episodes = 0
    eval_envs.seed(seed)
    obs = eval_envs.reset()
    frame_stack_tensor.update(obs)
    while True:
        obs, reward, done, info, masks, total_episodes, total_steps, \
        episode_rewards = step_envs(
            get_action(frame_stack_tensor), eval_envs, episode_rewards,
            frame_stack_tensor, reward_recorder, episode_length_recorder,
            total_steps, total_episodes, trainer.device, frame_stack == 1)
        if total_episodes >= num_episodes:
            break
    return reward_recorder, episode_length_recorder


class FrameStackTensor:
    def __init__(self, num_envs, obs_shape, frame_stack, device):
        self.num_channels = obs_shape[0]
        self.obs_shape = (obs_shape[0] * frame_stack, *obs_shape[1:])
        self.current_obs = torch.zeros(num_envs, *self.obs_shape, device=device,
                                       dtype=torch.float)
        self.mask_shape = [1] * self.current_obs.dim()
        self.mask_shape[0] = -1
        self.device = device

    def reset(self):
        self.current_obs.fill_(0)

    def update(self, obs, mask=None):
        """current_obs is a tensor with shape [num_envs, num_stacks, 84, 84].
        It keeps rolling at second dimension in order to stack the latest
        num_stacks frames.
        """
        if mask is not None:
            mask = mask.reshape(self.mask_shape)
            self.current_obs *= mask
        self.current_obs = self.current_obs.roll(
            shifts=-self.num_channels, dims=1)
        obs = torch.from_numpy(obs.astype(np.float32)).to(self.device)
        self.current_obs[:, -self.num_channels:] = obs
        return self.current_obs

    def get(self):
        return self.current_obs


class PrintConsole:
    def __init__(self, number_of_episode):
        self._number_of_episode = number_of_episode
        self.start_time = time.time()

    def start(self):
        self.start_time = time.time()

    # Print the info when starting an env
    def printStartingInfo(self, envName, action_space, obs_space):
        print("-------------------Env Info----------------------")
        print("[%s] Environment is successfully Made" % (envName))
        print("[%s] action_space=%s" % (envName, action_space))
        print("[%s] obs_space=%s" % (envName, obs_space))
        print("-------------------Env Info----------------------")

    # Print each Match information
    def printMatchInfo(self, envName, episode, matchTotalReward):
        if isinstance(matchTotalReward, list):
            print(
                "[%s] Episode %d/%d Reward: (%.1f, %.1f)."
                % (
                    envName,
                    episode + 1,
                    self._number_of_episode,
                    matchTotalReward[0],
                    matchTotalReward[1],
                )
            )
        else:
            print(
                "[%s] Episode %d/%d Reward: (%.1f)."
                % (
                    envName, episode + 1, self._number_of_episode,
                    matchTotalReward)
            )

    # Print Check Point information
    def printCheckPoint(self, envName, episode, gameResult):
        print("[%s] Check Point: Result saved at episode %d" % (
            envName, episode + 1))
        if isinstance(gameResult[0], list):
            print(
                "[%s] Player 1 - Win: %d, Draw: %d , Lose: %d"
                % (
                    envName, gameResult[0][0], gameResult[0][1],
                    gameResult[0][2])
            )
            print(
                "[%s] Player 2 - Win: %d, Draw: %d , Lose: %d"
                % (
                    envName, gameResult[1][0], gameResult[1][1],
                    gameResult[1][2])
            )
        else:
            print(
                "[%s] Win: %d, Draw: %d , Lose: %d"
                % (envName, gameResult[0], gameResult[1], gameResult[2])
            )

    # Print the info when ending the game
    def printResultInfo(self, envName, gameResult, print_time=False):
        print("----- {} -----".format(envName))
        # print("[%s] The game is ended. The result are as follow: " % (
        # envName))
        if isinstance(gameResult[0], list):
            print("-------------------Player 1----------------------")
            print("[%s] Win:\t\t%d" % (envName, gameResult[0][0]))
            print("[%s] Draw:\t\t%d" % (envName, gameResult[0][1]))
            print("[%s] Lose:\t\t%d" % (envName, gameResult[0][2]))
            print(
                "[%s] Win Rate:\t\t%.2f"
                % (envName, gameResult[0][0] / self._number_of_episode)
            )
            print("[%s] Cumulative Reward: %.1f" % (envName, gameResult[0][3]))
            print(
                "[%s] Average Reward: %.2f"
                % (envName, gameResult[0][3] / self._number_of_episode)
            )

            print("-------------------Player 2----------------------")
            print("[%s] Win: %d" % (envName, gameResult[1][0]))
            print("[%s] Draw: %d" % (envName, gameResult[1][1]))
            print("[%s] Lose: %d" % (envName, gameResult[1][2]))
            print(
                "[%s] Win Rate: %.2f"
                % (envName, gameResult[1][0] / self._number_of_episode)
            )
            print("[%s] Cumulative Reward: %.1f" % (envName, gameResult[1][3]))
            print(
                "[%s] Average Reward: %.2f"
                % (envName, gameResult[1][3] / self._number_of_episode)
            )
        else:
            print("[%s] Win:\t%d" % (envName, gameResult[0]))
            print("[%s] Draw:\t%d" % (envName, gameResult[1]))
            print("[%s] Lose:\t%d" % (envName, gameResult[2]))
            print(
                "[%s] Win Rate:\t%.2f"
                % (envName, gameResult[0] / self._number_of_episode)
            )
            print("[%s] Cumulative Reward:\t%.3f" % (envName, gameResult[3]))
            print(
                "[%s] Episode Reward:\t%.3f"
                % (envName, gameResult[3] / self._number_of_episode)
            )
            print(
                "[{}] Total Matches:\t{}".format(envName,
                                                 sum(gameResult[0: 3])))
        if print_time:
            print("[{}] Test time:\t{:.2f}".format(envName,
                                                   time.time() -
                                                   self.start_time))
