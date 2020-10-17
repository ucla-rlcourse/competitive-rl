import argparse

import numpy as np

import gym
import torch
import torch.nn as nn

from competitive_rl.car_racing.car_racing_multi_players import CarRacing
from competitive_rl.car_racing.controller import key_phrase

parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

from competitive_rl.car_racing.register import register_competitive_envs

register_competitive_envs()

class Env():
    """
    Test environment wrapper for CarRacing
    """

    def __init__(self,num_player=1):
        #self.env = gym.make('cCarRacing-v0')
        self.num_palyer = num_player
        self.env = CarRacing(num_player=num_player)
        self.env.seed(args.seed)
        #self.reward_threshold = self.env.spec.reward_threshold

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.dies = [False] * self.num_palyer
        img_rgbs = self.env.reset()
        img_grays = [self.rgb2gray(img_rgbs[i]) for i in range(self.num_palyer)]
        self.stacks = [[img_grays[i]] * args.img_stack for i in range(self.num_palyer)]
        return [np.array([img_grays[i]] * args.img_stack) for i in range(self.num_palyer)]

    def step(self, actions):
        total_rewards = [0] * self.num_palyer
        img_rgb = []
        for i in range(args.action_repeat):
            img_rgbs, rewards, dies, _ = self.env.step(actions)
            img_rgb = img_rgbs
            for i in range(self.num_palyer):
                # don't penalize "die state"
                if dies[i]:
                    rewards[i] += 100
                # green penalty
                if np.mean(img_rgbs[i][:, :, 1]) > 185.0:
                    rewards[i] -= 0.05
                total_rewards[i] += rewards[i]
                # if no reward recently, end the episode
                #dones[i] = True if self.av_r(reward) <= -0.1 else False
                #if done or dies[i]:
                if dies[i]:
                    break
        img_grays = [self.rgb2gray(img_rgb[i]) for i in range(self.num_palyer)]
        for i in range(self.num_palyer):
            self.stacks[i].pop(0)
            self.stacks[i].append(img_grays[i])
            assert len(self.stacks[i]) == args.img_stack
        #return np.array(self.stack), total_reward, done, die
        return [np.array(self.stacks[i]) for i in range(self.num_palyer)], total_rewards, False, dies

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(args.img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v


class Agent():
    """
    Agent for testing
    """

    def __init__(self):
        self.net = Net().float().to(device)

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)

        action = action.squeeze().cpu().numpy()
        return action

    def load_param(self):
        self.net.load_state_dict(torch.load('param/car0.0.pkl'))


if __name__ == "__main__":
    num_player = 4
    agents = []
    for _ in range(num_player):
        agent = Agent()
        agent.load_param()
        agents.append(agent)
    #agent = Agent()
    #agent.load_param()
    env = Env(num_player=num_player)
    training_records = []
    running_score = 0
    states = env.reset()
    a = [[0, 0, 0] * num_player]
    for i_ep in range(1):
        score = [0] * num_player
        states = env.reset()

        for t in range(1000):
            actions = []
            for i in range(num_player):
                actions.append(agents[i].select_action(states[i])* np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            #print(states[0].shape, states[1].shape)
            env.env.manage_input(key_phrase(a))
            if env.env.show_all_car_obs:
                env.env.show_all_obs(states, grayscale=False)
            state_, reward, done, die = env.step(actions)
            print(reward)
            env.render()
            for i in range(num_player):
                score[i] += reward[i]
            states = state_
            if any(die):
                break

        print(f'Ep {i_ep}\tScore: {score}\t')