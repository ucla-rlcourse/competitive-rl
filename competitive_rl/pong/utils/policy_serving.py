import logging
import os

import numpy as np
import torch

from competitive_rl.pong.utils import LightActorCritic, ActorCritic
from competitive_rl.pong.utils import FrameStackTensor


class Policy:
    def __init__(self, single_obs_space, single_action_space, num_envs,
                 checkpoint_path="", frame_stack=4,
                 use_light_model=False):
        self.num_envs = num_envs
        self.obs_shape = single_obs_space.shape

        # TODO Not sure whether should we use gpu if applicable.
        self.device = torch.device("cpu")

        num_feats = single_obs_space.shape
        self.num_feats = (num_feats[0] * frame_stack, *num_feats[1:])
        self.num_actions = single_action_space.n

        if use_light_model:
            self.model = LightActorCritic(self.num_feats, self.num_actions)
        else:
            self.model = ActorCritic(self.num_feats, self.num_actions)
        self.model = self.model.to(self.device)
        self.model.train()

        if checkpoint_path:
            assert os.path.isfile(checkpoint_path), checkpoint_path
            state_dict = torch.load(checkpoint_path, self.device)
            self.model.load_state_dict(state_dict["model"])
        else:
            logging.warning("Loading a policy without checkpoint!")

        # A potential bug is that, the frame stack is not properly reset in
        # a vectorized environment. We assume this will not impact the
        # performance significantly.
        self.frame_stack = FrameStackTensor(
            self.num_envs, self.obs_shape, 4, self.device
        )

    def reset(self):
        self.frame_stack.reset()

    def compute_action(self, obs, deterministic=True):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)
        logits, values = self.model(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            actions = dist.probs.argmax(dim=1, keepdim=True)
        else:
            actions = dist.sample()
        return actions.view(-1, 1)

    def __call__(self, obs):
        self.frame_stack.update(obs)
        action = self.compute_action(self.frame_stack.get(), True)
        if self.num_envs == 1:
            action = action.item()
        else:
            action = action.cpu().numpy()
        return action
