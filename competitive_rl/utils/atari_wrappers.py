from collections import deque

import cv2
import gym
import numpy as np
from gym import spaces

gym.logger.set_level(40)
cv2.ocl.setUseOpenCL(False)


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)
        if isinstance(self.env.unwrapped.observation_space, spaces.Tuple):
            self.observation_space = spaces.Tuple(
                [self.observation_space, self.observation_space])

    def observation(self, observation):
        if isinstance(observation, tuple):
            return tuple(self.parse_single_frame(f) for f in observation)
        elif isinstance(observation, dict):
            return {k: self.parse_single_frame(f) for k, f in observation.items()}
        else:
            return self.parse_single_frame(observation)

    def parse_single_frame(self, frame):
        assert frame.ndim == 3
        return frame.transpose(2, 0, 1)


def make_env_a2c_atari(env_id, seed, rank, log_dir, resized_dim=84, frame_stack=None):
    def _thunk():
        env = make_atari(env_id)
        env.seed(seed + rank)
        # if log_dir is not None:
        #     env = Monitor(env, os.path.join(log_dir, str(rank)))
        env = wrap_deepmind(env, resized_dim)
        if frame_stack is not None:
            env = FrameStack(env, frame_stack)
        env = WrapPyTorch(env)
        return env

    return _thunk


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.

        :param env: (Gym Environment) the environment to wrap
        :param noop_max: (int) the maximum value of no-ops to run
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        # assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        Return only every `skip`-th frame (frameskipping)

        :param env: (Gym Environment) the environment
        :param skip: (int) number of `skip`-th frame
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)

        if isinstance(env.observation_space, gym.spaces.Tuple):
            observation_space = env.observation_space[0]
        else:
            observation_space = env.observation_space
        self.multi_agent = isinstance(self.env.action_space, spaces.Tuple)
        if self.multi_agent:
            self._obs_buffer = [
                np.zeros(
                    (2,) + observation_space.shape,
                    dtype=observation_space.dtype
                )
                for _ in range(len(self.env.action_space))
            ]
        else:
            self._obs_buffer = np.zeros((2,) + observation_space.shape,
                                        dtype=observation_space.dtype)
        self._skip = skip

    def step(self, action):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation,
        reward, done, information
        """
        if self.multi_agent:
            total_reward = [0.0] * len(action)
        else:
            total_reward = 0.0

        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if self.multi_agent:
                if i == self._skip - 2:
                    self._obs_buffer[0][0] = obs[0]
                    self._obs_buffer[1][0] = obs[1]
                if i == self._skip - 1:
                    self._obs_buffer[0][1] = obs[0]
                    self._obs_buffer[1][1] = obs[1]
                for a_i, r_i in enumerate(reward):
                    total_reward[a_i] += r_i
            else:
                if i == self._skip - 2:
                    self._obs_buffer[0] = obs
                if i == self._skip - 1:
                    self._obs_buffer[1] = obs
                total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter

        if self.multi_agent:
            max_frame = tuple(buff.max(axis=0) for buff in self._obs_buffer)
        else:
            max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._steps = 0

    def reset(self, **kwargs):
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """ Bin reward to {+1, 0, -1} by its sign. """
        observation, reward, done, info = self.env.step(action)
        self._steps += 1
        info["real_reward"] = reward
        info["num_steps"] = self._steps
        return observation, np.sign(reward), done, info


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, resized_dim=84):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = resized_dim
        self.height = resized_dim

        if not isinstance(env.observation_space, gym.spaces.Tuple):
            dtype = env.observation_space.dtype
        else:
            dtype = env.observation_space[0].dtype
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1),
                                            dtype=dtype)

    def observation(self, frame):
        """
        returns the current observation from a frame

        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        if isinstance(frame, tuple):
            return tuple(self.parse_single_frame(f) for f in frame)
        else:
            return self.parse_single_frame(frame)

    def parse_single_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Stack n_frames last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames

        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * n_frames),
            dtype=env.observation_space.dtype
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n_frames
        return np.stack(self.frames, axis=2).squeeze(-1)


class MultipleFrameStack(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Stack n_frames last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames

        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        from collections import defaultdict
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames_dict = defaultdict(lambda: deque([], maxlen=n_frames))
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * n_frames),
            dtype=env.observation_space.dtype
        )

    def reset(self):
        obs = self.env.reset()
        assert isinstance(obs, dict)
        for k in obs:
            for _ in range(self.n_frames):
                self.frames_dict[k].append(obs[k])
        return self._get_ob(obs.keys())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        for k in obs:
            self.frames_dict[k].append(obs[k])
        return self._get_ob(obs.keys()), reward, done, info

    def _get_ob(self, keys):
        ret = dict()
        for k in keys:
            ret[k] = np.stack(self.frames_dict[k], axis=2).squeeze(-1)
        return ret


def make_atari(env_id):
    """
    Create a wrapped atari Environment

    :param env_id: (str) the environment ID
    :return: (Gym Environment) the wrapped atari environment
    """
    env = gym.make(env_id)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(env, resized_dim=84, clip_rewards=True):
    """
    Configure environment for DeepMind-style Atari.

    :param env: (Gym Environment) the atari environment
    :param clip_rewards: (bool) wrap the reward clipping wrapper
    :return: (Gym Environment) the wrapped atari environment
    """
    env = WarpFrame(env, resized_dim)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env
