import gym

from competitive_rl.register import register_competitive_envs
from competitive_rl.utils import DummyVecEnv, SubprocVecEnv
from competitive_rl.utils.atari_wrappers import WrapPyTorch, MultipleFrameStack

register_competitive_envs()


def make_competitive_car_racing(
        opponent_policy, seed=0, num_envs=3, asynchronous=False, frame_stack=4, action_repeat=None
):
    assert callable(opponent_policy)
    asynchronous = asynchronous and num_envs > 1

    class CarRacingWrapper(gym.Wrapper):
        def __init__(self, envs):
            super(CarRacingWrapper, self).__init__(envs)
            self.opponent_policy = opponent_policy
            self.opponent_action = None
            self.action_space = self.action_space[0]

        def step(self, action):
            o, r, d, i = self.env.step({0: action, 1: self.opponent_action})
            self.opponent_action = self.opponent_policy(o[1])
            return o[0], r[0], d[0], i[0]

        def reset(self, *args, **kwargs):
            o = self.env.reset(*args, **kwargs)
            self.opponent_action = self.opponent_policy(o[1])
            return o[0]

    def _make(env_id, seed, rank, frame_stack=None, action_repeat=None):
        assert "CarRacing" in env_id

        def _thunk():
            env = gym.make(env_id, action_repeat=action_repeat)
            env.seed(seed + rank)
            if frame_stack is not None:
                env = MultipleFrameStack(env, frame_stack)
            env = WrapPyTorch(env)
            env = CarRacingWrapper(env)
            return env

        return _thunk

    envs = [_make("cCarRacingDouble-v0", seed, i, frame_stack=frame_stack) for i in range(num_envs)]
    if asynchronous:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)
    return envs


if __name__ == '__main__':
    e = make_competitive_car_racing(lambda o: [0, 0, 1], asynchronous=False)
    e.reset()
    for _ in range(100):
        e.envs[0].render("human")
        e.step([[-0.5, 1, 0] for _ in range(3)])
