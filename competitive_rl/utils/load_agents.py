"""
This file defines a set of built-in agents that can be used as opponent in
training and evaluation.

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""

# from core.ppo_trainer import PPOTrainer, ppo_config
# from core.utils import FrameStackTensor
# from make_envs import make_envs
from competitive_rl.pong.utils import FrameStackTensor

# BUILTIN_AGENT_NAMES = [
#     "RANDOM",
#     "WEAK",
#     "MEDIUM",
#     "STRONG",
#     "RULE_BASED",
#     "ALPHA_PONG"  # Boss-level agent, not used for training
# ]


# def get_builtin_agent_names():
#     return BUILTIN_AGENT_NAMES


class PolicyAPI:
    """
    This class wrap an agent into a callable function that return action given
    an raw observation or a batch of raw observations from environment.

    This function maintain a frame stacker so that the user can securely use it.
    A reset function is provided so user can refresh the frame stacker when
    an episode is ended.
    """

    def __init__(self, num_envs=1, log_dir="", suffix="",
                 use_light_model=False):
        self.resized_dim = 42
        env = make_envs(num_envs=1, resized_dim=self.resized_dim)
        self.obs_shape = env.observation_space.shape



        self.agent = PPOTrainer(env, ppo_config,
                                use_light_model=use_light_model)



        if log_dir:  # log_dir is None only in testing
            self.agent.load_w(log_dir, suffix)
        self.num_envs = num_envs
        self.reset()

    def reset(self):
        # A potential bug is that, the frame stack is not properly reset in
        # a vectorized environment. We assume this will not impact the
        # performance significantly.
        self.frame_stack = FrameStackTensor(
            self.num_envs, self.obs_shape, 4, self.agent.device
        )

    def __call__(self, obs):
        self.frame_stack.update(obs)
        action = self.agent.compute_action(self.frame_stack.get(), True)[1]
        if self.num_envs == 1:
            action = action.item()
        else:
            action = action.cpu().numpy()
        return action

#
# def get_rule_based_policy(num_envs=1):
#     if num_envs == 1:
#         return lambda _: CHEAT_CODES
#     else:
#         return lambda _: [CHEAT_CODES] * num_envs
#
#
# def get_random_policy(num_envs=1):
#     if num_envs == 1:
#         return lambda obs: np.random.randint(3)
#     else:
#         return lambda obs: [np.random.randint(3) for _ in range(num_envs)]
#
#
# def get_compute_action_function(agent_name, num_envs=1):
#     if agent_name == "STRONG":
#         return PolicyAPI(
#             num_envs=num_envs,
#             log_dir=osp.join(osp.dirname(__file__), "rlplatform", "buildIn"),
#             suffix="strong", use_light_model=False
#         )
#     if agent_name == "MEDIUM":
#         return PolicyAPI(
#             num_envs=num_envs,
#             log_dir=osp.join(osp.dirname(__file__), "rlplatform", "buildIn"),
#             suffix="medium",
#             use_light_model=True
#         )
#     if agent_name == "ALPHA_PONG":
#         return PolicyAPI(
#             num_envs=num_envs,
#             log_dir=osp.join(osp.dirname(__file__), "rlplatform", "buildIn"),
#             suffix="alphapong",
#             use_light_model=False
#         )
#     if agent_name == "WEAK":
#         return PolicyAPI(
#             num_envs=num_envs,
#             log_dir=osp.join(osp.dirname(__file__), "rlplatform", "buildIn"),
#             suffix="weak",
#             use_light_model=True
#         )
#     if agent_name == "RANDOM":
#         return get_random_policy(num_envs)
#     if agent_name == "RULE_BASED":
#         return get_rule_based_policy(num_envs)
#     raise ValueError("Unknown agent name: {}".format(agent_name))
