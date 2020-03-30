from gym.envs.registration import register

from competitive_pong.competitive_pong_env import PongSinglePlayerEnv, \
    PongDoublePlayerEnv


def register_competitive_envs():
    register(
        id="CompetitivePong-v0",
        entry_point=PongSinglePlayerEnv,
        kwargs=dict(
            max_num_rounds=21
        )
    )
    register(
        id="CompetitivePongDouble-v0",
        entry_point=PongDoublePlayerEnv,
        kwargs=dict(
            max_num_rounds=21
        )
    )
    print(
        "Register CompetitivePong-v0 and CompetitivePongDouble-v0 "
        "environments.")
