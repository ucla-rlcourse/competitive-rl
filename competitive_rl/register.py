from competitive_rl.car_racing import register_car_racing
from competitive_rl.pong import register_pong


def register_competitive_envs():
    register_pong()
    register_car_racing()
