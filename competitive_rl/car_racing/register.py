import math

import gym
from gym.envs.registration import register

from competitive_rl.car_racing.car_racing_multi_players import CarRacing


def register_competitive_envs():
    register(
        id="cCarRacing-v0",
        entry_point=CarRacing,
        max_episode_steps=1000,
        reward_threshold=900
    )
    print(
        "Register car_racing_multiple_players environments.")

def log(x):
    return math.log(x,2)




if __name__ == "__main__":
    #i = 10 * log(10) + 10*log(10)
    #i = -9 * (1 / 1558) * log((1 / 1558))
    #i += -18 * ((5 / 1558) * log(5 / 1558))
    i = -9 * (1 / 1558) * log(1/1558) \
    -18 * ((5 / 1558) * log(5 / 1558)) \
    - 9 * ((7 / 1558) * log(7 / 1558)) \
    - 6 * ((6 / 1558) * log(6 / 1558)) \
    - 9 * ((25 / 1558) * log(25 / 1558)) \
    - 9 * ((35 / 1558) * log(35 / 1558)) \
    - 6 * ((30 / 1558) * log(30 / 1558)) \
    - 3 * ((2 / 1558) * log(2 / 1558)) \
    - 3 * ((10 / 1558) * log(10 / 1558)) \
    - 2 * ((14 / 1558) * log(14 / 1558)) \
    - 1 * ((12 / 1558) * log(12 / 1558)) \
    - 3 * ((42 / 1558) * log(42 / 1558))\
    - ((36 / 1558) * log(36 / 1558)) \
    - 3 * ((8 / 1558) * log(8 / 1558)) \
    - 3 * ((40 / 1558) * log(40 / 1558))\
    - 2 * ((56 / 1558) * log(56 / 1558)) \
    - ((48 / 1558) * log(48 / 1558)) \
    - 2 * ((49 / 1558) * log(49 / 1558))
    sum = 1+1+5+5+1+0+7+7+5+6
    me = -((1/sum) * log(1/sum) + (1/sum) * log(1/sum) + (5/sum) * log(5/sum) + (5/sum) * log(5/sum) + (1/sum) * log(1/sum) +
           (7/sum) * log(7/sum) + (7/sum) * log(7/sum) + (5/sum) * log(5/sum) + (6/sum) * log(6/sum))
    sum =1+1+5+5+1+2+6+8+7+5
    him = -((1/sum) * log(1/sum) + (1/sum) * log(1/sum) + (5/sum) * log(5/sum) + (5/sum) * log(5/sum) + (1/sum) * log(1/sum) +
           (2/sum) * log(2/sum) + (6/sum) * log(6/sum) + (8/sum) * log(8/sum) + (7/sum) * log(7/sum)+ (5/sum) * log(5/sum))
    s = -9 * (1 / 1558) \
    -18 * ((5 / 1558))  \
    - 9 * ((7 / 1558))  \
    - 6 * ((6 / 1558))  \
    - 9 * ((25 / 1558))  \
    - 9 * ((35 / 1558)) \
    - 6 * ((30 / 1558)) \
    - 3 * ((2 / 1558) ) \
    - 3 * ((10 / 1558)) \
    - 2 * ((14 / 1558) ) \
    - 1 * ((12 / 1558)) \
    - 3 * ((42 / 1558))\
    - ((36 / 1558) ) \
    - 3 * ((8 / 1558)) \
    - 3 * ((40 / 1558))\
    - 2 * ((56 / 1558)) \
    - ((48 / 1558)) \
    - 2 * ((49 / 1558))

    print(i,s,me, sum,him, him + me - i)