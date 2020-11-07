from competitive_rl.car_racing import make_car_racing

if __name__ == '__main__':
    envs = make_car_racing("cCarRacing-v0", 0, 0)()
    obs = envs.reset()
    print("Return obs shape: ", obs.shape)
    obs = envs.render(mode="rgb_array")
    print("Return obs shape: ", obs.shape)
