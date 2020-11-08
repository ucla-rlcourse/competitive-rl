from competitive_rl.car_racing import make_car_racing

if __name__ == '__main__':
    envs = make_car_racing("cCarRacing-v0", 0, 0)()
    obs = envs.reset()
    for _ in range(10000):
        _, _, d, _ = envs.step(envs.action_space.sample())
        # envs.render("human")
        if d:
            envs.reset()
    print("Return obs shape: ", obs.shape)
    print("Return obs shape: ", obs.shape)
