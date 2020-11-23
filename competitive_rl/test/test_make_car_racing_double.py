from competitive_rl.car_racing import make_car_racing_double

if __name__ == '__main__':
    e = make_car_racing_double(0, 0, 4, None)()
    e.reset()
    e.step(e.action_space.sample())
    e.close()
