from competitive_rl.car_racing import make_car_racing, register_car_racing
import gym

register_car_racing()


def test_blackbox():
    e = gym.make("cCarRacing-v0")
    e.reset()
    for _ in range(100):
        _, _, d, _ = e.step(e.action_space.sample())
        e.render("human")
        if d:
            e.reset()
    e.close()


def test_action_repetition():
    e = gym.make("cCarRacing-v0", action_repeat=1)
    e.seed(0)
    e.reset()
    for _ in range(100 * 2):
        e.render("human")
        ret = e.step([0.0, 1])
    print(ret)
    e.close()

    e = gym.make("cCarRacing-v0", action_repeat=5)
    e.seed(0)
    e.reset()
    for _ in range(20 * 2):
        e.render("human")
        ret = e.step([0.0, 1])
    print(ret)
    e.close()


if __name__ == '__main__':
    test_blackbox()
    test_action_repetition()
