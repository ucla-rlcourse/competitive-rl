from competitive_rl.make_envs import make_envs

if __name__ == '__main__':
    envs = make_envs("cCarRacing-v0", num_envs=1, asynchronous=False, resized_dim=84)
    obses = envs.reset()
