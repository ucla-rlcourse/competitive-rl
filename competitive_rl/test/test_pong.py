from competitive_rl import make_envs

if __name__ == '__main__':
    envs = make_envs(
        env_id="cPong-v0",
        seed=0,
        log_dir="demo",  # this will create a "demo" directory
        num_envs=1,
        asynchronous=False,
        resized_dim=42
    )

    env = envs.envs[0]

    obs = envs.reset()
    env.close()
    print(obs.shape)
    # envs.close()
