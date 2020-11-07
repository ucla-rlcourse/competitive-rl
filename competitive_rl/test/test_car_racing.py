from competitive_rl import make_envs

if __name__ == '__main__':
    envs = make_envs(
        env_id="cCarRacing-v0",
        seed=0,
        log_dir="demo",  # this will create a "demo" directory
        num_envs=5,
        asynchronous=True,
        resized_dim=42
    )
    obs = envs.reset()
    print(obs.shape)
    envs.close()
