from competitive_rl.car_racing.car_racing_multi_players import CarRacing

if __name__ == "__main__":
    env = CarRacing(num_player=1)
    # example: env.reset(use_local_track="./track/test.json",record_track_to="")
    # example: env.reset(use_local_track="",record_track_to="./track")
    env.reset(use_local_track="", record_track_to="")
    while True:
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        if env.show_all_car_obs:
            env.show_all_obs(observation)
