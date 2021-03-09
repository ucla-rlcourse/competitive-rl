import argparse
import shutil

from competitive_rl import get_builtin_agent_names, make_envs, \
    get_compute_action_function, evaluate_two_policies

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", default="RULE_BASED", type=str,
                        help="Left agent names, must in {}.".format(
                            get_builtin_agent_names()))
    parser.add_argument("--right", default="RULE_BASED", type=str,
                        help="Right agent names, must in {}.".format(
                            get_builtin_agent_names()))
    parser.add_argument("--num-episodes", "-N", default=3, type=int,
                        help="Number of episodes to run.")
    args = parser.parse_args()

    # collect builtin agents
    agent_names = get_builtin_agent_names() + ["MY_AGENT"]
    print("Agent names: ", agent_names)
    print("Your chosen agents: left - {}, right - {}".format(
        args.left, args.right))
    assert args.left in agent_names, agent_names
    assert args.right in agent_names, agent_names

    # create env and setup policies
    env = make_envs("cPongDouble-v0", num_envs=1, asynchronous=False, frame_stack=None,
                    log_dir="tmp_vis").envs[0]
    left = get_compute_action_function(args.left)
    right = get_compute_action_function(args.right)

    # evaluate
    result = evaluate_two_policies(
        left, right, env=env, render=True,
        num_episode=args.num_episodes,
        render_interval=0.05  # 20 FPS rendering
    )
    print(result)

    # clear
    env.close()
    shutil.rmtree("tmp_vis")
