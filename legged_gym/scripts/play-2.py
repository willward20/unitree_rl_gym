# pg_play.py
import os
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import torch

from custom_algorithms.policy import PolicyNetwork


def load_policy(obs_dim, act_dim, device, model_path):
    """ Load a pre-trained policy from a file. """
    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    return policy

def test(args):

    # Get the training configuration for the task.
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    
    # Overide parameters during testing (similar to deploying PPO). 
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.env.test = True
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # Create the isaac gym environment.
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    obs_dim = obs.shape[1]
    act_dim = env.num_actions

    # Load a pre-trained neural network policy. 
    model_path = os.path.join('logs/custom_go2/2025-04-27_16-51-52/checkpoints/model.pth')
    policy = load_policy(obs_dim, act_dim, env.device, model_path)

    # Store the rewards for each environment at each step.
    rewards = torch.zeros([int(env.max_episode_length), env.num_envs], device=env.device)

    # Rollout
    for i in range(int(env.max_episode_length)):
        with torch.no_grad():
            # Sample actions from a policy distribution. 
            dist = policy(obs)
            actions = dist.sample()
        
        # Apply the actions to the environment. 
        obs, _, rews, _, _ = env.step(actions)

        # Store the rewards for each environment at this step.
        rewards[i,:] = rews

        # Print the current step every 100 steps
        if i % 100 == 0:
            print(f"Step: {i}/{int(env.max_episode_length)}")

    # Take an average of the rewards over all environments at each step
    rewards = torch.mean(rewards, dim=1)

    # Plot the average rewards of each environment over each step using pyplot
    import matplotlib.pyplot as plt
    plt.plot(rewards.cpu().numpy(), label='Average Reward')
    plt.xlabel('Step')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards over steps')
    plt.show()

if __name__ == "__main__":
    args = get_args()
    test(args)
