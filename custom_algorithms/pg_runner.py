# custom_algorithms/my_pg_runner.py

import os
import torch
import torch.optim as optim
from datetime import datetime
from custom_algorithms.policy import PolicyNetwork
from torch.utils.tensorboard import SummaryWriter

class PGRunner:
    def __init__(self, env, train_cfg):
        # Extract the environment and training configuration.
        self.env = env
        self.device = env.device
        self.obs_dim = env.num_obs
        self.act_dim = env.num_actions

        # Create the policy network and optimizer.
        self.policy = PolicyNetwork(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=train_cfg.runner.learning_rate)

        # Set the training parameters.
        self.gamma = train_cfg.runner.gamma
        self.max_steps = train_cfg.runner.max_steps_per_iter
        self.num_envs = self.env.num_envs

        # Create a directory for logging training data and visualizing it in TensorBoard.
        self.log_dir = os.path.join("logs", "custom_go2", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(os.path.join(self.log_dir, "tensorboard"), exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def rollout(self):
        # obs_buf = torch.zeros((self.max_steps, self.num_envs, self.obs_dim), device=self.device)
        logp_buf = torch.zeros((self.max_steps, self.num_envs), device=self.device)
        reward_buf = torch.zeros((self.max_steps, self.num_envs), device=self.device)

        # Get observations from self.env.num_envs environments. 
        obs = self.env.get_observations()

        # Keep track of the environments that are done
        active_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)  # All environments are active at the start

        # Set the environment's max episode length to the max_steps.
        self.env.max_episode_length = self.max_steps

        for step in range(self.max_steps):
            # Input the observations into the policy neural network.
            # Returns a distribution over the action space.
            dist = self.policy(obs)

            # Sample from the action distribution. 
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1)  # Sum over action dims

            # Step in the vectorized environment
            with torch.no_grad():
                obs, _, rewards, dones, _ = self.env.step(actions)

            # If any environment is done, set the corresponding active_mask to 0
            active_mask[dones] = 0

            # Record data
            logp_buf[step, :] = log_probs
            reward_buf[step, active_mask] = torch.tensor(rewards[active_mask], device=self.device, dtype=torch.float32)

            # If all elements of active_mask are False, break the loop
            if not active_mask.any():
                max_step = step
                print(f"All environments are done at step {step}.")
                break
            else:
                max_step = step

        return logp_buf, reward_buf, max_step

    def compute_returns(self, rewards):

        # Initialize the returns tensor.
        returns = torch.zeros_like(rewards)
        G = torch.zeros(self.num_envs, device=self.device)

        # Comute the discounted sum of rewards.
        for t in reversed(range(rewards.shape[0])):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        return returns

    def update_policy(self, logps, returns):

        # Normalize across all returns (optional but helpful)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy loss: REINFORCE objective
        loss = -torch.mean(logps * returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def learn(self, num_learning_iterations):
        # For num_learning_iterations, run the policy gradient algorithm.
        for it in range(num_learning_iterations):
            # Get the log probabilities and rewards from the rollout.
            logps, rewards, max_step = self.rollout()

            # Compute the returns and the loss. 
            returns = self.compute_returns(rewards)
            loss = self.update_policy(logps, returns)

            # Log the mean episode reward
            ep_rewards = rewards.sum(dim=0).mean().item()
            print(f"[PG Iter {it}] Mean Episode Reward: {ep_rewards:.2f}")
            self.writer.add_scalar("Reward/MeanEpisodeReward", ep_rewards, it)

            # Log the mean log probability, the max step reached, and the loss.
            mean_logp = logps.mean().item()
            self.writer.add_scalar("Policy/MeanLogProb", mean_logp, it)
            self.writer.add_scalar("Rollout/MaxStep", max_step, it)
            self.writer.add_scalar("Loss/PolicyLoss", loss, it)

            # Save the model every 50 iterations.
            if it % 50 == 0:
                save_path = os.path.join(self.log_dir, 'checkpoints', f"model_{it}.pth")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.policy.state_dict(), save_path)
                print(f"Saved policy to: {save_path}")
        
        # Save trained model
        save_path = os.path.join(self.log_dir, 'checkpoints', 'model.pth')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.policy.state_dict(), save_path)
        print(f"Saved policy to: {save_path}")

        self.writer.close()
