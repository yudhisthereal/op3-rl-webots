# op3_ppo_env.py
# Webots controller for PPO (Proximal Policy Optimization) training

from controller import Supervisor
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import config

# ================== SCENARIO SELECTION ==================
# This section is automatically updated by main.py
from scenarios.fall_control import FallControl

SCENARIO_CLASS = FallControl

# ================== PPO AGENT IMPLEMENTATION ==================
class PPOActor(nn.Module):
    """Actor network for PPO."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, act_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))
    
    def forward(self, x):
        mean = self.net(x)
        return mean, self.log_std.expand_as(mean)

class PPOCritic(nn.Module):
    """Critic network for PPO."""
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class PPOAgent:
    """PPO agent implementation."""
    
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.actor = PPOActor(obs_dim, act_dim)
        self.critic = PPOCritic(obs_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LR_CRITIC)
        
        self.gamma = config.GAMMA
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01
        self.num_epochs = 10
        self.batch_size = 64
        
        self.device = torch.device("cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        self.buffer = []
        
    def get_action(self, obs, deterministic=False):
        """Get action from actor network."""
        obs_t = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.actor(obs_t)
            std = torch.exp(log_std)
            
        if deterministic:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action = torch.tanh(action)  # Ensure action is in [-1, 1]
        
        action = action.squeeze(0).cpu().numpy()
        return np.clip(action, -1.0, 1.0)
    
    def store_transition(self, obs, action, reward, next_obs, done):
        """Store a transition in the buffer."""
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def clear_buffer(self):
        """Clear the replay buffer."""
        self.buffer = []
    
    def compute_gae(self, values, rewards, dones, next_values):
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
        
        return advantages
    
    def update(self):
        """Update actor and critic networks using PPO."""
        if len(self.buffer) < self.batch_size:
            return
        
        # Convert buffer to numpy arrays
        obs, actions, rewards, next_obs, dones = zip(*self.buffer)
        
        obs = np.array(obs)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_obs = np.array(next_obs)
        dones = np.array(dones)
        
        # Compute values
        obs_t = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            values = self.critic(obs_t).squeeze().cpu().numpy()
            next_obs_t = torch.FloatTensor(next_obs[-1:]).to(self.device)
            next_value = self.critic(next_obs_t).squeeze().cpu().numpy()
        
        # Compute advantages
        advantages = self.compute_gae(values, rewards, dones, next_value)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device).unsqueeze(1)
        advantages_t = torch.FloatTensor(advantages).to(self.device).unsqueeze(1)
        
        # Store old log probs
        with torch.no_grad():
            old_mean, old_log_std = self.actor(obs_t)
            old_std = torch.exp(old_log_std)
            old_dist = torch.distributions.Normal(old_mean, old_std)
            old_log_probs = old_dist.log_prob(actions_t).sum(dim=1, keepdim=True)
        
        # PPO update for multiple epochs
        for epoch in range(self.num_epochs):
            indices = torch.randperm(len(obs_t))
            
            for start in range(0, len(obs_t), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_obs = obs_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Get current policy
                mean, log_std = self.actor(batch_obs)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                
                # New log probs
                new_log_probs = dist.log_prob(batch_actions).sum(dim=1, keepdim=True)
                
                # Ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Policy loss
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Value loss
                values_pred = self.critic(batch_obs)
                value_loss = nn.functional.mse_loss(values_pred, batch_returns)
                
                # Entropy bonus
                entropy = dist.entropy().mean()
                
                # Total loss
                total_loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
                
                # Update
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        self.clear_buffer()
    
    def save(self, filepath):
        """Save the agent's networks to a file."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
        }, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load an agent from a saved checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        agent = cls(checkpoint['obs_dim'], checkpoint['act_dim'])
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        return agent

# ================== MAIN TRAINING LOOP ==================
if __name__ == "__main__":
    robot = Supervisor()
    scenario = SCENARIO_CLASS(robot, timestep=config.TIMESTEP)
    
    # Create agent
    agent = PPOAgent(
        obs_dim=scenario.get_obs_dim(),
        act_dim=scenario.get_act_dim()
    )
    
    print("Starting PPO training...")
    
    # Training loop
    episode_rewards = []
    
    for ep in range(1, config.MAX_EPISODES + 1):
        obs = scenario.reset()
        total_reward = 0.0
        
        for step in range(config.MAX_STEPS):
            # Get action
            action = agent.get_action(obs)
            
            # Apply action
            scenario.apply_action(action)
            scenario.step()
            
            # Get next observation and reward
            next_obs = scenario.get_observation()
            reward, done, termination_reason = scenario.compute_reward(obs, action, next_obs, step + 1)
            
            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done)
            
            total_reward += reward
            obs = next_obs
            
            if done:
                break
        
        # Update agent after episode
        agent.update()
        
        episode_rewards.append(total_reward)
        
        # Print progress
        if ep % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {ep}/{config.MAX_EPISODES} | Avg Reward (last 10): {avg_reward:.3f}")
        
        # Save checkpoint periodically
        if ep % 100 == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"ppo_checkpoint_ep{ep}.pt")
            agent.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(config.CHECKPOINT_DIR, "ppo_final.pt")
    agent.save(final_path)
    print(f"Final model saved: {final_path}")
    
    # Exit Webots
    robot.simulationQuit(0)