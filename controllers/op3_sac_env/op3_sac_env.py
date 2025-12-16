# op3_sac_env.py
# Webots controller for SAC (Soft Actor-Critic) training

from controller import Supervisor
import time
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config

CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CONTROLLER_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)


# ================== SCENARIO SELECTION ==================
# This section is automatically updated by main.py
from scenarios.fall_control import FallControl

SCENARIO_CLASS = FallControl

# ================== SAC AGENT IMPLEMENTATION ==================
class SACActor(nn.Module):
    """Actor network for SAC with reparameterization trick."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.act_dim = act_dim
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        self.mean_layer = nn.Linear(256, act_dim)
        self.log_std_layer = nn.Linear(256, act_dim)
        
        # Action scaling
        self.action_scale = 1.0
        self.action_bias = 0.0
    
    def forward(self, x):
        hidden = self.net(x)
        mean = self.mean_layer(hidden)
        log_std = self.log_std_layer(hidden)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, x):
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        
        # Log probability with tanh transformation
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Scale action to [-1, 1]
        action = y_t * self.action_scale + self.action_bias
        return action, log_prob

class SACCritic(nn.Module):
    """Critic network for SAC."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        return self.net(x)

class SACAgent:
    """SAC agent implementation."""
    
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Networks
        self.actor = SACActor(obs_dim, act_dim)
        self.critic1 = SACCritic(obs_dim, act_dim)
        self.critic2 = SACCritic(obs_dim, act_dim)
        self.critic1_target = SACCritic(obs_dim, act_dim)
        self.critic2_target = SACCritic(obs_dim, act_dim)
        
        # Initialize target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LR_ACTOR)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.LR_CRITIC)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.LR_CRITIC)
        
        # Hyperparameters
        self.gamma = config.GAMMA
        self.tau = 0.005
        self.alpha = 0.2  # Temperature parameter
        self.target_entropy = -torch.prod(torch.Tensor([act_dim])).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.LR_ACTOR)
        
        # Replay buffer
        self.buffer = []
        self.buffer_size = 100000
        self.batch_size = 256
        
        self.device = torch.device("cpu")
        self.to(self.device)
    
    def to(self, device):
        self.actor.to(device)
        self.critic1.to(device)
        self.critic2.to(device)
        self.critic1_target.to(device)
        self.critic2_target.to(device)
        self.log_alpha.to(device)
    
    def get_action(self, obs, deterministic=False):
        """Get action from actor network."""
        obs_t = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        
        if deterministic:
            with torch.no_grad():
                mean, _ = self.actor(obs_t)
                action = torch.tanh(mean)
        else:
            with torch.no_grad():
                action, _ = self.actor.sample(obs_t)
        
        action = action.squeeze(0).cpu().numpy()
        return np.clip(action, -1.0, 1.0)
    
    def store_transition(self, obs, action, reward, next_obs, done):
        """Store a transition in the replay buffer."""
        self.buffer.append((obs, action, reward, next_obs, done))
        
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def sample_batch(self):
        """Sample a batch from the replay buffer."""
        indices = np.random.randint(0, len(self.buffer), self.batch_size)
        batch = [self.buffer[i] for i in indices]
        
        obs, actions, rewards, next_obs, dones = zip(*batch)
        
        obs = torch.FloatTensor(np.array(obs)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device).unsqueeze(1)
        next_obs = torch.FloatTensor(np.array(next_obs)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device).unsqueeze(1)
        
        return obs, actions, rewards, next_obs, dones
    
    def update(self):
        """Update SAC networks."""
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample batch
        obs, actions, rewards, next_obs, dones = self.sample_batch()
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            next_q1 = self.critic1_target(next_obs, next_actions)
            next_q2 = self.critic2_target(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * next_q
        
        # Critic loss
        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(obs)
        q1_new = self.critic1(obs, new_actions)
        q2_new = self.critic2(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        # Soft update target networks
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath):
        """Save the agent's networks to a file."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'log_alpha': self.log_alpha,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
        }, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load an agent from a saved checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        agent = cls(checkpoint['obs_dim'], checkpoint['act_dim'])
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        agent.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        agent.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        agent.log_alpha = checkpoint['log_alpha']
        agent.alpha = agent.log_alpha.exp().item()
        return agent

# ================== MAIN TRAINING LOOP ==================
if __name__ == "__main__":
    robot = Supervisor()
    scenario = SCENARIO_CLASS(robot, timestep=config.TIMESTEP, algorithm='sac')
    
    # Create agent
    agent = SACAgent(
        obs_dim=scenario.get_obs_dim(),
        act_dim=scenario.get_act_dim()
    )
    
    print("Starting SAC training...")
    
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
            
            # Update agent
            agent.update()
            
            total_reward += reward
            obs = next_obs
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        
        # Print progress
        if ep % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {ep}/{config.MAX_EPISODES} | Avg Reward (last 10): {avg_reward:.3f}")
        
        # Save checkpoint periodically
        if ep % 100 == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"sac_checkpoint_ep{ep}.pt")
            agent.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(config.CHECKPOINT_DIR, "sac_final.pt")
    agent.save(final_path)
    print(f"Final model saved: {final_path}")
    
    # Exit Webots
    robot.simulationQuit(0)