# ddpg_agent.py
# DDPG (Deep Deterministic Policy Gradient) agent implementation

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import config


class MLP(nn.Module):
    """Multi-layer perceptron network."""
    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class DDPG:
    """DDPG agent with actor-critic architecture."""
    
    def __init__(self, obs_dim, act_dim):
        self.actor = MLP(obs_dim, act_dim)
        self.actor_target = MLP(obs_dim, act_dim)
        self.critic = MLP(obs_dim + act_dim, 1)
        self.critic_target = MLP(obs_dim + act_dim, 1)
        
        # Initialize target networks with same weights as main networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.LR_ACTOR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.LR_CRITIC)
        self.buffer = []
        self.device = torch.device("cpu")

    def store(self, transition):
        """Store a transition in the replay buffer."""
        self.buffer.append(transition)
        if len(self.buffer) > config.BUFFER_SIZE:
            self.buffer.pop(0)

    def sample(self):
        """Sample a batch from the replay buffer."""
        idx = np.random.choice(len(self.buffer), config.BATCH_SIZE, replace=False)
        batch = [self.buffer[i] for i in idx]
        obs, act, rew, next_obs, done = map(np.stack, zip(*batch))
        return map(lambda x: torch.tensor(x, dtype=torch.float32, device=self.device),
                   (obs, act, rew, next_obs, done))

    def update(self):
        """Update actor and critic networks using DDPG algorithm."""
        if len(self.buffer) < config.BATCH_SIZE:
            return
        
        obs, act, rew, next_obs, done = self.sample()

        # Critic update
        with torch.no_grad():
            next_act = self.actor_target(next_obs)
            target_q = self.critic_target(torch.cat([next_obs, next_act], dim=1))
            y = rew.unsqueeze(1) + config.GAMMA * (1 - done.unsqueeze(1)) * target_q
        
        q = self.critic(torch.cat([obs, act], dim=1))
        critic_loss = nn.functional.mse_loss(q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        actor_loss = -self.critic(torch.cat([obs, self.actor(obs)], dim=1)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft target update
        for t, s in zip(self.actor_target.parameters(), self.actor.parameters()):
            t.data.copy_(t.data * (1 - config.TAU) + s.data * config.TAU)
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.copy_(t.data * (1 - config.TAU) + s.data * config.TAU)
    
    def get_action(self, obs, add_noise=True):
        """
        Get action from the actor network.
        
        Args:
            obs: Observation (numpy array)
            add_noise: Whether to add exploration noise
            
        Returns:
            Action (numpy array)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = self.actor(obs_t).numpy()
        
        if add_noise:
            action += np.random.normal(0, config.EXPLORATION_NOISE, size=action.shape)
        
        return action
    
    def save(self, filepath):
        """Save the agent's networks to a file."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'obs_dim': self.actor.net[0].in_features,
            'act_dim': self.actor.net[-1].out_features,
        }, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load an agent from a saved checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        agent = cls(checkpoint['obs_dim'], checkpoint['act_dim'])
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        return agent

