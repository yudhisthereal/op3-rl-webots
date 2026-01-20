"""
OP3 DDPG Controller - Fast learning for joint angle control
Trains ShoulderR (or multiple joints) to reach and maintain target angles
Uses DDPG (Deep Deterministic Policy Gradient) algorithm
"""

from controller import Supervisor
import numpy as np
import csv
import os
import sys
import json
import time
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from datetime import datetime

# ============================================================================
# CONSTANTS AND PATHS
# ============================================================================

# Directory constants
CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CONTROLLER_DIR, '..', '..'))
CHECKPOINT_DIR = os.path.join(CONTROLLER_DIR, "checkpoints")
STATS_DIR = os.path.join(CONTROLLER_DIR, "training_stats")
PLOTS_DIR = os.path.join(STATS_DIR, "plots")
FINAL_PLOTS_DIR = os.path.join(STATS_DIR, "final_plots")

# Run directory paths (set by main.py)
RUN_DIR = os.environ.get('RL_RUN_DIR', '')
RUN_LABEL = os.environ.get('RL_RUN_LABEL', 'default')

# Model paths based on run_label to avoid overwriting
FINAL_MODEL_DIR = os.path.join(CONTROLLER_DIR, "runs")
FINAL_MODEL_PATH = os.path.join(FINAL_MODEL_DIR, f"final_{RUN_LABEL}.pt")

# Best model path (inside run directory)
if RUN_DIR:
    BEST_MODEL_PATH = os.path.join(RUN_DIR, f"best_{RUN_LABEL}.pt")
else:
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f"best_{RUN_LABEL}.pt")

# Add project root to path
sys.path.insert(0, PROJECT_ROOT)

# Import logging
from logging_utils import (
    log, log_info, log_warning, log_error, log_success,
    log_debug, log_data, log_exception, log_section,
    start_timer, stop_timer, LogFunction
)

# Import stats manager for proper HDF5 logging
from stats_manager import (
    HDF5StatsLogger, create_stats_logger,
    StageInfo, EpisodeInfo, AgentInfo
)


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def _create_default_config(mode_str):
    """Create default configuration dictionary.
    
    Args:
        mode_str: Either "train" or "test"
        
    Returns:
        dict: Default configuration
    """
    if mode_str == "train":
        return {
            "mode": "train",
            "model_path": "controllers/op3_ddpg/checkpoints/ddpg_final.pt",
            "control_joints": ["ShoulderR"],
            "goal_angles": {
                "ShoulderR": 1.0
            },
            "push_force": {
                "enabled": False,
                "force": 5.0,
                "angle": 0.0,
                "delay_steps": 20
            },
            "initial_state": {
                "translation": [0.0, 0.0, 0.292665],
                "rotation": [0.0, 0.0, 1.0, 0.0],
                "joint_angles": {
                    "ShoulderR": 0.0
                }
            },
            "training": {
                "max_episodes": 1000,
                "max_steps": 200,
                "timestep": 8,
                "ddpg": {
                    "actor_lr": 1e-4,
                    "critic_lr": 1e-3,
                    "gamma": 0.99,
                    "tau": 0.005,
                    "noise_std_start": 0.3,
                    "noise_std_end": 0.01,
                    "noise_decay": 0.995,
                    "batch_size": 64,
                    "replay_buffer_size": 100000
                },
                "reward": {
                    "angle_tolerance": 0.1,
                    "success_reward": 10.0,
                    "angle_error_weight": -1.0,
                    "stability_bonus": 0.1
                },
                "joint_limits": {
                    "ShoulderR": [-1.57, 1.57]
                }
            },
            "early_stopping": {
                "enabled": True,
                "window_size": 50,
                "success_threshold": 1.0,
                "min_episodes": 100
            },
            "checkpoints": {
                "save_every": 100,
                "save_best": True
            }
        }
    else:
        return {
            "mode": "test",
            "model_path": "controllers/op3_ddpg/checkpoints/ddpg_final.pt",
            "control_joints": ["ShoulderR"],
            "goal_angles": {
                "ShoulderR": 1.0
            },
            "push_force": {
                "enabled": False,
                "force": 5.0,
                "angle": 0.0,
                "delay_steps": 20
            },
            "initial_state": {
                "translation": [0.0, 0.0, 0.292665],
                "rotation": [0.0, 0.0, 1.0, 0.0],
                "joint_angles": {
                    "ShoulderR": 0.0
                }
            },
            "test": {
                "max_steps": 30,
                "timestep": 8,
                "render": True
            },
            "reward": {
                "angle_tolerance": 0.1,
                "success_reward": 10.0,
                "angle_error_weight": -1.0,
                "stability_bonus": 0.1
            },
            "joint_limits": {
                "ShoulderR": [-1.57, 1.57]
            }
        }


def load_config():
    """Load configuration from JSON file based on train/test mode.
    
    Uses RL_TRAIN environment variable (set by main.py):
    - RL_TRAIN=true -> config_train.json (training mode)
    - RL_TRAIN not set -> config_test.json (test mode)
    - Falls back to config.json if neither exists
    - Falls back to template if config.json doesn't exist
    - Falls back to default config if template doesn't exist
    
    Returns:
        dict: Configuration dictionary
    """
    with LogFunction("DDPGController", "load_config"):
        log_info("DDPGController", "Loading configuration")
        
        # Determine config file based on mode
        is_train = os.environ.get('RL_TRAIN', '').lower() == 'true'
        
        if is_train:
            config_path = os.path.join(CONTROLLER_DIR, "config_train.json")
            template_path = os.path.join(CONTROLLER_DIR, "config_train.json.template")
            mode_str = "train"
            log_info("DDPGController", "Running in TRAINING mode")
        else:
            config_path = os.path.join(CONTROLLER_DIR, "config_test.json")
            template_path = os.path.join(CONTROLLER_DIR, "config_test.json.template")
            mode_str = "test"
            log_info("DDPGController", "Running in TESTING mode")
        
        # Create directories
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(STATS_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        os.makedirs(FINAL_PLOTS_DIR, exist_ok=True)
        
        # Check if config file exists
        if not os.path.exists(config_path):
            log_warning("DDPGController", f"Config file not found: {config_path}")
            
            # Try fallback to config.json
            fallback_path = os.path.join(CONTROLLER_DIR, "config.json")
            if os.path.exists(fallback_path):
                config_path = fallback_path
                log_info("DDPGController", f"Using fallback config: {config_path}")
            else:
                # Check if template exists
                if template_path and os.path.exists(template_path):
                    log_info("DDPGController", f"Creating config from template: {template_path}")
                    with open(template_path, 'r') as f:
                        config = json.load(f)
                else:
                    # Create default config
                    log_info("DDPGController", "Creating default configuration")
                    config = _create_default_config(mode_str)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                
                # Save the config file
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                log_success("DDPGController", f"Created config file: {config_path}")
        
        # Load the config file
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        log_success("DDPGController", f"Loaded config from: {config_path}")
        log_data("DDPGController", "Config keys", list(config.keys()))
        
        # Update model path to be absolute
        if not os.path.isabs(config.get("model_path", "")):
            config["model_path"] = os.path.join(PROJECT_ROOT, config["model_path"])
        
        log_data("DDPGController", "Model path", config["model_path"])
        
        # Log control joints
        if "control_joints" in config:
            log_data("DDPGController", "Control joints", config["control_joints"])
        
        return config


# Load configuration
CONFIG = load_config()


# ============================================================================
# STATISTICS AND PLOTTING
# ============================================================================




def generate_training_plots(episode_rewards, episode_steps, success_history, 
                           episode_errors, save_dir):
    """Generate and save training plots - matches PPO format."""
    with LogFunction("DDPGController", "generate_training_plots", args=(save_dir,)):
        
        log_info("DDPGController", f"Generating training plots in: {save_dir}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Episode Rewards
        axes[0, 0].plot(episode_rewards, alpha=0.7)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add moving average
        if len(episode_rewards) > 10:
            window = min(50, len(episode_rewards))
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(episode_rewards)), moving_avg, 
                           'r-', linewidth=2, label=f'{window}-ep MA')
            axes[0, 0].legend()
        
        # Plot 2: Episode Steps
        axes[0, 1].plot(episode_steps, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Length (Steps)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Success Rate
        axes[1, 0].plot(success_history, 'g-', linewidth=2)
        success_threshold = CONFIG["early_stopping"]["success_threshold"]
        axes[1, 0].axhline(y=success_threshold, color='r', linestyle='--', alpha=0.7, 
                          label=f'{success_threshold:.0%} Target')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_title(f'Success Rate (Last {CONFIG["early_stopping"]["window_size"]} Episodes)')
        axes[1, 0].set_ylim([0, 1.1])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Average Angle Error
        axes[1, 1].plot(episode_errors, alpha=0.7, color='orange')
        tolerance = CONFIG["training"]["reward"]["angle_tolerance"]
        axes[1, 1].axhline(y=tolerance, color='r', linestyle='--', alpha=0.7, 
                          label=f'Tolerance ({tolerance} rad)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Error (rad)')
        axes[1, 1].set_title('Average Joint Angle Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(save_dir, f'training_plots_{timestamp}.png')
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        log_success("DDPGController", f"Training plots saved to: {plot_path}")
        
        # Save summary data
        summary = {
            'timestamp': timestamp,
            'total_episodes': len(episode_rewards),
            'total_steps': sum(episode_steps),
            'final_success_rate': success_history[-1] if success_history else 0.0,
            'average_reward': float(np.mean(episode_rewards)),
            'best_reward': float(np.max(episode_rewards)),
            'average_steps': float(np.mean(episode_steps)),
            'average_error': float(np.mean(episode_errors)),
            'config': CONFIG,
        }
        
        summary_path = os.path.join(save_dir, f'training_summary_{timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        log_debug("DDPGController", f"Saved training summary: {summary_path}")
        
        return plot_path


# ============================================================================
# NEURAL NETWORKS
# ============================================================================

class Actor(nn.Module):
    """Actor network: observation → action"""
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        log_info("Actor", f"Initializing actor network: input={obs_dim}, output={act_dim}, hidden={hidden_dim}")
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        log_success("Actor", "Actor network initialized")
    
    def forward(self, x):
        log_debug("Actor", f"Forward pass: input shape={x.shape}")
        output = self.net(x)
        log_debug("Actor", f"Forward pass: output shape={output.shape}")
        return output


class Critic(nn.Module):
    """Critic network: (observation, action) → Q-value"""
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        log_info("Critic", f"Initializing critic network: obs={obs_dim}, act={act_dim}, hidden={hidden_dim}")
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        log_success("Critic", "Critic network initialized")
    
    def forward(self, state, action):
        log_debug("Critic", f"Forward pass: state shape={state.shape}, action shape={action.shape}")
        x = torch.cat([state, action], dim=1)
        output = self.net(x)
        log_debug("Critic", f"Forward pass: output shape={output.shape}")
        return output


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """Experience replay buffer for DDPG"""
    def __init__(self, capacity):
        log_info("ReplayBuffer", f"Initializing replay buffer with capacity: {capacity}")
        self.buffer = deque(maxlen=capacity)
        log_success("ReplayBuffer", "Replay buffer initialized")
    
    def push(self, state, action, reward, next_state, done):
        log_debug("ReplayBuffer", f"Pushing experience: reward={reward}, done={done}")
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        log_debug("ReplayBuffer", f"Sampling batch of size: {batch_size} from buffer size: {len(self.buffer)}")
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        log_debug("ReplayBuffer", f"Batch sampled: states shape={np.array(states).shape}")
        
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32).reshape(-1, 1),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32).reshape(-1, 1))
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# DDPG AGENT
# ============================================================================

class DDPGAgent:
    """DDPG agent implementation"""
    
    def __init__(self, obs_dim, act_dim, agent_id=None, parent_id=None):
        """Initialize DDPG agent.
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension
            agent_id: Unique agent identifier (auto-generated if None)
            parent_id: Parent agent ID for lineage tracking
        """
        log_section("DDPGAgent", "INITIALIZING DDPG AGENT")
        
        with LogFunction("DDPGAgent", "__init__", args=(obs_dim, act_dim, agent_id, parent_id)):
            
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            self.agent_id = agent_id or f"ddpg_{id(self)}"
            self.parent_id = parent_id
            ddpg_config = CONFIG["training"]["ddpg"]
            
            log_info("DDPGAgent", f"Agent ID: {self.agent_id}")
            log_info("DDPGAgent", f"Parent ID: {self.parent_id}")
            log_info("DDPGAgent", f"Observation dim: {obs_dim}, Action dim: {act_dim}")
            log_data("DDPGAgent", "DDPG config", ddpg_config)
            
            # Device
            self.device = torch.device("cpu")
            log_info("DDPGAgent", f"Using device: {self.device}")
            
            # Networks
            self.actor = Actor(obs_dim, act_dim).to(self.device)
            self.actor_target = Actor(obs_dim, act_dim).to(self.device)
            self.actor_target.load_state_dict(self.actor.state_dict())
            
            self.critic = Critic(obs_dim, act_dim).to(self.device)
            self.critic_target = Critic(obs_dim, act_dim).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())
            
            # Optimizers
            self.actor_optimizer = optim.Adam(self.actor.parameters(), 
                                             lr=ddpg_config["actor_lr"])
            self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                              lr=ddpg_config["critic_lr"])
            
            # Replay buffer
            self.replay_buffer = ReplayBuffer(ddpg_config["replay_buffer_size"])
            
            # DDPG parameters
            self.gamma = ddpg_config["gamma"]
            self.tau = ddpg_config["tau"]
            self.batch_size = ddpg_config["batch_size"]
            
            # Exploration noise
            self.noise_std = ddpg_config["noise_std_start"]
            self.noise_std_end = ddpg_config["noise_std_end"]
            self.noise_decay = ddpg_config["noise_decay"]
            
            log_info("DDPGAgent", f"Noise: start={self.noise_std}, end={self.noise_std_end}, decay={self.noise_decay}")
            
            # Training step counter
            self.train_step = 0
            
            # Lineage tracking
            self.creation_timestamp = ""
            self.lineage_depth = 0 if parent_id is None else 1
            
            log_success("DDPGAgent", f"DDPG agent initialized: {self.agent_id}")
    
    def get_action(self, state, add_noise=True):
        """Get action from actor network with optional exploration noise."""
        log_debug("DDPGAgent", f"get_action: state shape={state.shape}, add_noise={add_noise}")
        
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().squeeze()
        
        log_debug("DDPGAgent", f"Base action (no noise): {action}")
        
        if add_noise:
            # Add exploration noise (Ornstein-Uhlenbeck would be better, but Gaussian is simpler)
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
            log_debug("DDPGAgent", f"Added noise: std={self.noise_std}, noise={noise}")
        
        log_debug("DDPGAgent", f"Final action: {action}")
        return action
    
    def decay_noise(self):
        """Decay exploration noise over time."""
        old_noise = self.noise_std
        self.noise_std = max(self.noise_std * self.noise_decay, self.noise_std_end)
        log_debug("DDPGAgent", f"Decayed noise: {old_noise:.4f} -> {self.noise_std:.4f}")
    
    def update(self):
        """Update actor and critic networks."""
        log_debug("DDPGAgent", "update called")
        
        if len(self.replay_buffer) < self.batch_size:
            log_debug("DDPGAgent", f"Replay buffer too small: {len(self.replay_buffer)} < {self.batch_size}")
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        log_debug("DDPGAgent", f"Batch tensors: states={states_t.shape}, actions={actions_t.shape}")
        
        # ----------------- Update critic ----------------- #
        with torch.no_grad():
            # Target actions from target actor
            next_actions = self.actor_target(next_states_t)
            # Target Q-values from target critic
            target_q = self.critic_target(next_states_t, next_actions)
            # Compute target: r + γ * Q(s', a') * (1 - done)
            target_q = rewards_t + self.gamma * target_q * (1 - dones_t)
        
        # Current Q-values
        current_q = self.critic(states_t, actions_t)
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # ----------------- Update actor ----------------- #
        # Actor loss: maximize Q(s, π(s)) = minimize -Q(s, π(s))
        actor_actions = self.actor(states_t)
        actor_loss = -self.critic(states_t, actor_actions).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # ----------------- Soft update target networks ----------------- #
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.train_step += 1
        
        log_debug("DDPGAgent", f"Update step {self.train_step}: critic_loss={critic_loss.item():.4f}, actor_loss={actor_loss.item():.4f}")
        
        return critic_loss.item(), actor_loss.item()
    
    def save(self, filepath, agent_id=None):
        """Save the agent's networks to a file.
        
        Args:
            filepath: Path to save checkpoint
            agent_id: Optional agent ID to include in checkpoint
        """
        with LogFunction("DDPGAgent", "save", args=(filepath, agent_id)):
            log_info("DDPGAgent", f"Saving agent to: {filepath}")
            
            checkpoint = {
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_target_state_dict': self.actor_target.state_dict(),
                'critic_target_state_dict': self.critic_target.state_dict(),
                'obs_dim': self.obs_dim,
                'act_dim': self.act_dim,
                'noise_std': self.noise_std,
                'config': CONFIG,
                'agent_id': agent_id or self.agent_id,
                'parent_id': self.parent_id,
                'lineage_depth': self.lineage_depth,
                'creation_timestamp': self.creation_timestamp,
            }
            
            log_data("DDPGAgent", "Checkpoint keys", list(checkpoint.keys()))
            
            torch.save(checkpoint, filepath)
            log_success("DDPGAgent", f"DDPG model saved to {filepath} (agent_id: {checkpoint['agent_id']})")
    
    @classmethod
    def load(cls, filepath, agent_id=None):
        """Load an agent from a saved checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            agent_id: Optional agent ID override
            
        Returns:
            DDPGAgent instance
        """
        log_section("DDPGAgent", "LOADING AGENT FROM CHECKPOINT")
        
        with LogFunction("DDPGAgent", "load", args=(filepath, agent_id)):
            log_info("DDPGAgent", f"Loading agent from: {filepath}")
            
            checkpoint = torch.load(filepath, map_location='cpu')
            log_data("DDPGAgent", "Checkpoint keys", list(checkpoint.keys()))
            
            agent = cls(
                checkpoint['obs_dim'], 
                checkpoint['act_dim'], 
                agent_id=agent_id or checkpoint.get('agent_id'),
                parent_id=checkpoint.get('parent_id')
            )
            
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            agent.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
            agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            agent.noise_std = checkpoint.get('noise_std', 0.1)
            agent.lineage_depth = checkpoint.get('lineage_depth', 0)
            agent.creation_timestamp = checkpoint.get('creation_timestamp', '')
            
            log_success("DDPGAgent", f"DDPG model loaded from {filepath} (agent_id: {agent.agent_id})")
            log_data("DDPGAgent", "Loaded agent info", {
                'agent_id': agent.agent_id,
                'parent_id': agent.parent_id,
                'lineage_depth': agent.lineage_depth,
                'noise_std': agent.noise_std
            })
            
            return agent


# ============================================================================
# TRAINING ENVIRONMENT
# ============================================================================

class ArmTrainingEnvironment:
    """Environment for training joint angle control"""
    
    def __init__(self, robot):
        log_section("ArmTrainingEnvironment", "INITIALIZING TRAINING ENVIRONMENT")
        
        with LogFunction("ArmTrainingEnvironment", "__init__"):
            
            self.robot = robot
            
            # Get robot node for applying forces
            self.robot_node = robot.getSelf()
            
            # Joints to control
            self.control_joints = CONFIG["control_joints"]
            self.goal_angles = CONFIG["goal_angles"]
            self.joint_limits = CONFIG["training"]["joint_limits"]
            self.timestep = CONFIG["training"]["timestep"]
            
            log_info("ArmTrainingEnvironment", f"Control joints: {self.control_joints}")
            log_info("ArmTrainingEnvironment", f"Goal angles: {self.goal_angles}")
            log_info("ArmTrainingEnvironment", f"Timestep: {self.timestep}")
            log_data("ArmTrainingEnvironment", "Joint limits", self.joint_limits)
            
            # Push force configuration
            self.push_force_config = CONFIG.get("push_force", {
                "enabled": False,
                "force": 5.0,
                "angle": 0.0,
                "delay_steps": 20
            })
            
            log_info("ArmTrainingEnvironment", f"Push force enabled: {self.push_force_config['enabled']}")
            
            # Initial state configuration
            self.initial_state = CONFIG.get("initial_state", {
                "translation": [0.0, 0.0, 0.292665],
                "rotation": [0.0, 0.0, 1.0, 0.0],
                "joint_angles": {}
            })
            
            # Reward parameters
            reward_config = CONFIG["training"]["reward"]
            self.angle_tolerance = reward_config["angle_tolerance"]
            self.success_reward = reward_config["success_reward"]
            self.angle_error_weight = reward_config["angle_error_weight"]
            self.stability_bonus = reward_config.get("stability_bonus", 0.0)
            
            log_info("ArmTrainingEnvironment", f"Reward config: tolerance={self.angle_tolerance}, success_reward={self.success_reward}")
            
            # Initialize motors
            self.motors = {}
            self.sensors = {}
            
            # Initialize all joints from initial_state
            all_joints = set(self.control_joints)
            if "joint_angles" in self.initial_state:
                all_joints.update(self.initial_state["joint_angles"].keys())
            
            found_joints = []
            missing_joints = []
            
            for joint_name in all_joints:
                try:
                    motor = robot.getDevice(joint_name)
                    sensor = motor.getPositionSensor()
                    sensor.enable(self.timestep)
                    
                    self.motors[joint_name] = motor
                    self.sensors[joint_name] = sensor
                    found_joints.append(joint_name)
                except:
                    if joint_name in self.control_joints:
                        missing_joints.append(joint_name)
            
            log_info("ArmTrainingEnvironment", f"Found {len(found_joints)} joints: {found_joints}")
            if missing_joints:
                log_warning("ArmTrainingEnvironment", f"Missing {len(missing_joints)} joints: {missing_joints}")
            
            log_info("ArmTrainingEnvironment", f"Observation dimension: {len(self.control_joints)}")
            log_info("ArmTrainingEnvironment", f"Action dimension: {len(self.control_joints)}")
            
            # Track if push force has been applied
            self.push_force_applied = False
            self.push_step_counter = 0
            
            log_success("ArmTrainingEnvironment", "Training environment initialized")
    
    def get_observation(self):
        """Get current observation: delta between current and goal angles."""
        log_debug("ArmTrainingEnvironment", "get_observation called")
        
        obs = []
        
        for joint_name in self.control_joints:
            if joint_name in self.sensors:
                current_angle = self.sensors[joint_name].getValue()
                goal_angle = self.goal_angles.get(joint_name, 0.0)
                delta = current_angle - goal_angle
                obs.append(delta)
                log_debug("ArmTrainingEnvironment", f"Joint {joint_name}: current={current_angle:.3f}, goal={goal_angle:.3f}, delta={delta:.3f}")
            else:
                obs.append(0.0)
                log_debug("ArmTrainingEnvironment", f"Joint {joint_name}: sensor not found, using delta=0.0")
        
        obs_array = np.array(obs, dtype=np.float32)
        log_debug("ArmTrainingEnvironment", f"Observation: {obs_array}")
        
        return obs_array
    
    def apply_push_force(self):
        """Apply push force to the robot's head."""
        if not self.push_force_config["enabled"] or self.push_force_applied:
            log_debug("ArmTrainingEnvironment", "Push force not enabled or already applied")
            return
        
        self.push_step_counter += 1
        log_debug("ArmTrainingEnvironment", f"Push step counter: {self.push_step_counter}/{self.push_force_config['delay_steps']}")
        
        # Apply force after delay steps
        if self.push_step_counter >= self.push_force_config["delay_steps"]:
            force = self.push_force_config["force"]
            angle = self.push_force_config["angle"]
            
            # Calculate force vector based on angle
            # Webots: x is forward, y is left, z is up
            fx = force * np.cos(angle)
            fy = force * np.sin(angle)
            fz = 0.0  # No vertical force
            
            # Apply force to head (relative to robot coordinate system)
            # Using addForceWithOffset(force, offset, relative=True)
            force_vector = [fx, fy, fz]
            offset = [0.0, 0.0, 0.1]  # Offset from center to head (adjust as needed)
            
            self.robot_node.addForceWithOffset(force_vector, offset, True)
            self.push_force_applied = True
            
            log_info("ArmTrainingEnvironment", f"Applied push force: {force}N at angle: {angle:.2f} rad")
    
    def get_current_angles(self):
        """Get current joint angles."""
        log_debug("ArmTrainingEnvironment", "get_current_angles called")
        
        angles = {}
        for joint_name in self.control_joints:
            if joint_name in self.sensors:
                angles[joint_name] = self.sensors[joint_name].getValue()
        
        log_debug("ArmTrainingEnvironment", f"Current angles: {angles}")
        return angles
    
    def apply_action(self, action):
        """Apply action to motors with scaling."""
        log_debug("ArmTrainingEnvironment", f"apply_action called: action={action}")
        
        # Scale actions from [-1, 1] to joint limits
        for i, joint_name in enumerate(self.control_joints):
            if joint_name in self.motors:
                # Action is in [-1, 1], scale to joint limits
                low, high = self.joint_limits[joint_name]
                
                # Map from [-1, 1] to [low, high]
                scaled_action = low + (action[i] + 1) * (high - low) / 2
                
                self.motors[joint_name].setPosition(scaled_action)
                log_debug("ArmTrainingEnvironment", f"Joint {joint_name}: action={action[i]:.3f} -> position={scaled_action:.3f}")
    
    def step(self):
        """Step the simulation."""
        log_debug("ArmTrainingEnvironment", "Stepping simulation")
        self.robot.step(self.timestep)
    
    def reset(self):
        """Reset the environment to initial state with proper stabilization."""
        log_info("ArmTrainingEnvironment", "Resetting environment")
        
        # Reset push force tracking
        self.push_force_applied = False
        self.push_step_counter = 0
        
        # Reset robot translation and rotation
        translation = self.initial_state.get("translation", [0.0, 0.0, 0.292665])
        rotation = self.initial_state.get("rotation", [0.0, 0.0, 1.0, 0.0])
        
        translation_field = self.robot_node.getField("translation")
        rotation_field = self.robot_node.getField("rotation")
        
        translation_field.setSFVec3f(translation)
        rotation_field.setSFRotation(rotation)
        
        log_debug("ArmTrainingEnvironment", f"Reset translation: {translation}")
        log_debug("ArmTrainingEnvironment", f"Reset rotation: {rotation}")
        
        # Reset joint positions from initial_state
        if "joint_angles" in self.initial_state:
            for joint_name, angle in self.initial_state["joint_angles"].items():
                if joint_name in self.motors:
                    self.motors[joint_name].setPosition(angle)
                    log_debug("ArmTrainingEnvironment", f"Set joint {joint_name} to initial angle: {angle}")
        
        # Step simulation multiple times to allow robot to stabilize
        stabilization_steps = 10
        for i in range(stabilization_steps):
            self.robot.step(self.timestep)
        
        log_debug("ArmTrainingEnvironment", f"Completed {stabilization_steps} stabilization steps")
        
        # Check if joints have reached initial positions
        max_attempts = 100
        for attempt in range(max_attempts):
            all_at_initial = True
            
            if "joint_angles" in self.initial_state:
                for joint_name, target_angle in self.initial_state["joint_angles"].items():
                    if joint_name in self.sensors:
                        current_angle = self.sensors[joint_name].getValue()
                        if abs(current_angle - target_angle) > 0.01:  # Tolerance
                            all_at_initial = False
                            break
            
            if all_at_initial:
                log_debug("ArmTrainingEnvironment", f"All joints reached initial positions at attempt {attempt}")
                break
            
            self.robot.step(self.timestep)
        
        if not all_at_initial:
            log_warning("ArmTrainingEnvironment", f"Joints did not reach initial positions after {max_attempts} attempts")
        
        self.apply_push_force()
        
        obs = self.get_observation()
        log_info("ArmTrainingEnvironment", "Environment reset complete")
        
        return obs
    
    def compute_reward(self, obs, action, next_obs, timestep):
        """Simple reward function for joint angle control."""
        log_debug("ArmTrainingEnvironment", f"compute_reward called: timestep={timestep}")
        
        # Calculate total joint angle error
        joint_error = np.sum(np.abs(next_obs))
        
        # Reward is negative of error (smaller error = higher reward)
        reward = self.angle_error_weight * joint_error
        
        # Check if all joints are within tolerance
        in_tolerance = all(abs(delta) < self.angle_tolerance for delta in next_obs)
        
        # Large bonus for success
        if in_tolerance:
            reward += self.success_reward
        
        # Done if successful
        done = in_tolerance
        
        # Also done if max steps reached
        max_steps = CONFIG["training"]["max_steps"]
        if timestep >= max_steps:
            done = True
        
        log_debug("ArmTrainingEnvironment", f"Reward: error={joint_error:.3f}, total={reward:.3f}, done={done}, in_tolerance={in_tolerance}")
        
        return reward, done
    
    def is_success(self, obs):
        """Check if current state is a success."""
        success = all(abs(delta) < self.angle_tolerance for delta in obs)
        log_debug("ArmTrainingEnvironment", f"is_success: {success}")
        return success


# ============================================================================
# TRAINING MODE
# ============================================================================

def train_mode():
    """Run DDPG training mode."""
    log_section("DDPGController", "STARTING DDPG TRAINING MODE")
    
    with LogFunction("DDPGController", "train_mode"):
        
        # Check for multi-agent mode
        is_multi_agent = os.environ.get('RL_MULTI_AGENT', '').lower() == 'true'
        checkpoint_path = os.environ.get('RL_CHECKPOINT_PATH', None)
        results_file = os.environ.get('RL_RESULTS_FILE', None)
        episodes_per_run = int(os.environ.get('RL_EPISODES_PER_RUN', '1'))
        agent_id = os.environ.get('RL_AGENT_ID', None)
        stage_id = int(os.environ.get('RL_STAGE_ID', '0'))
        global_episode = int(os.environ.get('RL_GLOBAL_EPISODE', '0'))
        
        log_info("DDPGController", f"Multi-agent mode: {is_multi_agent}")
        if is_multi_agent:
            log_data("DDPGController", "Multi-agent info", {
                'agent_id': agent_id,
                'stage_id': stage_id,
                'global_episode': global_episode,
                'episodes_per_run': episodes_per_run,
                'checkpoint_path': checkpoint_path,
                'results_file': results_file
            })
        
        # Override config with stage environment if provided
        stage_env_str = os.environ.get('RL_STAGE_ENV', None)
        if stage_env_str:
            try:
                stage_env = json.loads(stage_env_str)
                if 'goal_angles' in stage_env:
                    CONFIG['goal_angles'] = stage_env['goal_angles']
                    log_info("DDPGController", f"Overridden goal angles from stage environment")
                    log_data("DDPGController", "New goal angles", CONFIG['goal_angles'])
            except Exception as e:
                log_exception("DDPGController", e, "Failed to parse stage environment")
        
        # Override hyperparameters with stage hyperparameters if provided
        for key in ['actor_lr', 'critic_lr', 'gamma', 'tau']:
            env_key = f'RL_HP_{key.upper()}'
            if env_key in os.environ:
                try:
                    CONFIG['training']['ddpg'][key] = float(os.environ[env_key])
                    log_info("DDPGController", f"Overridden {key} from stage: {CONFIG['training']['ddpg'][key]}")
                except Exception as e:
                    log_exception("DDPGController", e, f"Failed to parse {env_key}")
        
        log_info("DDPGController", f"Control joints: {CONFIG['control_joints']}")
        log_info("DDPGController", f"Goal angles: {CONFIG['goal_angles']}")
        
        if is_multi_agent:
            max_episodes = episodes_per_run
            log_info("DDPGController", f"Multi-agent: training {max_episodes} episodes")
        else:
            max_episodes = CONFIG["training"]["max_episodes"]
            log_info("DDPGController", f"Single-agent: training {max_episodes} episodes")
        
        robot = Supervisor()
        
        # Create environment
        env = ArmTrainingEnvironment(robot)
        
        # Create or load agent
        if is_multi_agent and checkpoint_path and os.path.exists(checkpoint_path):
            # Try to load agent from checkpoint
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                # Check if this is a minimal "new agent" checkpoint
                if checkpoint.get('is_new_agent', False) or 'actor_state_dict' not in checkpoint:
                    # This is a new agent marker - create new agent instead
                    log_info("DDPGController", f"New agent detected from checkpoint marker, creating fresh agent (agent_id: {agent_id})")
                    agent = DDPGAgent(obs_dim=len(CONFIG["control_joints"]),
                                     act_dim=len(CONFIG["control_joints"]),
                                     agent_id=agent_id)
                else:
                    # Load existing agent from checkpoint
                    agent = DDPGAgent.load(checkpoint_path, agent_id=agent_id)
                    log_success("DDPGController", f"Loaded agent from checkpoint: {checkpoint_path}")
            except Exception as e:
                log_exception("DDPGController", e, f"Error loading checkpoint {checkpoint_path}")
                log_info("DDPGController", "Creating new agent instead...")
                # Create new agent if loading fails
                agent = DDPGAgent(obs_dim=len(CONFIG["control_joints"]),
                                 act_dim=len(CONFIG["control_joints"]),
                                 agent_id=agent_id)
        else:
            # Create new agent (single-agent mode or no checkpoint)
            agent = DDPGAgent(obs_dim=len(CONFIG["control_joints"]),
                             act_dim=len(CONFIG["control_joints"]),
                             agent_id=agent_id)
        
        # Training statistics
        episode_rewards = []
        episode_steps = []
        success_history = []
        episode_successes = []
        episode_errors = []
        
        start_time = time.time()
        best_reward = -float('inf')
        
        if is_multi_agent:
            max_episodes = episodes_per_run
            save_every = max_episodes + 1  # Don't save mid-run in multi-agent mode
            early_stopping = False  # Disable early stopping in multi-agent mode
            window_size = 50
            success_threshold = 0.0
            min_episodes = 0
            local_episode_id = 0  # Track local episode within this run
        else:
            max_episodes = CONFIG["training"]["max_episodes"]
            save_every = CONFIG["checkpoints"]["save_every"]
            early_stopping = CONFIG["early_stopping"]["enabled"]
            window_size = CONFIG["early_stopping"]["window_size"]
            success_threshold = CONFIG["early_stopping"]["success_threshold"]
            min_episodes = CONFIG["early_stopping"]["min_episodes"]
        
        # Initialize HDF5 stats logger for both single-agent and multi-agent modes
        stats_logger = create_stats_logger(CONTROLLER_DIR, "ddpg")

        # Log stage information
        stage_info = StageInfo(
            stage_id=stage_id if is_multi_agent else 0,
            stage_name=f"ddpg_training_stage_{stage_id if is_multi_agent else 0}",
            start_episode_global=global_episode if is_multi_agent else 0,
            hyperparameters=CONFIG["training"]["ddpg"],
            metrics={}
        )
        stats_logger.log_stage(stage_info)

        # Log agent information
        agent_info = AgentInfo(
            agent_id=agent.agent_id,
            stage_id=stage_id if is_multi_agent else 0,
            agent_type="ddpg",
            parameters_hash="",  # Could compute hash of config
            parent_id=agent.parent_id,
            lineage_depth=agent.lineage_depth
        )
        stats_logger.log_agent(agent_info)

        log_info("DDPGController", "Starting DDPG training...")
        log_info("DDPGController", f"Early stopping: {early_stopping}, window: {window_size}, threshold: {success_threshold}")

        for episode in range(1, max_episodes + 1):
            log_info("DDPGController", f"Starting episode {episode}/{max_episodes}")
            
            obs = env.reset()
            total_reward = 0.0
            steps = 0
            done = False
            
            # Track per-step data for detailed stats
            step_rewards = []
            step_errors = []
            step_actions = []
            
            while not done and steps < CONFIG["training"]["max_steps"]:
                # Get action with exploration noise
                action = agent.get_action(obs, add_noise=True)
                
                # Apply action and step
                env.apply_action(action)
                env.step()
                steps += 1
                
                # Get new observation
                next_obs = env.get_observation()
                
                # Compute reward
                reward, done = env.compute_reward(obs, action, next_obs, steps)
                
                # Store experience in replay buffer
                agent.replay_buffer.push(obs.copy(), action.copy(), reward, next_obs.copy(), done)
                
                # Update networks (multiple times if buffer is large enough)
                if len(agent.replay_buffer) >= agent.batch_size:
                    # Update more frequently in early training
                    update_times = 2 if episode < 100 else 1
                    for update_idx in range(update_times):
                        critic_loss, actor_loss = agent.update()
                        log_debug("DDPGController", f"Episode {episode}, step {steps}, update {update_idx}: critic_loss={critic_loss:.4f}, actor_loss={actor_loss:.4f}")
                
                # Track stats
                total_reward += reward
                step_rewards.append(float(reward))
                step_error = np.mean(np.abs(next_obs))
                step_errors.append(float(step_error))
                step_actions.append([float(a) for a in action])
                
                obs = next_obs
            
            # Decay exploration noise
            agent.decay_noise()
            
            # Check if episode was successful
            success = 1 if env.is_success(obs) else 0
            episode_successes.append(success)
            avg_error = np.mean(step_errors) if step_errors else 0.0
            
            # Update statistics
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            episode_errors.append(avg_error)
            
            log_info("DDPGController", f"Episode {episode}: reward={total_reward:.2f}, steps={steps}, success={success}, avg_error={avg_error:.4f}")
            
            # In multi-agent mode, save results after each episode
            if is_multi_agent and results_file:
                local_episode_id = episode - 1
                results = {
                    'agent_id': agent_id,
                    'stage_id': stage_id,
                    'global_episode_id': global_episode,
                    'local_episode_id': local_episode_id,
                    'total_reward': float(total_reward),
                    'steps': steps,
                    'success': bool(success),
                    'termination_reason': 'normal' if not done else 'time_limit',
                    'average_error': float(avg_error),
                    'start_idx': 0,  # Will be updated by stats logger
                    'end_idx': steps
                }
                
                try:
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    log_success("DDPGController", f"Saved results to {results_file}")
                except Exception as e:
                    log_exception("DDPGController", e, f"Failed to save results to {results_file}")
                
                # Save updated checkpoint
                if checkpoint_path:
                    updated_checkpoint = checkpoint_path.replace('.pt', '_updated.pt')
                    try:
                        agent.save(updated_checkpoint)
                        log_success("DDPGController", f"Saved updated checkpoint: {updated_checkpoint}")
                    except Exception as e:
                        log_exception("DDPGController", e, f"Failed to save updated checkpoint")

            # Log episode to HDF5 stats
            episode_info = EpisodeInfo(
                global_episode_id=episode - 1,  # 0-based indexing
                stage_id=stage_id if is_multi_agent else 0,
                local_episode_id=episode - 1,
                total_steps=steps,
                total_reward=float(total_reward),
                success=bool(success),
                termination_reason='normal' if not done else 'time_limit',
                agent_id=agent.agent_id,
                start_idx=0,  # Will be updated by stats logger
                end_idx=steps
            )
            stats_logger.log_episode(episode_info)

            # Episode stats are now saved to HDF5 only

            # Calculate success rate
            if episode >= window_size:
                recent_successes = episode_successes[-window_size:]
                success_rate = np.mean(recent_successes)
                success_history.append(success_rate)
            else:
                success_rate = np.mean(episode_successes) if episode_successes else 0.0
                success_history.append(success_rate)
            
            # Print progress every episode in early training, then every 10
            if episode <= 20 or episode % 10 == 0:
                elapsed = time.time() - start_time
                
                # Get current angles for debugging
                current_angles = env.get_current_angles()
                
                log_info("DDPGController", 
                        f"Ep {episode:4d} | "
                        f"Reward: {total_reward:7.2f} | "
                        f"Steps: {steps:3d} | "
                        f"Success: {success} | "
                        f"Error: {avg_error:.4f} | "
                        f"Rate: {success_rate:.1%} | "
                        f"Noise: {agent.noise_std:.3f} | "
                        f"Angles: {[f'{v:.2f}' for v in current_angles.values()]}")
            
            # Save best model (inside run directory)
            if CONFIG["checkpoints"]["save_best"] and total_reward > best_reward:
                best_reward = total_reward
                agent.save(BEST_MODEL_PATH)
                log_success("DDPGController", f"New best model saved: {BEST_MODEL_PATH} (reward: {best_reward:.2f})")
            
            # Save checkpoint and generate plots periodically (matches PPO)
            if episode % save_every == 0:
                checkpoint_path_single = os.path.join(CHECKPOINT_DIR, f"ddpg_checkpoint_ep{episode}.pt")
                agent.save(checkpoint_path_single)
        
                # Generate plots
                generate_training_plots(episode_rewards, episode_steps, 
                                    success_history, episode_errors, PLOTS_DIR)
                
                log_info("DDPGController", f"Checkpoint saved: {checkpoint_path_single}")
            
            # Early stopping
            if early_stopping and episode >= min_episodes and episode >= window_size:
                if success_rate >= success_threshold:
                    log_success("DDPGController", f"Early stopping at episode {episode} with {success_rate:.1%} success rate!")
                    break
        
        # Save final model (inside runs directory, not overwritten by different run_label)
        os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
        agent.save(FINAL_MODEL_PATH)
        log_success("DDPGController", f"Final model saved to: {FINAL_MODEL_PATH}")

        # Generate final plots (matches PPO)
        if not is_multi_agent:
            generate_training_plots(episode_rewards, episode_steps, 
                                success_history, episode_errors, FINAL_PLOTS_DIR)
        
        total_time = time.time() - start_time

        # Log final stage metrics
        if episode_rewards:
            final_success_rate = success_history[-1] if success_history else 0.0
            stats_logger.log_stage_summary(
                stage_id=stage_id if is_multi_agent else 0,
                mean_reward=float(np.mean(episode_rewards)),
                success_rate=final_success_rate,
                mean_episode_length=float(np.mean(episode_steps)),
                window_size=window_size
            )

        # Update stage status to completed
        stats_logger.update_stage(
            stage_id=stage_id if is_multi_agent else 0,
            end_episode_global=(global_episode + len(episode_rewards) - 1) if is_multi_agent else len(episode_rewards) - 1,
            status="completed"
        )

        log_section("DDPGController", "TRAINING COMPLETE")
        log_info("DDPGController", f"Total episodes: {len(episode_rewards)}")
        log_info("DDPGController", f"Total steps: {sum(episode_steps)}")
        log_info("DDPGController", f"Total time: {total_time/60:.1f} minutes")
        log_info("DDPGController", f"Average reward: {np.mean(episode_rewards):.3f}")
        log_info("DDPGController", f"Best reward: {best_reward:.3f}")
        log_info("DDPGController", f"Final success rate: {success_history[-1] if success_history else 0.0:.1%}")
        log_info("DDPGController", f"Final model saved to: {FINAL_MODEL_PATH}")
        log_info("DDPGController", f"Best model saved to: {BEST_MODEL_PATH}")
        log_info("DDPGController", f"Stats saved to: {STATS_DIR}")
        
        # In multi-agent mode, exit immediately after saving results
        if is_multi_agent:
            log_success("DDPGController", f"Multi-agent training complete: {len(episode_rewards)} episodes")
            robot.simulationQuit(0)
            return
        
        # Wait a bit before exiting (single-agent mode only)
        for i in range(50):
            robot.step(CONFIG["training"]["timestep"])
        
        robot.simulationSetMode(0) # Pause simulation at the end
        
        log_success("DDPGController", "DDPG training completed successfully")


# ============================================================================
# TEST MODE
# ============================================================================

def test_mode():
    """Run test mode with a trained DDPG model."""
    log_section("DDPGController", "STARTING DDPG TEST MODE")
    
    with LogFunction("DDPGController", "test_mode"):
        
        robot = Supervisor()
        model_path = CONFIG["model_path"]
        
        log_info("DDPGController", f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            log_error("DDPGController", f"Model file not found: {model_path}")
            log_error("DDPGController", "Please train a model first or update model_path in config.json")
            return
        
        env = ArmTrainingEnvironment(robot)
        agent = DDPGAgent.load(model_path)
        
        # Set to evaluation mode
        agent.actor.eval()
        agent.actor_target.eval()
        
        log_info("DDPGController", "Starting test simulation...")
        log_info("DDPGController", "Press Ctrl+C in Webots to stop")
        
        episode_count = 0
        try:
            while True:
                episode_count += 1
                log_info("DDPGController", f"Test Episode {episode_count}")
                
                obs = env.reset()
                total_reward = 0.0
                steps = 0
                done = False
                
                while not done and steps < 100:
                    # Get deterministic action (no noise)
                    action = agent.get_action(obs, add_noise=False)
                    
                    # Apply action
                    env.apply_action(action)
                    env.step()
                    steps += 1
                    
                    # Get new observation
                    next_obs = env.get_observation()
                    
                    # Compute reward (for monitoring)
                    reward, done = env.compute_reward(obs, action, next_obs, steps)
                    total_reward += reward
                    
                    # Print progress
                    if steps % 20 == 0 or done:
                        current_angles = env.get_current_angles()
                        error = np.sum(np.abs(next_obs))
                        log_info("DDPGController", 
                                f"  Step {steps:3d}: Reward: {total_reward:7.2f} | "
                                f"Error: {error:.3f} | "
                                f"Angles: {current_angles}")
                    
                    obs = next_obs
                
                # Episode summary
                final_angles = env.get_current_angles()
                final_error = np.sum(np.abs(obs))
                
                log_success("DDPGController", f"Episode {episode_count} complete:")
                log_info("DDPGController", f"   Steps: {steps}")
                log_info("DDPGController", f"   Total reward: {total_reward:.2f}")
                log_info("DDPGController", f"   Final error: {final_error:.3f}")
                log_info("DDPGController", f"   Final angles: {final_angles}")
                log_info("DDPGController", f"   Goal angles: {CONFIG['goal_angles']}")
                
                # Wait between episodes
                for wait_step in range(30):
                    robot.step(CONFIG["training"]["timestep"])
                    
        except KeyboardInterrupt:
            log_warning("DDPGController", "Test interrupted by user")
        finally:
            log_info("DDPGController", "Test simulation ended.")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function - read mode from config.json."""
    log_section("DDPGController", "STARTING DDPG CONTROLLER")
    
    with LogFunction("DDPGController", "main"):
        mode = CONFIG["mode"]
        
        log_info("DDPGController", f"DDPG Controller - Mode: {mode}")
        
        if mode == "train":
            train_mode()
        elif mode == "test":
            model_path = CONFIG["model_path"]
            if not os.path.exists(model_path):
                log_error("DDPGController", f"Model file not found: {model_path}")
                log_error("DDPGController", "Please train a model first or update model_path in config.json")
                return
            test_mode()
        else:
            log_error("DDPGController", f"Unknown mode: {mode}")
            log_error("DDPGController", "Please set mode to 'train' or 'test' in config.json")


if __name__ == "__main__":
    main()