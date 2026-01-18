# controllers/op3_ppo/op3_ppo.py
"""
OP3 PPO Controller - CORRECTED VERSION with proper PPO implementation
"""

from controller import Supervisor
import numpy as np
import os
import sys
import json
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# ============================================================================
# CONSTANTS AND PATHS
# ============================================================================

# Directory constants
CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(CONTROLLER_DIR, "checkpoints")
STATS_DIR = os.path.join(CONTROLLER_DIR, "training_stats")
PROJECT_ROOT = os.path.abspath(os.path.join(CONTROLLER_DIR, '..', '..'))

# Add project root to path
sys.path.insert(0, PROJECT_ROOT)


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
            "model_path": "controllers/op3_ppo/checkpoints/ppo_final.pt",
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
                "max_episodes": 2000,
                "max_steps": 200,
                "timestep": 8,
                "ppo": {
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "clip_epsilon": 0.2,
                    "num_epochs": 10,
                    "batch_size": 64,
                    "entropy_coeff": 0.01,
                    "value_coeff": 0.5
                },
                "reward": {
                    "angle_tolerance": 0.05,
                    "time_penalty": -0.001,
                    "success_reward": 10.0,
                    "angle_error_weight": -1.0,
                    "progress_bonus": 0.1,
                    "stability_bonus": 0.1
                },
                "joint_limits": {
                    "ShoulderR": [-1.57, 1.57]
                }
            },
            "early_stopping": {
                "enabled": True,
                "window_size": 100,
                "success_threshold": 1.0,
                "min_episodes": 200
            },
            "checkpoints": {
                "save_every": 100,
                "save_best": True
            }
        }
    else:
        return {
            "mode": "test",
            "model_path": "controllers/op3_ppo/checkpoints/ppo_final.pt",
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
                "render": True,
                "sleep_time": 0.01
            },
            "reward": {
                "angle_tolerance": 0.05,
                "time_penalty": -0.01,
                "success_reward": 1.0,
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
    # Determine config file based on mode
    is_train = os.environ.get('RL_TRAIN', '').lower() == 'true'
    
    if is_train:
        config_path = os.path.join(CONTROLLER_DIR, "config_train.json")
        template_path = os.path.join(CONTROLLER_DIR, "config_train.json.template")
        mode_str = "train"
        print(f"🤖 PPO Controller - Running in TRAINING mode")
    else:
        config_path = os.path.join(CONTROLLER_DIR, "config_test.json")
        template_path = os.path.join(CONTROLLER_DIR, "config_test.json.template")
        mode_str = "test"
        print(f"🤖 PPO Controller - Running in TESTING mode")
    
    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"⚠️  Config file not found: {config_path}")
        
        # Try fallback to config.json
        fallback_path = os.path.join(CONTROLLER_DIR, "config.json")
        if os.path.exists(fallback_path):
            config_path = fallback_path
            print(f"ℹ️  Using fallback config: {config_path}")
        else:
            # Check if template exists
            if template_path and os.path.exists(template_path):
                print(f"📄 Creating config from template: {template_path}")
                with open(template_path, 'r') as f:
                    config = json.load(f)
            else:
                # Create default config
                print(f"🔧 Creating default configuration")
                config = _create_default_config(mode_str)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Save the config file
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Created config file: {config_path}")
    
    # Load the config file
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✅ Loaded config from: {config_path}")
    
    # Update model path to be absolute
    if not os.path.isabs(config.get("model_path", "")):
        config["model_path"] = os.path.join(PROJECT_ROOT, config["model_path"])
    
    # Log control joints
    if "control_joints" in config:
        print(f"🎯 Control joints: {config['control_joints']}")
    if "goal_angles" in config:
        print(f"📍 Goal angles: {config['goal_angles']}")
    
    return config


# Load configuration
CONFIG = load_config()


class SimplePPOActor(nn.Module):
    """Actor network with proper action scaling via tanh"""
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, act_dim),
        )
        # Learnable log std
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))
    
    def forward(self, x):
        mean = self.net(x)
        return mean, self.log_std.expand_as(mean)


class SimplePPOCritic(nn.Module):
    """Critic network"""
    def __init__(self, obs_dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, x):
        return self.net(x)


class SimplePPOAgent:
    """CORRECT PPO agent implementation"""
    
    def __init__(self, obs_dim, act_dim, config, agent_id=None, parent_id=None):
        """Initialize PPO agent.
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension
            config: Configuration dictionary
            agent_id: Unique agent identifier (auto-generated if None)
            parent_id: Parent agent ID for lineage tracking
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config
        self.agent_id = agent_id or f"ppo_{id(self)}"
        self.parent_id = parent_id
        self.train_config = config["training"]["ppo"]
        
        # Networks
        self.actor = SimplePPOActor(obs_dim, act_dim, hidden_size=64)
        self.critic = SimplePPOCritic(obs_dim, hidden_size=64)
        
        # Optimizers
        lr = self.train_config["learning_rate"]
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # PPO parameters
        self.gamma = self.train_config["gamma"]
        self.clip_epsilon = self.train_config["clip_epsilon"]
        self.num_epochs = self.train_config["num_epochs"]
        self.batch_size = self.train_config["batch_size"]
        self.value_coeff = self.train_config["value_coeff"]
        self.entropy_coeff = self.train_config["entropy_coeff"]
        
        # Device
        self.device = torch.device("cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        # Experience buffer - now stores log_probs too!
        self.buffer = []
        
        # Lineage tracking
        self.creation_timestamp = ""
        self.lineage_depth = 0 if parent_id is None else 1
    
    def get_action(self, obs, deterministic=False):
        """Get action from actor network with proper distribution."""
        obs_t = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.actor(obs_t)
            std = torch.exp(log_std)
            
            if deterministic:
                # For testing: just use the mean (with tanh for bounds)
                action = torch.tanh(mean)
            else:
                # For training: sample from distribution
                dist = torch.distributions.Normal(mean, std)
                action_raw = dist.sample()
                
                # Apply tanh for action bounds [-1, 1]
                action = torch.tanh(action_raw)
                
                # Compute log prob for the raw action (before tanh)
                # This is the correct way to handle tanh in PPO
                log_prob_raw = dist.log_prob(action_raw).sum(dim=-1)
                
                # Adjust for tanh transform
                log_prob = log_prob_raw - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
                log_prob = log_prob.unsqueeze(-1)
        
        action_np = action.squeeze(0).cpu().numpy()
        
        if deterministic:
            return action_np, None
        else:
            return action_np, log_prob.item()
    
    def store_transition(self, obs, action, reward, next_obs, done, log_prob=None):
        """Store a transition in the buffer WITH log probability."""
        self.buffer.append((obs, action, reward, next_obs, done, log_prob))
    
    def clear_buffer(self):
        """Clear the replay buffer."""
        self.buffer = []
    
    def compute_gae(self, values, rewards, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gamma * (1 - dones[t]) * last_advantage
        
        return advantages
    
    def update(self):
        """CORRECT PPO update implementation."""
        if len(self.buffer) < self.batch_size:
            return
        
        # Unpack buffer
        obs, actions, rewards, next_obs, dones, old_log_probs = zip(*self.buffer)
        
        obs = np.array(obs)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_obs = np.array(next_obs)
        dones = np.array(dones)
        old_log_probs = np.array(old_log_probs).reshape(-1, 1)
        
        # Convert to tensors
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        
        # Compute values and advantages
        with torch.no_grad():
            values = self.critic(obs_t).squeeze().cpu().numpy()
            next_obs_t = torch.FloatTensor(next_obs[-1:]).to(self.device)
            next_value = self.critic(next_obs_t).squeeze().cpu().numpy()
        
        # Compute advantages
        advantages = self.compute_gae(values, rewards, dones, next_value)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns_t = torch.FloatTensor(returns).to(self.device).unsqueeze(1)
        advantages_t = torch.FloatTensor(advantages).to(self.device).unsqueeze(1)
        
        # Multiple epochs of updates
        for epoch in range(self.num_epochs):
            # Shuffle indices for minibatch updates
            indices = torch.randperm(len(obs_t))
            
            for start in range(0, len(obs_t), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_obs = obs_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                
                # Get current policy output
                mean, log_std = self.actor(batch_obs)
                std = torch.exp(log_std)
                
                # Create distribution and compute log probs
                dist = torch.distributions.Normal(mean, std)
                
                # Compute log probability for the action (before tanh)
                action_raw = torch.atanh(torch.clamp(batch_actions, -0.999, 0.999))
                log_prob_raw = dist.log_prob(action_raw).sum(dim=-1, keepdim=True)
                
                # Adjust for tanh transform
                log_prob = log_prob_raw - torch.log(1 - batch_actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
                
                # PPO ratio
                ratio = torch.exp(log_prob - batch_old_log_probs)
                
                # PPO losses
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
                
                # Backpropagation
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        self.clear_buffer()
    
    def save(self, filepath, agent_id=None):
        """Save the agent's networks to a file.
        
        Args:
            filepath: Path to save checkpoint
            agent_id: Optional agent ID to include in checkpoint
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'config': self.config,
            'agent_id': agent_id or self.agent_id,
            'parent_id': self.parent_id,
            'lineage_depth': self.lineage_depth,
            'creation_timestamp': self.creation_timestamp,
        }
        torch.save(checkpoint, filepath)
        print(f"✅ Model saved to {filepath} (agent_id: {checkpoint['agent_id']})")
    
    @classmethod
    def load(cls, filepath, agent_id=None):
        """Load an agent from a saved checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            agent_id: Optional agent ID override
            
        Returns:
            SimplePPOAgent instance
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        agent = cls(
            checkpoint['obs_dim'], 
            checkpoint['act_dim'], 
            checkpoint.get('config', CONFIG),
            agent_id=agent_id or checkpoint.get('agent_id'),
            parent_id=checkpoint.get('parent_id')
        )
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.lineage_depth = checkpoint.get('lineage_depth', 0)
        agent.creation_timestamp = checkpoint.get('creation_timestamp', '')
        print(f"✅ Model loaded from {filepath} (agent_id: {agent.agent_id})")
        return agent


class ArmTrainingEnvironment:
    """Environment with improved reward function"""
    
    def __init__(self, robot, config):
        self.robot = robot
        self.config = config
        
        # Get robot node for applying forces
        self.robot_node = robot.getSelf()
        
        # Joints to control
        self.control_joints = config["control_joints"]
        self.goal_angles = config["goal_angles"]
        self.joint_limits = config["training"]["joint_limits"]
        self.timestep = config["training"]["timestep"]
        
        # Push force configuration
        self.push_force_config = config.get("push_force", {
            "enabled": False,
            "force": 5.0,
            "angle": 0.0,
            "delay_steps": 20
        })
        
        # Initial state configuration
        self.initial_state = config.get("initial_state", {
            "translation": [0.0, 0.0, 0.292665],
            "rotation": [0.0, 0.0, 1.0, 0.0],
            "joint_angles": {}
        })
        
        # Reward parameters
        reward_config = config["training"]["reward"]
        self.angle_tolerance = reward_config["angle_tolerance"]
        self.time_penalty = reward_config["time_penalty"]
        self.success_reward = reward_config["success_reward"]
        self.angle_error_weight = reward_config["angle_error_weight"]
        self.progress_bonus = reward_config.get("progress_bonus", 0.0)
        self.stability_bonus = reward_config.get("stability_bonus", 0.0)
        
        # Initialize motors
        self.motors = {}
        self.sensors = {}
        
        # Initialize all joints from initial_state
        all_joints = set(self.control_joints)
        if "joint_angles" in self.initial_state:
            all_joints.update(self.initial_state["joint_angles"].keys())
        
        for joint_name in all_joints:
            try:
                motor = robot.getDevice(joint_name)
                sensor = motor.getPositionSensor()
                sensor.enable(self.timestep)
                
                self.motors[joint_name] = motor
                self.sensors[joint_name] = sensor
                print(f"✅ Found joint: {joint_name}")
            except:
                if joint_name in self.control_joints:
                    print(f"❌ Could not find joint: {joint_name}")
        
        print(f"Observation dimension: {len(self.control_joints)}")
        print(f"Action dimension: {len(self.control_joints)}")
        print(f"Goal angles: {self.goal_angles}")
        
        # Track if push force has been applied
        self.push_force_applied = False
        self.push_step_counter = 0
    
    def get_observation(self):
        """Get current observation: delta between current and goal angles."""
        obs = []
        
        for joint_name in self.control_joints:
            if joint_name in self.sensors:
                current_angle = self.sensors[joint_name].getValue()
                goal_angle = self.goal_angles.get(joint_name, 0.0)
                delta = current_angle - goal_angle
                obs.append(delta)
            else:
                obs.append(0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def get_current_angles(self):
        """Get current joint angles."""
        angles = {}
        for joint_name in self.control_joints:
            if joint_name in self.sensors:
                angles[joint_name] = self.sensors[joint_name].getValue()
        return angles
    
    def apply_action(self, action):
        """Apply action to motors with scaling."""
        # Scale actions from [-1, 1] to joint limits
        for i, joint_name in enumerate(self.control_joints):
            if joint_name in self.motors:
                # Action is in [-1, 1], scale to joint limits
                low, high = self.joint_limits[joint_name]
                
                # Map from [-1, 1] to [low, high]
                scaled_action = low + (action[i] + 1) * (high - low) / 2
                
                self.motors[joint_name].setPosition(scaled_action)
    
    def apply_push_force(self):
        """Apply push force to the robot's head."""
        if not self.push_force_config["enabled"] or self.push_force_applied:
            return
        
        self.push_step_counter += 1
        
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
            
            print(f"💨 Applied push force: {force}N at angle: {angle:.2f} rad")
    
    def step(self):
        """Step the simulation."""
        self.robot.step(self.timestep)
    
    def reset(self):
        """Reset the environment to initial state with proper stabilization."""
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
        
        # Reset joint positions from initial_state
        if "joint_angles" in self.initial_state:
            for joint_name, angle in self.initial_state["joint_angles"].items():
                if joint_name in self.motors:
                    self.motors[joint_name].setPosition(angle)
        
        # Step simulation multiple times to allow robot to stabilize
        stabilization_steps = 10
        for _ in range(stabilization_steps):
            self.robot.step(self.timestep)
        
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
                break
            
            self.robot.step(self.timestep)
        
        self.apply_push_force()
        
        return self.get_observation()
    
    def compute_reward(self, obs, action, next_obs, timestep):
        """SIMPLE reward function: negative joint error."""
        # Calculate total joint angle error (sum of absolute deltas)
        joint_error = np.sum(np.abs(next_obs))
        
        # Reward is negative of error (smaller error = higher reward)
        reward = -joint_error
        
        # Add small time penalty to encourage efficiency
        reward += self.time_penalty
        
        # Check if all joints are within tolerance
        in_tolerance = all(abs(delta) < self.angle_tolerance for delta in next_obs)
        
        # Small bonus for success
        if in_tolerance:
            reward += self.success_reward
        
        # Done if successful (termination condition)
        done = in_tolerance
        
        # Also done if max steps reached
        max_steps = self.config["training"]["max_steps"]
        if timestep >= max_steps:
            done = True
        
        return reward, done
    
    def is_success(self, obs):
        """Check if current state is a success."""
        return all(abs(delta) < self.angle_tolerance for delta in obs)

def save_detailed_stats(episode_stats, episode_num, config):
    """Save detailed training statistics."""
    os.makedirs(STATS_DIR, exist_ok=True)
    
    stats_file = os.path.join(STATS_DIR, f"episode_{episode_num:04d}_stats.json")
    
    with open(stats_file, 'w') as f:
        json.dump(episode_stats, f, indent=2)
    
    # Also save to a cumulative file
    cumulative_file = os.path.join(STATS_DIR, "all_episodes_summary.csv")
    
    if episode_num == 1:
        with open(cumulative_file, 'w') as f:
            f.write("episode,total_reward,steps,success,average_error,max_error,min_error\n")
    
    with open(cumulative_file, 'a') as f:
        f.write(f"{episode_num},"
                f"{episode_stats['total_reward']:.4f},"
                f"{episode_stats['steps']},"
                f"{int(episode_stats['success'])},"
                f"{episode_stats['average_error']:.4f},"
                f"{episode_stats['max_error']:.4f},"
                f"{episode_stats['min_error']:.4f}\n")


def generate_training_plots(episode_rewards, episode_steps, success_history, 
                           episode_errors, save_dir, config):
    """Generate and save training plots."""
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
    success_threshold = config["early_stopping"]["success_threshold"]
    axes[1, 0].axhline(y=success_threshold, color='r', linestyle='--', alpha=0.7, 
                      label=f'{success_threshold:.0%} Target')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_title(f'Success Rate (Last {config["early_stopping"]["window_size"]} Episodes)')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Average Angle Error
    axes[1, 1].plot(episode_errors, alpha=0.7, color='orange')
    tolerance = config["training"]["reward"]["angle_tolerance"]
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
    
    print(f"📈 Training plots saved to: {plot_path}")
    
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
        'config': config,
    }
    
    summary_path = os.path.join(save_dir, f'training_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return plot_path

def train_mode(config):
    """Run training mode with proper PPO and save statistics."""
    print("\n" + "="*70)
    print("🤖 OP3 ARM CONTROL - PPO TRAINING (WITH STATS)")
    print("="*70)
    print(f"Control joints: {config['control_joints']}")
    print(f"Goal angles: {config['goal_angles']}")
    print(f"Max episodes: {config['training']['max_episodes']}")
    print(f"Max steps per episode: {config['training']['max_steps']}")
    print(f"Stats directory: {os.path.join(CONTROLLER_DIR, 'training_stats')}")
    print("="*70)
    
    robot = Supervisor()
    
    # Create environment
    env = ArmTrainingEnvironment(robot, config)
    
    # Create agent
    agent = SimplePPOAgent(obs_dim=len(config["control_joints"]), 
                          act_dim=len(config["control_joints"]), 
                          config=config)
    
    # Training statistics
    episode_rewards = []
    episode_steps = []
    success_history = []
    episode_errors = []
    episode_successes = []
    
    start_time = time.time()
    best_reward = -float('inf')
    
    max_episodes = config["training"]["max_episodes"]
    save_every = config["checkpoints"]["save_every"]
    early_stopping = config["early_stopping"]["enabled"]
    window_size = config["early_stopping"]["window_size"]
    success_threshold = config["early_stopping"]["success_threshold"]
    min_episodes = config["early_stopping"]["min_episodes"]
    
    print("\n🏁 Starting PPO training...")
    print("-"*70)
    
    for episode in range(1, max_episodes + 1):
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        
        # Per-episode detailed stats
        step_rewards = []
        step_errors = []
        step_actions = []
        
        while not done and steps < config["training"]["max_steps"]:
            # Get action with log probability
            action, log_prob = agent.get_action(obs)
            
            # Debug output for first episode
            if episode == 1 and steps < 3:
                current_angles = env.get_current_angles()
                print(f"Debug Ep1 Step {steps}: "
                      f"Action={[f'{a:.3f}' for a in action]}, "
                      f"Angles={[f'{v:.3f}' for v in current_angles.values()]}")
            
            # Apply action and step
            env.apply_action(action)
            env.step()
            steps += 1
            
            # Get new observation
            next_obs = env.get_observation()
            
            # Compute reward
            reward, done = env.compute_reward(obs, action, next_obs, steps)
            
            # Store transition WITH log probability
            agent.store_transition(obs, action, reward, next_obs, done, log_prob)
            
            # Track stats
            total_reward += reward
            step_rewards.append(float(reward))
            step_errors.append(float(np.mean(np.abs(next_obs))))
            step_actions.append([float(a) for a in action])
            
            obs = next_obs
        
        # Update agent
        agent.update()
        
        # Check if episode was successful
        success = 1 if env.is_success(obs) else 0
        episode_successes.append(success)
        avg_error = np.mean(step_errors) if step_errors else 0.0
        
        # Update statistics
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_errors.append(avg_error)
        
        # Calculate success rate
        if episode >= window_size:
            recent_successes = episode_successes[-window_size:]
            success_rate = np.mean(recent_successes)
            success_history.append(success_rate)
        else:
            success_rate = np.mean(episode_successes) if episode_successes else 0.0
            success_history.append(success_rate)
        
        # Save detailed episode stats
        episode_stats = {
            'episode': episode,
            'total_reward': float(total_reward),
            'steps': steps,
            'success': bool(success),
            'average_error': float(avg_error),
            'max_error': float(np.max(step_errors)) if step_errors else 0.0,
            'min_error': float(np.min(step_errors)) if step_errors else 0.0,
            'final_angles': {joint: float(env.sensors[joint].getValue() if joint in env.sensors else 0.0)
                           for joint in config["control_joints"]},
            'goal_angles': config["goal_angles"],
            'step_rewards': step_rewards,
            'step_errors': step_errors,
            'step_actions': step_actions,
        }
        
        save_detailed_stats(episode_stats, episode, config)
        
        # Print progress
        if episode % 10 == 0 or episode == 1:
            elapsed = time.time() - start_time
            avg_reward = np.mean(episode_rewards[-10:]) if episode >= 10 else total_reward
            avg_steps = np.mean(episode_steps[-10:]) if episode >= 10 else steps
            
            # Get current angles for debugging
            current_angles = env.get_current_angles()
            
            print(f"Ep {episode:4d}/{max_episodes} | "
                  f"Reward: {total_reward:7.2f} (avg: {avg_reward:7.2f}) | "
                  f"Steps: {steps:3d} | "
                  f"Success: {success} | "
                  f"Error: {avg_error:.4f} | "
                  f"Rate: {success_rate:.1%} | "
                  f"Time: {elapsed/60:5.1f} min"
                  f"Angles: {[f'{v:.2f}' for v in current_angles.values()]}")
        
        # Save best model
        if config["checkpoints"]["save_best"] and total_reward > best_reward:
            best_reward = total_reward
            best_path = os.path.join(CHECKPOINT_DIR, "ppo_best.pt")
            agent.save(best_path)
        
        # Save checkpoint and generate plots periodically
        if episode % save_every == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ppo_checkpoint_ep{episode}.pt")
            agent.save(checkpoint_path)
            
            # Generate plots
            plot_dir = os.path.join(STATS_DIR, "plots")
            generate_training_plots(episode_rewards, episode_steps, 
                                  success_history, episode_errors, plot_dir, config)
            
            print(f"✅ Checkpoint saved: {checkpoint_path}")
        
        # Early stopping
        if early_stopping and episode >= min_episodes and episode >= window_size:
            if success_rate >= success_threshold:
                print(f"\n🎉 Early stopping at episode {episode} with {success_rate:.1%} success rate!")
                break
    
    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "ppo_final.pt")
    agent.save(final_path)
    
    # Generate final plots
    plot_dir = os.path.join(STATS_DIR, "final_plots")
    generate_training_plots(episode_rewards, episode_steps, 
                          success_history, episode_errors, plot_dir, config)
    
    total_time = time.time() - start_time
    final_success_rate = success_history[-1] if success_history else 0.0
    
    print("\n" + "="*70)
    print("✅ PPO TRAINING COMPLETE")
    print("="*70)
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Total steps: {sum(episode_steps)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average reward: {np.mean(episode_rewards):.3f}")
    print(f"Best reward: {best_reward:.3f}")
    print(f"Final success rate: {final_success_rate:.1%}")
    print(f"Final model saved to: {final_path}")
    print(f"Stats saved to: {STATS_DIR}")
    print("="*70)
    
    # Wait a bit before exiting
    for _ in range(50):
        robot.step(config["training"]["timestep"])
    
    robot.simulationSetMode(0) # Pause simulation at the end


def test_mode(config):
    """Run test mode."""
    print("\n" + "="*70)
    print("🤖 OP3 ARM CONTROL - TEST MODE")
    print("="*70)
    
    robot = Supervisor()
    env = ArmTrainingEnvironment(robot, config)
    agent = SimplePPOAgent.load(config["model_path"])
    agent.actor.eval()
    
    print("\n🏁 Starting test...")
    
    try:
        while True:
            obs = env.reset()
            steps = 0
            
            while steps < 50:
                action, _ = agent.get_action(obs, deterministic=True)
                env.apply_action(action)
                env.step()
                steps += 1
                
                current_angles = env.get_current_angles()
                if steps % 10 == 0:
                    print(f"Step {steps}: {current_angles}")
            
            print("-" * 40)
            
    except KeyboardInterrupt:
        print("\n👋 Test ended")


def main():
    """Main function."""
    mode = CONFIG["mode"]
    
    if mode == "train":
        train_mode(CONFIG)
    elif mode == "test":
        test_mode(CONFIG)


if __name__ == "__main__":
    main()