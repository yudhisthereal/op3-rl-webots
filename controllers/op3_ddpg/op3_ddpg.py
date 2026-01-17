# controllers/op3_ddpg/op3_ddpg.py
#!/usr/bin/env python3
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

# Add project root to path
CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CONTROLLER_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)


def load_config():
    """Load configuration from JSON file, using template if available."""
    config_path = os.path.join(CONTROLLER_DIR, "config.json")
    template_path = os.path.join(CONTROLLER_DIR, "config.json.template")
    
    if not os.path.exists(config_path):
        # Check if template exists, otherwise create default config
        if os.path.exists(template_path):
            print(f"Using template to create config: {template_path}")
            with open(template_path, 'r') as f:
                config = json.load(f)
        else:
            # Create default config with push_force and initial_state
            config = {
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
                    "timestep": 32,
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
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created config file: {config_path}")
        
        return config
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create directories
    CHECKPOINT_DIR = os.path.join(CONTROLLER_DIR, "checkpoints")
    STATS_DIR = os.path.join(CONTROLLER_DIR, "training_stats")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)
    
    # Update model path to be absolute
    if not os.path.isabs(config["model_path"]):
        config["model_path"] = os.path.join(PROJECT_ROOT, config["model_path"])
    
    return config


# Load configuration
CONFIG = load_config()

def save_detailed_stats(episode_stats, episode_num, config, stats_dir):
    """Save detailed training statistics - matches PPO format exactly."""
    os.makedirs(stats_dir, exist_ok=True)
    
    stats_file = os.path.join(stats_dir, f"episode_{episode_num:04d}_stats.json")
    
    with open(stats_file, 'w') as f:
        json.dump(episode_stats, f, indent=2)
    
    # Also save to a cumulative file
    cumulative_file = os.path.join(stats_dir, "all_episodes_summary.csv")
    
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
    return cumulative_file


def generate_training_plots(episode_rewards, episode_steps, success_history, 
                           episode_errors, save_dir, config):
    """Generate and save training plots - matches PPO format."""
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


class Actor(nn.Module):
    """Actor network: observation → action"""
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    """Critic network: (observation, action) → Q-value"""
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer for DDPG"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32).reshape(-1, 1),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32).reshape(-1, 1))
    
    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    """DDPG agent implementation"""
    
    def __init__(self, obs_dim, act_dim, config):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config
        ddpg_config = config["training"]["ddpg"]
        
        # Device
        self.device = torch.device("cpu")
        
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
        
        # Training step counter
        self.train_step = 0
    
    def get_action(self, state, add_noise=True):
        """Get action from actor network with optional exploration noise."""
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().squeeze()
        
        if add_noise:
            # Add exploration noise (Ornstein-Uhlenbeck would be better, but Gaussian is simpler)
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        
        return action
    
    def decay_noise(self):
        """Decay exploration noise over time."""
        self.noise_std = max(self.noise_std * self.noise_decay, self.noise_std_end)
    
    def update(self):
        """Update actor and critic networks."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
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
        return critic_loss.item(), actor_loss.item()
    
    def save(self, filepath):
        """Save the agent's networks to a file."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'noise_std': self.noise_std,
            'config': self.config,
        }, filepath)
        print(f"✅ DDPG model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load an agent from a saved checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        agent = cls(checkpoint['obs_dim'], checkpoint['act_dim'], checkpoint.get('config', CONFIG))
        
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        agent.noise_std = checkpoint.get('noise_std', 0.1)
        
        print(f"✅ DDPG model loaded from {filepath}")
        return agent


class ArmTrainingEnvironment:
    """Environment for training joint angle control"""
    
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
        self.success_reward = reward_config["success_reward"]
        self.angle_error_weight = reward_config["angle_error_weight"]
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
        """Simple reward function for joint angle control."""
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
        max_steps = self.config["training"]["max_steps"]
        if timestep >= max_steps:
            done = True
        
        return reward, done
    
    def is_success(self, obs):
        """Check if current state is a success."""
        return all(abs(delta) < self.angle_tolerance for delta in obs)


def train_mode(config):
    """Run DDPG training mode."""
    print("\n" + "="*70)
    print("🤖 OP3 ARM CONTROL - DDPG TRAINING")
    print("="*70)
    print(f"Control joints: {config['control_joints']}")
    print(f"Goal angles: {config['goal_angles']}")
    print(f"Max episodes: {config['training']['max_episodes']}")
    print("="*70)
    
    robot = Supervisor()
    
    # Create environment
    env = ArmTrainingEnvironment(robot, config)
    
    # Create agent
    agent = DDPGAgent(obs_dim=len(config["control_joints"]),
                     act_dim=len(config["control_joints"]),
                     config=config)
    
    # Training statistics
    episode_rewards = []
    episode_steps = []
    success_history = []
    episode_successes = []
    episode_errors = []
    
    start_time = time.time()
    best_reward = -float('inf')
    
    max_episodes = config["training"]["max_episodes"]
    save_every = config["checkpoints"]["save_every"]
    early_stopping = config["early_stopping"]["enabled"]
    window_size = config["early_stopping"]["window_size"]
    success_threshold = config["early_stopping"]["success_threshold"]
    min_episodes = config["early_stopping"]["min_episodes"]
    
    print("\n🏁 Starting DDPG training...")
    print("-"*70)
    
    for episode in range(1, max_episodes + 1):
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        
        # Track per-step data for detailed stats
        step_rewards = []
        step_errors = []
        step_actions = []
        
        while not done and steps < config["training"]["max_steps"]:
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
                for _ in range(update_times):
                    critic_loss, actor_loss = agent.update()
            
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

        # Save detailed episode stats (matches PPO)
        STATS_DIR = os.path.join(CONTROLLER_DIR, "training_stats")
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

        save_detailed_stats(episode_stats, episode, config, STATS_DIR)
        
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
            
            print(f"Ep {episode:4d} | "
                  f"Reward: {total_reward:7.2f} | "
                  f"Steps: {steps:3d} | "
                  f"Success: {success} | "
                  f"Error: {avg_error:.4f} | "
                  f"Rate: {success_rate:.1%} | "
                  f"Noise: {agent.noise_std:.3f} | "
                  f"Angles: {[f'{v:.2f}' for v in current_angles.values()]}")
        
        # Save best model
        if config["checkpoints"]["save_best"] and total_reward > best_reward:
            best_reward = total_reward
            best_path = os.path.join(CONTROLLER_DIR, "checkpoints", "ddpg_best.pt")
            agent.save(best_path)
        
        # Save checkpoint and generate plots periodically (matches PPO)
        if episode % save_every == 0:
            checkpoint_path = os.path.join(CONTROLLER_DIR, "checkpoints", f"ddpg_checkpoint_ep{episode}.pt")
            agent.save(checkpoint_path)
    
            # Generate plots
            STATS_DIR = os.path.join(CONTROLLER_DIR, "training_stats")
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
    final_path = os.path.join(CONTROLLER_DIR, "checkpoints", "ddpg_final.pt")
    agent.save(final_path)

    # Generate final plots (matches PPO)
    STATS_DIR = os.path.join(CONTROLLER_DIR, "training_stats")
    plot_dir = os.path.join(STATS_DIR, "final_plots")
    generate_training_plots(episode_rewards, episode_steps, 
                        success_history, episode_errors, plot_dir, config)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ DDPG TRAINING COMPLETE")
    print("="*70)
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Total steps: {sum(episode_steps)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average reward: {np.mean(episode_rewards):.3f}")
    print(f"Best reward: {best_reward:.3f}")
    print(f"Final success rate: {success_history[-1] if success_history else 0.0:.1%}")
    print(f"Final model saved to: {final_path}")
    print(f"Stats saved to: {STATS_DIR}")
    print("="*70)
    
    # Wait a bit before exiting
    for _ in range(50):
        robot.step(config["training"]["timestep"])
    
    robot.simulationSetMode(0) # Pause simulation at the end


def test_mode(config):
    """Run test mode with a trained DDPG model."""
    print("\n" + "="*70)
    print("🤖 OP3 ARM CONTROL - TEST MODE (DDPG)")
    print("="*70)
    
    robot = Supervisor()
    env = ArmTrainingEnvironment(robot, config)
    agent = DDPGAgent.load(config["model_path"])
    
    # Set to evaluation mode
    agent.actor.eval()
    agent.actor_target.eval()
    
    print("\n🏁 Starting test simulation...")
    print("Press Ctrl+C in Webots to stop")
    print("-"*70)
    
    episode_count = 0
    try:
        while True:
            episode_count += 1
            obs = env.reset()
            total_reward = 0.0
            steps = 0
            done = False
            
            print(f"\nTest Episode {episode_count}")
            print("-"*40)
            
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
                    print(f"  Step {steps:3d}: Reward: {total_reward:7.2f} | "
                          f"Error: {error:.3f} | "
                          f"Angles: {current_angles}")
                
                obs = next_obs
            
            # Episode summary
            final_angles = env.get_current_angles()
            final_error = np.sum(np.abs(obs))
            
            print(f"\n✅ Episode {episode_count} complete:")
            print(f"   Steps: {steps}")
            print(f"   Total reward: {total_reward:.2f}")
            print(f"   Final error: {final_error:.3f}")
            print(f"   Final angles: {final_angles}")
            print(f"   Goal angles: {config['goal_angles']}")
            print("-"*40)
            
            # Wait between episodes
            for _ in range(30):
                robot.step(config["training"]["timestep"])
                
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    finally:
        print("\nTest simulation ended.")


def main():
    """Main function - read mode from config.json."""
    mode = CONFIG["mode"]
    
    print(f"DDPG Controller - Mode: {mode}")
    
    if mode == "train":
        train_mode(CONFIG)
    elif mode == "test":
        model_path = CONFIG["model_path"]
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("Please train a model first or update model_path in config.json")
            return
        test_mode(CONFIG)
    else:
        print(f"❌ Unknown mode: {mode}")
        print("Please set mode to 'train' or 'test' in config.json")


if __name__ == "__main__":
    main()