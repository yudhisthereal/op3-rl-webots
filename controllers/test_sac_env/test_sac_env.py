#!/usr/bin/env python3
"""
Test controller for SAC agent.
"""

from controller import Supervisor
import numpy as np
import os
import sys
import json
import time
import matplotlib.pyplot as plt

# Add project root to path
CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CONTROLLER_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from scenarios.fall_control import FallControl
from controllers.op3_sac_env.sac_agent import SACAgent

# Get config (use SAC config)
from controllers.op3_sac_env import config as train_config


def plot_test_results(rewards, timesteps, joint_errors, save_dir):
    """Plot test results."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Cumulative reward
    axes[0, 0].plot(np.cumsum(rewards))
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Cumulative Reward')
    axes[0, 0].set_title('Cumulative Reward Over Time')
    axes[0, 0].grid(True)
    
    # Plot 2: Instantaneous reward
    axes[0, 1].plot(rewards)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].set_title('Instantaneous Reward')
    axes[0, 1].grid(True)
    
    # Plot 3: Joint errors
    if joint_errors:
        joint_errors = np.array(joint_errors)
        for i in range(min(5, joint_errors.shape[1])):  # Plot first 5 joints
            axes[1, 0].plot(joint_errors[:, i], label=f'Joint {i}')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Error (rad)')
        axes[1, 0].set_title('Joint Angle Errors')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot 4: Reward distribution
    axes[1, 1].hist(rewards, bins=20, alpha=0.7)
    axes[1, 1].axvline(x=np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.3f}')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'test_results.png')
    plt.savefig(plot_path, dpi=100)
    plt.close()
    
    print(f"Test plots saved to: {plot_path}")


if __name__ == "__main__":
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    
    print("=" * 70)
    print("🤖 SAC AGENT TESTING")
    print("=" * 70)
    
    # Create scenario
    scenario = FallControl(robot, timestep=train_config.TIMESTEP, algorithm='sac')
    
    # Load checkpoint
    checkpoint_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'op3_sac_env',
        'checkpoints',
        'sac_final.pt'
    )
    
    # Override with environment variable if set
    if 'CHECKPOINT_PATH' in os.environ:
        env_path = os.environ['CHECKPOINT_PATH']
        if not os.path.isabs(env_path):
            checkpoint_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                'op3_sac_env',
                'checkpoints',
                env_path
            )
        else:
            checkpoint_path = env_path
    
    print(f"Loading SAC agent from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pt'):
                    print(f"  - {f}")
        robot.simulationQuit(1)
    
    try:
        agent = SACAgent.load(checkpoint_path)
        print("✅ SAC agent loaded successfully")
        print(f"  Observation dim: {agent.obs_dim}")
        print(f"  Action dim: {agent.act_dim}")
        print(f"  Temperature (alpha): {agent.alpha:.4f}")
    except Exception as e:
        print(f"❌ Failed to load agent: {e}")
        robot.simulationQuit(1)
    
    # Test loop
    print("\n" + "-" * 70)
    print("Starting test episode...")
    print("-" * 70)
    
    obs = scenario.reset()
    total_reward = 0.0
    step = 0
    rewards = []
    joint_errors = []
    
    while robot.step(timestep) != -1 and step < train_config.MAX_STEPS:
        # Get action (deterministic for testing)
        action = agent.get_action(obs, deterministic=True)
        
        # Apply action
        scenario.apply_action(action)
        
        # Get next observation and reward
        next_obs = scenario.get_observation()
        reward, done, termination_reason = scenario.compute_reward(obs, action, next_obs, step + 1)
        
        # Track joint errors
        current_joints = next_obs[:20]
        joint_error = []
        for i, joint_name in enumerate(scenario.CONTROL_JOINTS[:20]):
            current_pos = current_joints[i]
            goal_pos = scenario.GOAL_POSITIONS.get(joint_name, 0.0)
            joint_error.append(abs(current_pos - goal_pos))
        joint_errors.append(joint_error)
        
        total_reward += reward
        rewards.append(reward)
        obs = next_obs
        step += 1
        
        if step % 50 == 0:
            avg_joint_error = np.mean(joint_error) if joint_error else 0.0
            print(f"Step {step:4d} | Reward: {reward:7.3f} | Total: {total_reward:7.3f} | Avg Joint Error: {avg_joint_error:.4f}")
        
        if done:
            print(f"\nEpisode terminated: {termination_reason}")
            break
    
    # Results
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS")
    print("=" * 70)
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Average reward per step: {np.mean(rewards):.3f}")
    print(f"Max reward: {np.max(rewards):.3f}")
    print(f"Min reward: {np.min(rewards):.3f}")
    
    # Calculate average joint error
    if joint_errors:
        avg_joint_errors = np.mean(joint_errors, axis=0)
        print(f"\nAverage joint errors:")
        for i, joint_name in enumerate(scenario.CONTROL_JOINTS[:min(10, len(avg_joint_errors))]):
            print(f"  {joint_name:12s}: {avg_joint_errors[i]:.4f} rad")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(checkpoint_path), 'test_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save data
    results_data = {
        'total_steps': step,
        'total_reward': float(total_reward),
        'average_reward': float(np.mean(rewards)),
        'max_reward': float(np.max(rewards)),
        'min_reward': float(np.min(rewards)),
        'rewards': [float(r) for r in rewards],
        'termination_reason': termination_reason if 'termination_reason' in locals() else "max_steps",
        'checkpoint_path': checkpoint_path,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    if joint_errors:
        results_data['average_joint_errors'] = [float(e) for e in avg_joint_errors]
    
    results_file = os.path.join(results_dir, f'test_{time.strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate plots
    plot_test_results(rewards, list(range(step)), joint_errors, results_dir)
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)
    
    robot.simulationQuit(0)