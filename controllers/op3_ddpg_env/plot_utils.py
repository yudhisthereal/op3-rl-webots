# plot_utils.py
# Utility functions for generating training plots

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def moving_average(data, window_size):
    """
    Moving average that starts at the first data point.
    - For i < window_size, average over data[:i+1]
    - For i >= window_size, average over trailing 'window_size' points
    Returns an array with the same length as 'data'.
    """
    n = len(data)
    if n == 0:
        return np.array([])
    if window_size <= 1:
        return np.array(data, dtype=float)
    arr = np.array(data, dtype=float)
    out = np.empty(n, dtype=float)
    # Prefix: expanding window
    limit = min(window_size - 1, n - 1)
    for i in range(limit + 1):
        out[i] = np.mean(arr[: i + 1])
    # Trailing fixed-size window
    if n > window_size - 1:
        csum = np.cumsum(arr)
        start = window_size - 1
        for i in range(start, n):
            total = csum[i] - (csum[i - window_size] if i - window_size >= 0 else 0.0)
            out[i] = total / window_size
    return out


def generate_training_plots(
    episode_rewards,
    episode_timesteps,
    last_episode_acceleration,
    last_episode_speed,
    last_episode_rewards,
    last_episode_timesteps,
    plots_dir,
    agent_id=None,
    stage=None,
    window_size=10,
    include_accel_speed=True
):
    """
    Generate all training plots.
    
    Args:
        episode_rewards: List of total rewards per episode
        episode_timesteps: List of timesteps per episode
        last_episode_acceleration: List of acceleration values for last episode
        last_episode_speed: List of speed values for last episode
        last_episode_rewards: List of rewards per timestep for last episode
        last_episode_timesteps: List of timesteps for last episode
        plots_dir: Directory to save plots
        agent_id: Optional agent ID for plot titles
        stage: Optional stage number for plot titles
        window_size: Window size for moving average smoothing
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    title_suffix = ""
    if agent_id is not None:
        title_suffix += f"Agent {agent_id}"
    if stage is not None:
        if title_suffix:
            title_suffix += f" - Stage {stage}"
        else:
            title_suffix = f"Stage {stage}"
    
    try:
        # Plot 1: Acceleration and Speed over timestep (last episode) - RAW only (no smoothing)
        if include_accel_speed and len(last_episode_timesteps) > 0 and len(last_episode_acceleration) > 0 and len(last_episode_speed) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            ax1.plot(last_episode_timesteps, last_episode_acceleration, 'r-', linewidth=1.5, label='Acceleration Magnitude')
            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('Acceleration (m/s²)')
            ax1.set_title(f'{title_suffix} - Acceleration over Timestep (Last Episode)' if title_suffix else 'Acceleration over Timestep (Last Episode)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2.plot(last_episode_timesteps, last_episode_speed, 'b-', linewidth=1.5, label='Speed')
            ax2.set_xlabel('Timestep')
            ax2.set_ylabel('Speed (m/s)')
            ax2.set_title(f'{title_suffix} - Speed over Timestep (Last Episode)' if title_suffix else 'Speed over Timestep (Last Episode)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            filename = f'agent_{agent_id}_accel_speed_last_episode.png' if agent_id is not None else 'accel_speed_last_episode.png'
            plt.savefig(os.path.join(plots_dir, filename), dpi=150)
            plt.close()
        
        # Plot 2: Reward over timestep (last episode) - RAW only (no smoothing)
        if len(last_episode_timesteps) > 0 and len(last_episode_rewards) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(last_episode_timesteps, last_episode_rewards, 'g-', linewidth=1.5, label='Reward')
            plt.xlabel('Timestep')
            plt.ylabel('Reward')
            plt.title(f'{title_suffix} - Reward over Timestep (Last Episode)' if title_suffix else 'Reward over Timestep (Last Episode)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            filename = f'agent_{agent_id}_reward_timestep_last_episode.png' if agent_id is not None else 'reward_timestep_last_episode.png'
            plt.savefig(os.path.join(plots_dir, filename), dpi=150)
            plt.close()
        
        # Plot 3: Reward over episodes (raw + moving average starting from first point)
        if len(episode_rewards) > 0:
            plt.figure(figsize=(10, 6))
            episodes = list(range(1, len(episode_rewards) + 1))
            
            # Plot raw data
            plt.plot(episodes, episode_rewards, 'b-', linewidth=0.5, alpha=0.3, label='Episode Reward (raw)')
            
            # Plot smoothed data
            if len(episode_rewards) > 1 and window_size > 1:
                reward_smooth = moving_average(episode_rewards, window_size)
                plt.plot(episodes, reward_smooth, 'b-', linewidth=2, label=f'Moving Average (window={window_size})')
            
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title(f'{title_suffix} - Reward over Episodes' if title_suffix else 'Reward over Episodes')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            filename = f'agent_{agent_id}_reward_episodes.png' if agent_id is not None else 'reward_episodes.png'
            plt.savefig(os.path.join(plots_dir, filename), dpi=150)
            plt.close()
        
        # Plot 4: Timesteps over episodes (raw + moving average starting from first point)
        if len(episode_timesteps) > 0:
            plt.figure(figsize=(10, 6))
            episodes = list(range(1, len(episode_timesteps) + 1))
            
            # Plot raw data
            plt.plot(episodes, episode_timesteps, 'm-', linewidth=0.5, alpha=0.3, label='Timesteps per Episode (raw)')
            
            # Plot smoothed data
            if len(episode_timesteps) > 1 and window_size > 1:
                timesteps_smooth = moving_average(episode_timesteps, window_size)
                plt.plot(episodes, timesteps_smooth, 'm-', linewidth=2, label=f'Moving Average (window={window_size})')
            
            plt.xlabel('Episode')
            plt.ylabel('Timesteps')
            plt.title(f'{title_suffix} - Timesteps over Episodes' if title_suffix else 'Timesteps over Episodes')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            filename = f'agent_{agent_id}_timesteps_episodes.png' if agent_id is not None else 'timesteps_episodes.png'
            plt.savefig(os.path.join(plots_dir, filename), dpi=150)
            plt.close()
        
        return True
    except Exception as e:
        print(f"Warning - Could not generate plots: {e}")
        return False

