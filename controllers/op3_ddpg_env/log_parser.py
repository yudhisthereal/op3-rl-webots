import re
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple, Dict

def parse_training_log(log_file_path: str) -> Dict[str, List[float]]:
    """
    Parse training log file and extract episode numbers, steps, rewards, and ETA.
    
    Args:
        log_file_path: Path to the training log file
        
    Returns:
        Dictionary containing lists of episode numbers, steps, rewards, and ETA values
    """
    # Regular expressions to match the log format
    episode_pattern = r"Ep (\d+)/(\d+)"
    steps_pattern = r"Steps:\s*(\d+)"
    reward_pattern = r"Reward:\s*([-\d.]+)"
    eta_pattern = r"ETA:\s*([\d.]+) min"
    
    episodes = []
    total_episodes = 0
    steps_list = []
    rewards = []
    eta_list = []
    
    with open(log_file_path, 'r') as file:
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Extract episode information
            ep_match = re.search(episode_pattern, line)
            steps_match = re.search(steps_pattern, line)
            reward_match = re.search(reward_pattern, line)
            eta_match = re.search(eta_pattern, line)
            
            if ep_match and steps_match and reward_match:
                current_ep = int(ep_match.group(1))
                total_ep = int(ep_match.group(2))
                steps = int(steps_match.group(1))
                reward = float(reward_match.group(1))
                eta = float(eta_match.group(1)) if eta_match else None
                
                episodes.append(current_ep)
                total_episodes = total_ep
                steps_list.append(steps)
                rewards.append(reward)
                if eta is not None:
                    eta_list.append(eta)
    
    return {
        'episodes': episodes,
        'total_episodes': total_episodes,
        'steps': steps_list,
        'rewards': rewards,
        'eta': eta_list if eta_list else None
    }

def plot_training_from_log(log_file_path: str, save_plots: bool = True, show_plots: bool = True):
    """
    Generate comprehensive training plots from a log file.
    
    Args:
        log_file_path: Path to the training log file
        save_plots: Whether to save the plots as images
        show_plots: Whether to display the plots
    """
    # Parse the log file
    data = parse_training_log(log_file_path)
    
    if not data['episodes']:
        print("No valid training data found in the log file.")
        return
    
    episodes = data['episodes']
    steps = data['steps']
    rewards = data['rewards']
    eta = data['eta']
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Reward per episode
    ax1.plot(episodes, rewards, 'b-', alpha=0.7, linewidth=1)
    ax1.set_title('Training: Reward per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3)
    
    # Add moving average for rewards
    if len(rewards) > 10:
        window = min(10, len(rewards) // 10)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, 
                label=f'Moving Avg (window={window})')
        ax1.legend()
    
    # Plot 2: Steps per episode
    ax2.plot(episodes, steps, 'g-', alpha=0.7, linewidth=1)
    ax2.set_title('Training: Steps per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True, alpha=0.3)
    
    # Add moving average for steps
    if len(steps) > 10:
        window = min(10, len(steps) // 10)
        moving_avg_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
        ax2.plot(episodes[window-1:], moving_avg_steps, 'orange', linewidth=2,
                label=f'Moving Avg (window={window})')
        ax2.legend()
    
    # Plot 3: Reward distribution (histogram)
    ax3.hist(rewards, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax3.set_title('Distribution of Episode Rewards')
    ax3.set_xlabel('Reward')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Add vertical line for mean reward
    mean_reward = np.mean(rewards)
    ax3.axvline(mean_reward, color='red', linestyle='--', 
               label=f'Mean: {mean_reward:.3f}')
    ax3.legend()
    
    # Plot 4: ETA over time (if available)
    if eta and len(eta) > 0:
        ax4.plot(episodes[:len(eta)], eta, 'orange', alpha=0.7, linewidth=1)
        ax4.set_title('Estimated Time Remaining')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('ETA (minutes)')
        ax4.grid(True, alpha=0.3)
        
        # Add a trend line for ETA
        if len(eta) > 5:
            z = np.polyfit(episodes[:len(eta)], eta, 1)
            p = np.poly1d(z)
            ax4.plot(episodes[:len(eta)], p(episodes[:len(eta)]), 'r--',
                    label='Trend line')
            ax4.legend()
    else:
        # If no ETA data, plot steps vs rewards scatter
        ax4.scatter(steps, rewards, alpha=0.5, c=episodes, cmap='viridis')
        ax4.set_title('Steps vs Reward (colored by episode)')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Reward')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(ax4.collections[0], ax=ax4, label='Episode')
    
    plt.tight_layout()
    
    # Save plots if requested
    if save_plots:
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Generate output filename based on input log file
        base_name = os.path.splitext(os.path.basename(log_file_path))[0]
        output_path = f'results/{base_name}_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plots saved to: {output_path}")
    
    # Show plots if requested
    if show_plots:
        plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("TRAINING SUMMARY STATISTICS")
    print("="*50)
    print(f"Total episodes processed: {len(episodes)}/{data['total_episodes']}")
    print(f"Average reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Min reward: {np.min(rewards):.3f}")
    print(f"Max reward: {np.max(rewards):.3f}")
    print(f"Average steps per episode: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print(f"Best episode (highest reward): Episode {episodes[np.argmax(rewards)]}")
    
    return data

def plot_multiple_logs_comparison(log_files: List[Tuple[str, str]], save_plots: bool = True):
    """
    Compare multiple training runs from different log files.
    
    Args:
        log_files: List of tuples (file_path, label)
        save_plots: Whether to save the comparison plot
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (log_file, label) in enumerate(log_files):
        data = parse_training_log(log_file)
        if data['episodes']:
            color = colors[i % len(colors)]
            plt.plot(data['episodes'], data['rewards'], 
                    color=color, alpha=0.7, label=label, linewidth=1.5)
    
    plt.title('Comparison of Training Runs: Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_plots:
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/training_comparison.png', dpi=300, bbox_inches='tight')
        print("📊 Comparison plot saved to: results/training_comparison.png")
    
    plt.show()

def export_training_data(log_file_path: str, output_format: str = 'csv'):
    """
    Export parsed training data to a structured format.
    
    Args:
        log_file_path: Path to the training log file
        output_format: Output format ('csv', 'npy', or 'json')
    """
    data = parse_training_log(log_file_path)
    
    if not data['episodes']:
        print("No data to export.")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(log_file_path))[0]
    
    if output_format == 'csv':
        import csv
        output_file = f'results/{base_name}_data.csv'
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Steps', 'Reward', 'ETA'])
            for ep, st, rw, et in zip(data['episodes'], data['steps'], 
                                    data['rewards'], 
                                    data['eta'] if data['eta'] else [None]*len(data['episodes'])):
                writer.writerow([ep, st, rw, et])
        print(f"📁 Data exported to CSV: {output_file}")
    
    elif output_format == 'npy':
        output_file = f'results/{base_name}_data.npz'
        np.savez(output_file, **data)
        print(f"📁 Data exported to NPZ: {output_file}")
    
    elif output_format == 'json':
        import json
        output_file = f'results/{base_name}_data.json'
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"📁 Data exported to JSON: {output_file}")

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot training results from log file')
    parser.add_argument('log_file', help='Path to the training log file')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')
    parser.add_argument('--no-save', action='store_true', help='Do not save plots')
    parser.add_argument('--export', choices=['csv', 'json', 'npy'], 
                       help='Export data to specified format')
    
    args = parser.parse_args()
    
    # Generate plots
    data = plot_training_from_log(
        args.log_file, 
        save_plots=not args.no_save,
        show_plots=not args.no_show
    )
    
    # Export data if requested
    if args.export:
        export_training_data(args.log_file, args.export)