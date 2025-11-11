# train_genetic.py
# Multi-agent training with genetic algorithm
# Runs multiple agents in parallel, selects top performers, and reproduces for next stage

import multiprocessing as mp
import numpy as np
import os
import sys
import time
import torch
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import config
import genetic_config
from ddpg_agent import DDPG
from scenarios.arm_control_pak_gembong import ArmControlPakGembong
# from scenarios.arm_control_yudhis import ArmControlYudhis

SCENARIO_CLASS = ArmControlPakGembong
# SCENARIO_CLASS = ArmControlYudhis


def train_agent_worker(agent_id, stage, checkpoint_path=None, result_queue=None):
    """
    Worker function to train a single agent.
    This runs in a separate process and simulates training.
    
    Args:
        agent_id: Unique identifier for this agent
        stage: Current training stage
        checkpoint_path: Path to load agent from (None for new agent)
        result_queue: Queue to return results
        
    Returns:
        Dictionary with agent_id, performance metrics, and checkpoint path
    """
    # Import here to avoid issues with multiprocessing
    from controller import Supervisor
    
    try:
        # Initialize robot and scenario
        robot = Supervisor()
        scenario = SCENARIO_CLASS(robot, timestep=config.TIMESTEP)
        
        # Create or load agent
        if checkpoint_path and os.path.exists(checkpoint_path):
            agent = DDPG.load(checkpoint_path)
            # Apply small mutation for diversity
            agent.mutate(mutation_rate=0.05, mutation_strength=0.02)
        else:
            agent = DDPG(obs_dim=scenario.get_obs_dim(), act_dim=scenario.get_act_dim())
        
        # Training loop for this stage
        episode_rewards = []
        episode_distances = []
        success_count = 0
        
        for ep in range(1, genetic_config.EPISODES_PER_STAGE + 1):
            obs = scenario.reset()
            total_reward = 0.0
            min_dist = float('inf')
            
            for step in range(config.MAX_STEPS):
                action = agent.get_action(obs, add_noise=True)
                scenario.apply_action(action)
                scenario.step()
                next_obs = scenario.get_observation()
                reward, done = scenario.compute_reward(obs, action, next_obs, step + 1)
                
                agent.store((obs, action, reward, next_obs, float(done)))
                agent.update()
                
                dist = np.linalg.norm(next_obs - scenario.TARGET)
                min_dist = min(min_dist, dist)
                
                obs = next_obs
                total_reward += reward
                
                if done or dist < 0.01:
                    if dist < 0.01:
                        success_count += 1
                    break
            
            episode_rewards.append(total_reward)
            episode_distances.append(min_dist)
        
        # Save agent checkpoint
        checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, f"stage_{stage}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        agent_checkpoint = os.path.join(checkpoint_dir, f"agent_{agent_id}.pt")
        agent.save(agent_checkpoint)
        
        # Calculate performance metrics
        avg_reward = np.mean(episode_rewards)
        max_reward = np.max(episode_rewards)
        success_rate = success_count / genetic_config.EPISODES_PER_STAGE
        avg_final_distance = np.mean(episode_distances)
        
        result = {
            'agent_id': agent_id,
            'checkpoint_path': agent_checkpoint,
            'avg_reward': avg_reward,
            'max_reward': max_reward,
            'success_rate': success_rate,
            'avg_final_distance': avg_final_distance,
            'episode_rewards': episode_rewards,
        }
        
        if result_queue:
            result_queue.put(result)
        
        return result
        
    except Exception as e:
        print(f"Error in agent {agent_id}: {e}")
        import traceback
        traceback.print_exc()
        if result_queue:
            result_queue.put({'agent_id': agent_id, 'error': str(e)})
        return None


def rank_agents(results, metric='avg_reward'):
    """
    Rank agents based on specified metric.
    
    Args:
        results: List of agent result dictionaries
        metric: Metric to rank by ('avg_reward', 'max_reward', 'success_rate', 'final_distance')
        
    Returns:
        Sorted list of results (best first)
    """
    if metric == 'avg_reward':
        return sorted(results, key=lambda x: x['avg_reward'], reverse=True)
    elif metric == 'max_reward':
        return sorted(results, key=lambda x: x['max_reward'], reverse=True)
    elif metric == 'success_rate':
        return sorted(results, key=lambda x: x['success_rate'], reverse=True)
    elif metric == 'final_distance':
        return sorted(results, key=lambda x: x['avg_final_distance'], reverse=False)
    else:
        return sorted(results, key=lambda x: x['avg_reward'], reverse=True)


def reproduce_population(top_agents, population_size, stage):
    """
    Create new population from top performers using genetic algorithm.
    
    Args:
        top_agents: List of top-performing agent results
        population_size: Size of new population
        stage: Current stage number
        
    Returns:
        List of checkpoint paths for new agents
    """
    new_checkpoints = []
    num_elites = len(top_agents)
    
    for i in range(population_size):
        # Select parent (with preference for better performers)
        parent_idx = min(i % num_elites, num_elites - 1)
        parent_checkpoint = top_agents[parent_idx]['checkpoint_path']
        
        # Create new agent from parent
        new_agent = DDPG.load(parent_checkpoint)
        
        # Apply mutation (stronger for non-elite copies)
        if i < int(population_size * genetic_config.ELITE_COPY_RATE):
            # Exact copies of elites (with tiny mutation for numerical stability)
            new_agent.mutate(mutation_rate=0.01, mutation_strength=0.001)
        else:
            # Mutated copies
            new_agent.mutate(
                mutation_rate=genetic_config.MUTATION_RATE,
                mutation_strength=genetic_config.MUTATION_STRENGTH
            )
        
        # Save new agent
        checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, f"stage_{stage}_reproduced")
        os.makedirs(checkpoint_dir, exist_ok=True)
        new_checkpoint = os.path.join(checkpoint_dir, f"agent_{i}.pt")
        new_agent.save(new_checkpoint)
        new_checkpoints.append(new_checkpoint)
    
    return new_checkpoints


def main():
    """Main genetic algorithm training loop."""
    print("=" * 70)
    print("🧬 GENETIC ALGORITHM MULTI-AGENT TRAINING")
    print("=" * 70)
    print(f"Population size: {genetic_config.POPULATION_SIZE}")
    print(f"Top N: {genetic_config.TOP_N}")
    print(f"Number of stages: {genetic_config.NUM_STAGES}")
    print(f"Episodes per stage: {genetic_config.EPISODES_PER_STAGE}")
    print(f"Ranking metric: {genetic_config.RANKING_METRIC}")
    print("=" * 70)
    
    # Initialize population (first stage)
    current_checkpoints = [None] * genetic_config.POPULATION_SIZE
    
    for stage in range(1, genetic_config.NUM_STAGES + 1):
        print(f"\n{'='*70}")
        print(f"STAGE {stage}/{genetic_config.NUM_STAGES}")
        print(f"{'='*70}")
        
        # Train all agents in parallel
        print(f"Training {genetic_config.POPULATION_SIZE} agents in parallel...")
        start_time = time.time()
        
        # Use multiprocessing to train agents
        processes = []
        result_queue = mp.Queue()
        
        for agent_id in range(genetic_config.POPULATION_SIZE):
            checkpoint = current_checkpoints[agent_id] if current_checkpoints[agent_id] else None
            p = mp.Process(
                target=train_agent_worker,
                args=(agent_id, stage, checkpoint, result_queue)
            )
            p.start()
            processes.append(p)
        
        # Collect results
        results = []
        for _ in range(genetic_config.POPULATION_SIZE):
            result = result_queue.get()
            if 'error' not in result:
                results.append(result)
        
        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f} seconds")
        
        # Rank agents
        ranked_results = rank_agents(results, metric=genetic_config.RANKING_METRIC)
        
        # Display results
        print(f"\n📊 Stage {stage} Results:")
        print("-" * 70)
        for i, result in enumerate(ranked_results[:genetic_config.TOP_N]):
            print(f"  Rank {i+1} (Agent {result['agent_id']}):")
            print(f"    Avg Reward: {result['avg_reward']:.3f}")
            print(f"    Max Reward: {result['max_reward']:.3f}")
            print(f"    Success Rate: {result['success_rate']:.1%}")
            print(f"    Avg Final Distance: {result['avg_final_distance']:.4f}")
        
        # Select top N
        top_agents = ranked_results[:genetic_config.TOP_N]
        
        # Save best agent
        best_agent = top_agents[0]
        best_checkpoint = os.path.join(config.CHECKPOINT_DIR, f"best_stage_{stage}.pt")
        import shutil
        shutil.copy(best_agent['checkpoint_path'], best_checkpoint)
        print(f"\n✅ Best agent saved to: {best_checkpoint}")
        
        # Reproduce for next stage (if not last stage)
        if stage < genetic_config.NUM_STAGES:
            print(f"\n🧬 Reproducing population for stage {stage + 1}...")
            current_checkpoints = reproduce_population(
                top_agents,
                genetic_config.POPULATION_SIZE,
                stage
            )
            print(f"✅ Created {len(current_checkpoints)} new agents")
    
    # Final best agent
    final_best = os.path.join(config.CHECKPOINT_DIR, f"best_stage_{genetic_config.NUM_STAGES}.pt")
    final_checkpoint = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)
    import shutil
    shutil.copy(final_best, final_checkpoint)
    print(f"\n{'='*70}")
    print(f"✅ GENETIC TRAINING COMPLETE")
    print(f"Final best agent saved to: {final_checkpoint}")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Note: This script is designed to run outside Webots
    # For Webots integration, we'll need a different approach
    print("⚠️  This genetic training script requires running outside Webots")
    print("⚠️  For Webots integration, use the Webots-based training controller")
    main()

