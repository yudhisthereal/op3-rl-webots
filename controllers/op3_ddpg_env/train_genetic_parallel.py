# train_genetic_parallel.py
# Parallel multi-agent genetic algorithm training
# Launches multiple Webots instances in parallel for true multi-agent training

import multiprocessing as mp
import subprocess
import os
import sys
import time
import numpy as np
import shutil
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import config
import genetic_config
from ddpg_agent import DDPG

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
WORLD_FILE = PROJECT_ROOT / "worlds" / "robotis_op3_train.wbt"
CONTROLLER_DIR = Path(__file__).parent


def find_webots():
    """Find Webots executable."""
    webots_home = os.environ.get("WEBOTS_HOME")
    if webots_home:
        webots_path = os.path.join(webots_home, "webots")
        if os.path.exists(webots_path):
            return webots_path
    
    possible_paths = [
        "/usr/local/webots/webots",
        "/opt/webots/webots",
        os.path.expanduser("~/webots/webots"),
    ]
    
    import shutil
    webots_path = shutil.which("webots")
    if webots_path:
        return webots_path
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def create_agent_controller(agent_id, stage, checkpoint_path, scenario_class_name):
    """
    Create a temporary controller file for a specific agent.
    
    Returns:
        Path to the created controller file
    """
    controller_template = f"""# Temporary controller for agent {agent_id}
from controller import Supervisor
import time
import os
import numpy as np
import config
from ddpg_agent import DDPG

# Scenario
from scenarios.{scenario_class_name.lower()} import {scenario_class_name}
SCENARIO_CLASS = {scenario_class_name}

if __name__ == "__main__":
    robot = Supervisor()
    scenario = SCENARIO_CLASS(robot, timestep=config.TIMESTEP)
    
    # Load or create agent
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent = DDPG.load(checkpoint_path)
        agent.mutate(mutation_rate=0.05, mutation_strength=0.02)
    else:
        agent = DDPG(obs_dim=scenario.get_obs_dim(), act_dim=scenario.get_act_dim())
    
    # Training
    episode_rewards = []
    episode_metrics = []  # Generic metric tracking (distance, acceleration, etc.)
    success_count = 0
    
    for ep in range(1, {genetic_config.EPISODES_PER_STAGE} + 1):
        obs = scenario.reset()
        total_reward = 0.0
        min_metric = float('inf')
        
        for step in range(config.MAX_STEPS):
            action = agent.get_action(obs, add_noise=True)
            scenario.apply_action(action)
            scenario.step()
            next_obs = scenario.get_observation()
            reward, done = scenario.compute_reward(obs, action, next_obs, step + 1)
            
            agent.store((obs, action, reward, next_obs, float(done)))
            agent.update()
            
            # Get episode metric from scenario (e.g., distance to target, acceleration)
            metric = scenario.get_episode_metric(next_obs)
            min_metric = min(min_metric, metric)
            
            obs = next_obs
            total_reward += reward
            
            if done:
                # Check success using scenario's success criteria
                if scenario.is_success(next_obs, done):
                    success_count += 1
                break
        
        episode_rewards.append(total_reward)
        episode_metrics.append(min_metric)
    
    # Save agent
    checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, f"stage_{{stage}}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    agent_checkpoint = os.path.join(checkpoint_dir, f"agent_{{agent_id}}.pt")
    agent.save(agent_checkpoint)
    
    # Save results
    results = {{
        'agent_id': {agent_id},
        'checkpoint_path': agent_checkpoint,
        'avg_reward': np.mean(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'success_rate': success_count / {genetic_config.EPISODES_PER_STAGE},
        'avg_episode_metric': np.mean(episode_metrics),
    }}
    
    result_file = os.path.join(checkpoint_dir, f"agent_{{agent_id}}_results.json")
    with open(result_file, 'w') as f:
        json.dump(results, f)
"""
    
    # Replace placeholders
    controller_code = controller_template.format(
        agent_id=agent_id,
        stage=stage,
        checkpoint_path=f'"{checkpoint_path}"' if checkpoint_path else 'None'
    )
    
    # Create temporary controller directory and file
    temp_controller_dir = CONTROLLER_DIR / f"temp_agent_{agent_id}"
    temp_controller_dir.mkdir(exist_ok=True)
    
    # Copy necessary files
    for file in ['config.py', 'ddpg_agent.py', 'scenarios']:
        src = CONTROLLER_DIR / file
        if src.is_dir():
            shutil.copytree(src, temp_controller_dir / file, dirs_exist_ok=True)
        elif src.exists():
            shutil.copy(src, temp_controller_dir / file)
    
    # Write controller file
    controller_file = temp_controller_dir / "op3_ddpg_env.py"
    with open(controller_file, 'w') as f:
        f.write(controller_code)
    
    return temp_controller_dir


def train_agent_parallel(agent_id, stage, checkpoint_path, scenario_class_name, world_file):
    """
    Train a single agent in a separate Webots process.
    """
    try:
        # Create temporary controller
        controller_dir = create_agent_controller(agent_id, stage, checkpoint_path, scenario_class_name)
        
        # Launch Webots in headless mode
        webots_path = find_webots()
        if not webots_path:
            raise RuntimeError("Webots not found")
        
        # Run Webots
        cmd = [
            webots_path,
            "--mode=fast",
            "--no-rendering",
            "--stdout",
            "--stderr",
            str(world_file.absolute())
        ]
        
        # Change to controller directory
        env = os.environ.copy()
        # Webots will look for controller in the directory matching the controller name
        # So we need to set the controller path
        
        result = subprocess.run(
            cmd,
            cwd=str(controller_dir),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        # Load results
        checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, f"stage_{stage}")
        result_file = os.path.join(checkpoint_dir, f"agent_{agent_id}_results.json")
        
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'agent_id': agent_id,
                'error': 'Results file not found',
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
    except Exception as e:
        return {
            'agent_id': agent_id,
            'error': str(e)
        }
    finally:
        # Cleanup temporary directory
        if 'controller_dir' in locals() and controller_dir.exists():
            shutil.rmtree(controller_dir, ignore_errors=True)


def rank_agents(results, metric='avg_reward'):
    """Rank agents based on specified metric."""
    valid_results = [r for r in results if 'error' not in r]
    if metric == 'avg_reward':
        return sorted(valid_results, key=lambda x: x.get('avg_reward', -float('inf')), reverse=True)
    elif metric == 'max_reward':
        return sorted(valid_results, key=lambda x: x.get('max_reward', -float('inf')), reverse=True)
    elif metric == 'success_rate':
        return sorted(valid_results, key=lambda x: x.get('success_rate', 0), reverse=True)
    elif metric == 'final_distance':
        return sorted(valid_results, key=lambda x: x.get('avg_episode_metric', float('inf')), reverse=False)
    else:
        return sorted(valid_results, key=lambda x: x.get('avg_reward', -float('inf')), reverse=True)


def reproduce_population(top_agents, population_size, stage, scenario_class_name):
    """Create new population from top performers."""
    new_checkpoints = []
    num_elites = len(top_agents)
    
    for i in range(population_size):
        # Select parent
        parent_idx = min(i % num_elites, num_elites - 1)
        parent_checkpoint = top_agents[parent_idx]['checkpoint_path']
        
        # Load and mutate
        new_agent = DDPG.load(parent_checkpoint)
        
        if i < int(population_size * genetic_config.ELITE_COPY_RATE):
            new_agent.mutate(mutation_rate=0.01, mutation_strength=0.001)
        else:
            new_agent.mutate(
                mutation_rate=genetic_config.MUTATION_RATE,
                mutation_strength=genetic_config.MUTATION_STRENGTH
            )
        
        # Save
        checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, f"stage_{stage}_reproduced")
        os.makedirs(checkpoint_dir, exist_ok=True)
        new_checkpoint = os.path.join(checkpoint_dir, f"agent_{i}.pt")
        new_agent.save(new_checkpoint)
        new_checkpoints.append(new_checkpoint)
    
    return new_checkpoints


def main():
    """Main parallel genetic algorithm training."""
    # Get scenario from command line or default
    scenario_class_name = sys.argv[1] if len(sys.argv) > 1 else "ArmControlYudhis"
    
    print("=" * 70)
    print("🧬 PARALLEL GENETIC ALGORITHM MULTI-AGENT TRAINING")
    print("=" * 70)
    print(f"Population size: {genetic_config.POPULATION_SIZE}")
    print(f"Top N: {genetic_config.TOP_N}")
    print(f"Number of stages: {genetic_config.NUM_STAGES}")
    print(f"Episodes per stage: {genetic_config.EPISODES_PER_STAGE}")
    print(f"Scenario: {scenario_class_name}")
    print("=" * 70)
    
    # Initialize population
    current_checkpoints = [None] * genetic_config.POPULATION_SIZE
    
    for stage in range(1, genetic_config.NUM_STAGES + 1):
        print(f"\n{'='*70}")
        print(f"STAGE {stage}/{genetic_config.NUM_STAGES}")
        print(f"{'='*70}")
        
        # Train all agents in parallel
        print(f"Training {genetic_config.POPULATION_SIZE} agents in parallel...")
        start_time = time.time()
        
        # Use multiprocessing Pool
        with mp.Pool(processes=genetic_config.POPULATION_SIZE) as pool:
            args = [
                (agent_id, stage, current_checkpoints[agent_id], scenario_class_name, WORLD_FILE)
                for agent_id in range(genetic_config.POPULATION_SIZE)
            ]
            results = pool.starmap(train_agent_parallel, args)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time/60:.1f} minutes")
        
        # Rank agents
        ranked_results = rank_agents(results, metric=genetic_config.RANKING_METRIC)
        
        # Display results
        print(f"\n📊 Stage {stage} Results:")
        print("-" * 70)
        for i, result in enumerate(ranked_results[:genetic_config.TOP_N]):
            print(f"  Rank {i+1} (Agent {result['agent_id']}):")
            print(f"    Avg Reward: {result.get('avg_reward', 0):.3f}")
            print(f"    Max Reward: {result.get('max_reward', 0):.3f}")
            print(f"    Success Rate: {result.get('success_rate', 0):.1%}")
            print(f"    Avg Episode Metric: {result.get('avg_episode_metric', 0):.4f}")
        
        # Select top N
        top_agents = ranked_results[:genetic_config.TOP_N]
        
        # Save best agent
        best_agent = top_agents[0]
        best_checkpoint = os.path.join(config.CHECKPOINT_DIR, f"best_stage_{stage}.pt")
        shutil.copy(best_agent['checkpoint_path'], best_checkpoint)
        print(f"\n✅ Best agent saved to: {best_checkpoint}")
        
        # Reproduce for next stage
        if stage < genetic_config.NUM_STAGES:
            print(f"\n🧬 Reproducing population for stage {stage + 1}...")
            current_checkpoints = reproduce_population(
                top_agents,
                genetic_config.POPULATION_SIZE,
                stage,
                scenario_class_name
            )
            print(f"✅ Created {len(current_checkpoints)} new agents")
    
    # Final best agent
    final_best = os.path.join(config.CHECKPOINT_DIR, f"best_stage_{genetic_config.NUM_STAGES}.pt")
    final_checkpoint = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)
    if os.path.exists(final_best):
        shutil.copy(final_best, final_checkpoint)
    
    print(f"\n{'='*70}")
    print(f"✅ PARALLEL GENETIC TRAINING COMPLETE")
    print(f"Final best agent saved to: {final_checkpoint}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

