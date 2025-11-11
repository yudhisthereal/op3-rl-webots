#!/usr/bin/env python3
"""
Parallel genetic algorithm training launcher.
Launches multiple Webots instances in parallel, each training one agent.
"""

import multiprocessing as mp
import subprocess
import os
import sys
import time
import json
import shutil
from pathlib import Path

# Add current directory to path
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
    
    import shutil as shutil_module
    webots_path = shutil_module.which("webots")
    if webots_path:
        return webots_path
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def launch_agent_webots(agent_id, stage, checkpoint_path, scenario_class_name):
    """
    Launch a single Webots instance to train one agent.
    """
    webots_path = find_webots()
    if not webots_path:
        raise RuntimeError("Webots not found. Set WEBOTS_HOME or add webots to PATH.")
    
    # Set environment variables for the agent
    env = os.environ.copy()
    env['AGENT_ID'] = str(agent_id)
    env['STAGE'] = str(stage)
    env['SCENARIO_NAME'] = scenario_class_name.lower()
    if checkpoint_path:
        env['CHECKPOINT_PATH'] = str(checkpoint_path)
    else:
        env.pop('CHECKPOINT_PATH', None)
    
    # Copy genetic controller to main controller location (Webots requirement)
    genetic_controller = CONTROLLER_DIR / "op3_ddpg_genetic_parallel.py"
    main_controller = CONTROLLER_DIR / "op3_ddpg_env.py"
    
    # Backup original if exists
    backup_path = main_controller.with_suffix('.py.backup')
    if main_controller.exists():
        shutil.copy(main_controller, backup_path)
    
    # Copy genetic controller
    shutil.copy(genetic_controller, main_controller)
    
    try:
        # Launch Webots in headless batch mode (exits automatically)
        cmd = [
            webots_path,
            "--mode=fast",
            "--no-rendering",
            "--batch",  # Exit automatically when simulation ends
            "--stdout",
            "--stderr",
            str(WORLD_FILE.absolute())
        ]
        
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(CONTROLLER_DIR),
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        # Restore original controller
        if backup_path.exists():
            shutil.copy(backup_path, main_controller)
            backup_path.unlink()
        
        # Load results (with scenario name in path)
        scenario_name = scenario_class_name.lower()
        checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, scenario_name, f"stage_{stage}")
        result_file = os.path.join(checkpoint_dir, f"agent_{agent_id}_results.json")
        
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'agent_id': agent_id,
                'error': 'Results file not found',
                'stdout': result.stdout[-500:] if result.stdout else '',
                'stderr': result.stderr[-500:] if result.stderr else ''
            }
            
    except subprocess.TimeoutExpired:
        return {'agent_id': agent_id, 'error': 'Timeout'}
    except Exception as e:
        return {'agent_id': agent_id, 'error': str(e)}
    finally:
        # Restore original controller if backup exists
        if backup_path.exists():
            shutil.copy(backup_path, main_controller)
            backup_path.unlink()


def rank_agents(results, metric='avg_reward'):
    """Rank agents based on specified metric."""
    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        return results
    
    if metric == 'avg_reward':
        return sorted(valid_results, key=lambda x: x.get('avg_reward', -float('inf')), reverse=True)
    elif metric == 'max_reward':
        return sorted(valid_results, key=lambda x: x.get('max_reward', -float('inf')), reverse=True)
    elif metric == 'success_rate':
        return sorted(valid_results, key=lambda x: x.get('success_rate', 0), reverse=True)
    elif metric == 'final_distance':
        return sorted(valid_results, key=lambda x: x.get('avg_final_distance', float('inf')), reverse=False)
    else:
        return sorted(valid_results, key=lambda x: x.get('avg_reward', -float('inf')), reverse=True)


def reproduce_population(top_agents, population_size, stage, scenario_class_name):
    """Create new population from top performers."""
    new_checkpoints = []
    num_elites = len(top_agents)
    scenario_name = scenario_class_name.lower()
    
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
        
        # Save (with scenario name in path)
        checkpoint_base_dir = os.path.join(config.CHECKPOINT_DIR, scenario_name)
        checkpoint_dir = os.path.join(checkpoint_base_dir, f"stage_{stage}_reproduced")
        os.makedirs(checkpoint_dir, exist_ok=True)
        new_checkpoint = os.path.join(checkpoint_dir, f"agent_{i}.pt")
        new_agent.save(new_checkpoint)
        new_checkpoints.append(new_checkpoint)
    
    return new_checkpoints


def main():
    """Main parallel genetic algorithm training."""
    # Get scenario class name from command line or default
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
        print(f"Launching {genetic_config.POPULATION_SIZE} Webots instances in parallel...")
        start_time = time.time()
        
        # Use multiprocessing Pool to launch Webots instances
        with mp.Pool(processes=genetic_config.POPULATION_SIZE) as pool:
            args = [
                (agent_id, stage, current_checkpoints[agent_id], scenario_class_name)
                for agent_id in range(genetic_config.POPULATION_SIZE)
            ]
            results = pool.starmap(launch_agent_webots, args)
        
        training_time = time.time() - start_time
        print(f"\nAll agents completed in {training_time/60:.1f} minutes")
        
        # Rank agents
        ranked_results = rank_agents(results, metric=genetic_config.RANKING_METRIC)
        
        # Display results
        print(f"\n📊 Stage {stage} Results:")
        print("-" * 70)
        for i, result in enumerate(ranked_results[:genetic_config.TOP_N]):
            if 'error' in result:
                print(f"  Rank {i+1} (Agent {result['agent_id']}): ERROR - {result['error']}")
            else:
                print(f"  Rank {i+1} (Agent {result['agent_id']}):")
                print(f"    Avg Reward: {result.get('avg_reward', 0):.3f}")
                print(f"    Max Reward: {result.get('max_reward', 0):.3f}")
                print(f"    Success Rate: {result.get('success_rate', 0):.1%}")
                print(f"    Avg Final Distance: {result.get('avg_final_distance', 0):.4f}")
        
        # Select top N (filter out errors)
        top_agents = [r for r in ranked_results[:genetic_config.TOP_N] if 'error' not in r]
        
        if not top_agents:
            print("❌ No successful agents in this stage!")
            break
        
        # Save best agent (with scenario name in path)
        best_agent = top_agents[0]
        scenario_name = scenario_class_name.lower()
        checkpoint_base_dir = os.path.join(config.CHECKPOINT_DIR, scenario_name)
        os.makedirs(checkpoint_base_dir, exist_ok=True)
        best_checkpoint = os.path.join(checkpoint_base_dir, f"best_stage_{stage}.pt")
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
    
    # Final best agent (with scenario name in path)
    scenario_name = scenario_class_name.lower()
    checkpoint_base_dir = os.path.join(config.CHECKPOINT_DIR, scenario_name)
    final_best = os.path.join(checkpoint_base_dir, f"best_stage_{genetic_config.NUM_STAGES}.pt")
    final_checkpoint = os.path.join(checkpoint_base_dir, config.CHECKPOINT_NAME)
    if os.path.exists(final_best):
        shutil.copy(final_best, final_checkpoint)
    
    print(f"\n{'='*70}")
    print(f"✅ PARALLEL GENETIC TRAINING COMPLETE")
    print(f"Final best agent saved to: {final_checkpoint}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

