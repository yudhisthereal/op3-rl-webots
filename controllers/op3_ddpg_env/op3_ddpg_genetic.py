# op3_ddpg_genetic.py
# Webots controller for multi-agent genetic algorithm training
# Runs multiple agents sequentially, selects top performers, and reproduces for next stage

from controller import Supervisor
import time
import os
import numpy as np
import config
import genetic_config
from ddpg_agent import DDPG

# ================== SCENARIO SELECTION ==================
# from scenarios.arm_control_pak_gembong import ArmControlPakGembong
from scenarios.arm_control_yudhis import ArmControlYudhis

# SCENARIO_CLASS = ArmControlPakGembong
SCENARIO_CLASS = ArmControlYudhis


def train_agent(robot, scenario, agent, agent_id, stage, episodes):
    """
    Train a single agent for a number of episodes.
    
    Returns:
        Dictionary with performance metrics
    """
    episode_rewards = []
    episode_distances = []
    success_count = 0
    
    for ep in range(1, episodes + 1):
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
        
        # Progress update every 10 episodes
        if ep % 10 == 0:
            print(f"  Agent {agent_id} | Ep {ep}/{episodes} | "
                  f"Avg Reward: {np.mean(episode_rewards[-10:]):.3f}")
    
    return {
        'agent_id': agent_id,
        'avg_reward': np.mean(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'success_rate': success_count / episodes,
        'avg_final_distance': np.mean(episode_distances),
        'episode_rewards': episode_rewards,
    }


def rank_agents(results, metric='avg_reward'):
    """Rank agents based on specified metric."""
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


# ================== MAIN TRAINING ==================
if __name__ == "__main__":
    robot = Supervisor()
    scenario = SCENARIO_CLASS(robot, timestep=config.TIMESTEP)
    
    print("=" * 70)
    print("🧬 GENETIC ALGORITHM MULTI-AGENT TRAINING")
    print("=" * 70)
    print(f"Population size: {genetic_config.POPULATION_SIZE}")
    print(f"Top N: {genetic_config.TOP_N}")
    print(f"Number of stages: {genetic_config.NUM_STAGES}")
    print(f"Episodes per stage: {genetic_config.EPISODES_PER_STAGE}")
    print(f"Ranking metric: {genetic_config.RANKING_METRIC}")
    print("=" * 70)
    
    # Initialize population
    agents = []
    for i in range(genetic_config.POPULATION_SIZE):
        agent = DDPG(obs_dim=scenario.get_obs_dim(), act_dim=scenario.get_act_dim())
        agents.append(agent)
    
    # Training stages
    for stage in range(1, genetic_config.NUM_STAGES + 1):
        print(f"\n{'='*70}")
        print(f"STAGE {stage}/{genetic_config.NUM_STAGES}")
        print(f"{'='*70}")
        
        # Train all agents
        results = []
        stage_start_time = time.time()
        
        for agent_id, agent in enumerate(agents):
            print(f"\nTraining Agent {agent_id + 1}/{genetic_config.POPULATION_SIZE}...")
            result = train_agent(
                robot, scenario, agent, agent_id, stage,
                genetic_config.EPISODES_PER_STAGE
            )
            
            # Save agent checkpoint
            checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, f"stage_{stage}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"agent_{agent_id}.pt")
            agent.save(checkpoint_path)
            result['checkpoint_path'] = checkpoint_path
            
            results.append(result)
        
        stage_time = time.time() - stage_start_time
        print(f"\nStage {stage} completed in {stage_time/60:.1f} minutes")
        
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
        top_agents_results = ranked_results[:genetic_config.TOP_N]
        
        # Save best agent
        best_result = top_agents_results[0]
        best_checkpoint = os.path.join(config.CHECKPOINT_DIR, f"best_stage_{stage}.pt")
        import shutil
        shutil.copy(best_result['checkpoint_path'], best_checkpoint)
        print(f"\n✅ Best agent saved to: {best_checkpoint}")
        
        # Reproduce for next stage (if not last stage)
        if stage < genetic_config.NUM_STAGES:
            print(f"\n🧬 Reproducing population for stage {stage + 1}...")
            
            # Load top agents
            top_agents = []
            for result in top_agents_results:
                top_agent = DDPG.load(result['checkpoint_path'])
                top_agents.append(top_agent)
            
            # Create new population
            new_agents = []
            num_elites = len(top_agents)
            
            for i in range(genetic_config.POPULATION_SIZE):
                # Select parent
                parent_idx = min(i % num_elites, num_elites - 1)
                parent = top_agents[parent_idx]
                
                # Create new agent from parent
                new_agent = DDPG(obs_dim=scenario.get_obs_dim(), act_dim=scenario.get_act_dim())
                new_agent.copy_from(parent)
                
                # Apply mutation (stronger for non-elite copies)
                if i < int(genetic_config.POPULATION_SIZE * genetic_config.ELITE_COPY_RATE):
                    # Exact copies of elites (with tiny mutation for numerical stability)
                    new_agent.mutate(mutation_rate=0.01, mutation_strength=0.001)
                else:
                    # Mutated copies
                    new_agent.mutate(
                        mutation_rate=genetic_config.MUTATION_RATE,
                        mutation_strength=genetic_config.MUTATION_STRENGTH
                    )
                
                new_agents.append(new_agent)
            
            agents = new_agents
            print(f"✅ Created {len(agents)} new agents for next stage")
    
    # Final best agent
    final_best = os.path.join(config.CHECKPOINT_DIR, f"best_stage_{genetic_config.NUM_STAGES}.pt")
    final_checkpoint = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)
    import shutil
    if os.path.exists(final_best):
        shutil.copy(final_best, final_checkpoint)
    
    print(f"\n{'='*70}")
    print(f"✅ GENETIC TRAINING COMPLETE")
    print(f"Final best agent saved to: {final_checkpoint}")
    print(f"{'='*70}")

