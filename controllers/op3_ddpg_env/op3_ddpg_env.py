# op3_ddpg_genetic_parallel.py
# Webots controller for parallel multi-agent genetic training
# Each Webots instance trains one agent, identified by AGENT_ID environment variable

from controller import Supervisor
import time
import os
import numpy as np
import config
import genetic_config
from ddpg_agent import DDPG

# ================== SCENARIO SELECTION ==================
# This section is automatically updated by main.py
from scenarios.arm_control_yudhis import ArmControlYudhis
# from scenarios.arm_control_pak_gembong import ArmControlPakGembong
# from scenarios.fall_control import FallControl

SCENARIO_CLASS = ArmControlYudhis
# SCENARIO_CLASS = ArmControlPakGembong
# SCENARIO_CLASS = FallControl

# ================== GET AGENT INFO FROM ENVIRONMENT ==================
AGENT_ID = int(os.environ.get('AGENT_ID', '0'))
STAGE = int(os.environ.get('STAGE', '1'))
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', None)
SCENARIO_NAME = os.environ.get('SCENARIO_NAME', 'default')

if __name__ == "__main__":
    robot = Supervisor()
    scenario = SCENARIO_CLASS(robot, timestep=config.TIMESTEP)
    
    print(f"Agent {AGENT_ID} starting training (Stage {STAGE})...")
    
    # Load or create agent
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        agent = DDPG.load(CHECKPOINT_PATH)
        # Apply small mutation for diversity
        agent.mutate(mutation_rate=0.05, mutation_strength=0.02)
        print(f"Agent {AGENT_ID}: Loaded from checkpoint")
    else:
        agent = DDPG(obs_dim=scenario.get_obs_dim(), act_dim=scenario.get_act_dim())
        print(f"Agent {AGENT_ID}: Created new agent")
    
    # Training loop
    episode_rewards = []
    episode_metrics = []  # Generic metric tracking (distance, acceleration, etc.)
    success_count = 0
    
    for ep in range(1, genetic_config.EPISODES_PER_STAGE + 1):
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
        
        if ep % 10 == 0:
            print(f"Agent {AGENT_ID} | Ep {ep}/{genetic_config.EPISODES_PER_STAGE} | "
                  f"Avg Reward: {np.mean(episode_rewards[-10:]):.3f}")
    
    # Save agent (with scenario name in path)
    # Use scenario_name from scenario class
    scenario_name = SCENARIO_CLASS.scenario_name
    
    checkpoint_base_dir = os.path.join(config.CHECKPOINT_DIR, scenario_name)
    checkpoint_dir = os.path.join(checkpoint_base_dir, f"stage_{STAGE}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"agent_{AGENT_ID}.pt")
    agent.save(checkpoint_path)
    
    # Save results to file
    results = {
        'agent_id': AGENT_ID,
        'checkpoint_path': checkpoint_path,
        'avg_reward': float(np.mean(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'success_rate': float(success_count / genetic_config.EPISODES_PER_STAGE),
        'avg_episode_metric': float(np.mean(episode_metrics)),
    }
    
    import json
    result_file = os.path.join(checkpoint_dir, f"agent_{AGENT_ID}_results.json")
    with open(result_file, 'w') as f:
        json.dump(results, f)
    
    print(f"Agent {AGENT_ID} completed!")
    print(f"  Avg Reward: {results['avg_reward']:.3f}")
    print(f"  Success Rate: {results['success_rate']:.1%}")
    
    # Exit Webots automatically
    print(f"Agent {AGENT_ID} exiting Webots...")
    robot.simulationQuit(0)

