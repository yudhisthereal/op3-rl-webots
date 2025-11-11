# op3_ddpg_env.py
# Main Webots controller for DDPG training
# Usage: Set SCENARIO_CLASS to switch between different training scenarios

from controller import Supervisor
import time
import os
import config
from ddpg_agent import DDPG

# ================== SCENARIO SELECTION ==================
# Change this to switch between scenarios
from scenarios.arm_control_pak_gembong import ArmControlPakGembong
# from scenarios.arm_control_yudhis import ArmControlYudhis

SCENARIO_CLASS = ArmControlPakGembong
# SCENARIO_CLASS = ArmControlYudhis

# ================== INITIALIZATION ==================
# Only execute training code when run directly by Webots (not when imported)
if __name__ == "__main__":
    robot = Supervisor()
    scenario = SCENARIO_CLASS(robot, timestep=config.TIMESTEP)

    # Create DDPG agent
    agent = DDPG(obs_dim=scenario.get_obs_dim(), act_dim=scenario.get_act_dim())

    print(f"Training with scenario: {SCENARIO_CLASS.__name__}")
    print(f"Observation dim: {scenario.get_obs_dim()}, Action dim: {scenario.get_act_dim()}")
    print(f"Max episodes: {config.MAX_EPISODES}, Max steps per episode: {config.MAX_STEPS}")
    print("-" * 60)

    # ================== TRAINING LOOP ==================
    start_time = time.time()

    for ep in range(1, config.MAX_EPISODES + 1):
        # Reset environment at the start of each episode
        obs = scenario.reset()
        total_reward = 0.0
        
        for step in range(config.MAX_STEPS):
            # Get action from agent (with exploration noise)
            action = agent.get_action(obs, add_noise=True)
            
            # Apply action
            scenario.apply_action(action)
            scenario.step()
            
            # Get next observation
            next_obs = scenario.get_observation()
            
            # Compute reward and check if done
            reward, done = scenario.compute_reward(obs, action, next_obs, step + 1)
            
            # Store transition
            agent.store((obs, action, reward, next_obs, float(done)))
            
            # Update agent
            agent.update()
            
            obs = next_obs
            total_reward += reward
            
            if done:
                break
        
        # Logging
        elapsed = time.time() - start_time
        avg_time = elapsed / ep
        eta = avg_time * (config.MAX_EPISODES - ep)
        print(f"Ep {ep}/{config.MAX_EPISODES} | Steps: {step+1:3d} | "
              f"Reward: {total_reward:7.3f} | ETA: {eta/60:.1f} min")

    print("✅ Training finished.")

    # ================== SAVE MODEL ==================
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)
    agent.save(checkpoint_path)
    print(f"✅ Model saved to {checkpoint_path}")

