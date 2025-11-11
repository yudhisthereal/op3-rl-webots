# test_policy.py
# Webots controller to test trained DDPG policy
# Loads the saved model and demonstrates the agent's behavior

from controller import Supervisor
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'op3_ddpg_env'))

# Import directly from files (not as package) to avoid executing op3_ddpg_env.py
import config
from ddpg_agent import DDPG
from scenarios.arm_control_yudhis import ArmControlYudhis
# from scenarios.arm_control_pak_gembong import ArmControlPakGembong

# ================== SCENARIO SELECTION ==================
# Should match the scenario used during training
# SCENARIO_CLASS = ArmControlPakGembong
SCENARIO_CLASS = ArmControlYudhis

NUM_TEST_EPISODES = 5

# ================== INITIALIZATION ==================
robot = Supervisor()
scenario = SCENARIO_CLASS(robot, timestep=config.TIMESTEP)

# ================== LOAD MODEL ==================
checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'op3_ddpg_env', 
                                config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)

if not os.path.exists(checkpoint_path):
    print(f"❌ Error: Model checkpoint not found at {checkpoint_path}")
    print("Please train the model first using train.py")
    robot.step(config.TIMESTEP)
    sys.exit(1)

print(f"Loading model from {checkpoint_path}...")
agent = DDPG.load(checkpoint_path)
agent.actor.eval()  # Set to evaluation mode

print("✅ Model loaded successfully!")
print(f"Testing policy for {NUM_TEST_EPISODES} episodes...")
print(f"Scenario: {SCENARIO_CLASS.__name__}")
print(f"Target angles: {scenario.TARGET}")
print("-" * 60)

# ================== TEST LOOP ==================
for ep in range(1, NUM_TEST_EPISODES + 1):
    # Reset environment at the start of each episode
    obs = scenario.reset()
    total_reward = 0.0
    min_dist = float('inf')
    
    for step in range(config.MAX_STEPS):
        # Get action from trained policy (no exploration noise)
        action = agent.get_action(obs, add_noise=False)
        
        # Apply action
        scenario.apply_action(action)
        scenario.step()
        
        # Get next observation
        next_obs = scenario.get_observation()
        
        # Compute reward and check if done
        reward, done = scenario.compute_reward(obs, action, next_obs, step + 1)
        
        # Compute distance to target for logging
        dist = np.linalg.norm(next_obs - scenario.TARGET)
        min_dist = min(min_dist, dist)
        
        total_reward += reward
        obs = next_obs
        
        # Check if target reached
        if done or dist < 0.01:
            success_msg = " | ✅ SUCCESS" if dist < 0.01 else ""
            print(f"Episode {ep}/{NUM_TEST_EPISODES} | Steps: {step+1:3d} | "
                  f"Final angles: [{next_obs[0]:6.3f}, {next_obs[1]:6.3f}] | "
                  f"Distance: {dist:.4f}{success_msg}")
            break
    
    if dist >= 0.01 and not done:
        print(f"Episode {ep}/{NUM_TEST_EPISODES} | Steps: {step+1:3d} | "
              f"Final angles: [{obs[0]:6.3f}, {obs[1]:6.3f}] | "
              f"Min distance: {min_dist:.4f} | Total reward: {total_reward:.3f}")
    
    # Small pause between episodes
    for _ in range(10):
        robot.step(config.TIMESTEP)

print("-" * 60)
print("✅ Testing completed!")
