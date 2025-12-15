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
# from scenarios.arm_control_yudhis import ArmControlYudhis
# from scenarios.arm_control_pak_gembong import ArmControlPakGembong
from scenarios.fall_control import FallControl

# ================== SCENARIO SELECTION ==================
# Should match the scenario used during training
# SCENARIO_CLASS = ArmControlPakGembong
# SCENARIO_CLASS = ArmControlYudhis
SCENARIO_CLASS = FallControl

NUM_TEST_EPISODES = 5

# ================== INITIALIZATION ==================
robot = Supervisor()
scenario = SCENARIO_CLASS(robot, timestep=config.TIMESTEP)

# ================== LOAD MODEL ==================
# Get checkpoint path from environment variable or use default
checkpoint_arg = os.environ.get('CHECKPOINT_PATH', None)

if checkpoint_arg:
    # Checkpoint path provided (can be relative like "yudhis/ddpg_model.pt" or full path)
    if '/' in checkpoint_arg or '\\' in checkpoint_arg:
        # Has subdirectory - construct full path
        path_parts = checkpoint_arg.replace('\\', '/').split('/')
        checkpoint_path = os.path.join(
            os.path.dirname(__file__), '..', 'op3_ddpg_env',
            config.CHECKPOINT_DIR, *path_parts
        )
    else:
        # Just filename - use default location
        checkpoint_path = os.path.join(
            os.path.dirname(__file__), '..', 'op3_ddpg_env',
            config.CHECKPOINT_DIR, checkpoint_arg
        )
else:
    # Default: try scenario-specific checkpoint first, then fallback to default
    # Use scenario_name from scenario class
    scenario_name = SCENARIO_CLASS.scenario_name
    
    scenario_checkpoint = os.path.join(
        os.path.dirname(__file__), '..', 'op3_ddpg_env',
        config.CHECKPOINT_DIR, scenario_name, config.CHECKPOINT_NAME
    )
    
    default_checkpoint = os.path.join(
        os.path.dirname(__file__), '..', 'op3_ddpg_env',
        config.CHECKPOINT_DIR, config.CHECKPOINT_NAME
    )
    
    # Try scenario-specific first, then default
    if os.path.exists(scenario_checkpoint):
        checkpoint_path = scenario_checkpoint
    else:
        checkpoint_path = default_checkpoint

if not os.path.exists(checkpoint_path):
    print(f"❌ Error: Model checkpoint not found at {checkpoint_path}")
    print("Please train the model first or specify checkpoint with CHECKPOINT_PATH environment variable")
    print(f"Example: CHECKPOINT_PATH='yudhis/ddpg_model.pt' python test_policy.py")
    robot.step(config.TIMESTEP)
    sys.exit(1)

print(f"Loading model from {checkpoint_path}...")
agent = DDPG.load(checkpoint_path)
agent.actor.eval()  # Set to evaluation mode

print("✅ Model loaded successfully!")
print(f"Testing policy for {NUM_TEST_EPISODES} episodes...")
print(f"Scenario: {SCENARIO_CLASS.__name__}")
if hasattr(scenario, 'TARGET'):
    print(f"Target angles: {scenario.TARGET}")
print("-" * 60)

# ================== TEST LOOP ==================
for ep in range(1, NUM_TEST_EPISODES + 1):
    # Reset environment at the start of each episode
    obs = scenario.reset()
    total_reward = 0.0
    
    for step in range(config.MAX_STEPS):
        # Get action from trained policy (no exploration noise)
        action = agent.get_action(obs, add_noise=False)
        
        # Apply action
        scenario.apply_action(action)
        scenario.step()
        
        # Get next observation
        next_obs = scenario.get_observation()
        
        # Compute reward and check if done
        reward, done, _ = scenario.compute_reward(obs, action, next_obs, step + 1)
        
        total_reward += reward
        obs = next_obs
        
    print(f"Episode {ep}/{NUM_TEST_EPISODES} | Steps: {step+1:3d} | "
            f"Final obs: [{obs[0]:6.3f}, {obs[1]:6.3f}] | "
            f"Total reward: {total_reward:.3f}")
    
    # Small pause between episodes
    for _ in range(10):
        robot.step(config.TIMESTEP)

print("-" * 60)
print("✅ Testing completed!")
