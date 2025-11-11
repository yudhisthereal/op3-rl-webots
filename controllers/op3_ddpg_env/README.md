# Modular DDPG Training Structure

This directory contains a modular implementation of DDPG training for the ROBOTIS OP3 robot.

## Project Structure

```
op3_ddpg_env/
├── config.py                    # Configuration and hyperparameters
├── ddpg_agent.py               # DDPG agent implementation
├── op3_ddpg_env.py             # Main Webots controller (training script)
├── op3_ddpg_env_reference.py  # Original monolithic file (for reference)
├── scenarios/
│   ├── __init__.py
│   ├── base_scenario.py        # Base class for scenarios
│   ├── arm_control_pak_gembong.py  # Scenario with Pak Gembong's reward
│   └── arm_control_yudhis.py       # Scenario with Yudhis's reward
└── checkpoints/                # Saved models
```

## Switching Between Scenarios

To switch between training scenarios, edit `op3_ddpg_env.py` and change the `SCENARIO_CLASS`:

```python
# For Pak Gembong's scenario
from scenarios.arm_control_pak_gembong import ArmControlPakGembong
SCENARIO_CLASS = ArmControlPakGembong

# For Yudhis's scenario
# from scenarios.arm_control_yudhis import ArmControlYudhis
# SCENARIO_CLASS = ArmControlYudhis
```

## Creating a New Scenario

1. Create a new file in `scenarios/` directory (e.g., `scenarios/my_scenario.py`)
2. Inherit from `BaseScenario` and implement all required methods:

```python
from scenarios.base_scenario import BaseScenario
import config

class MyScenario(BaseScenario):
    def __init__(self, robot, timestep=32):
        # Initialize your scenario-specific variables
        super().__init__(robot, timestep)
    
    def _setup_environment(self):
        # Setup motors, sensors, etc.
        pass
    
    def get_obs_dim(self):
        # Return observation space dimension
        return 2
    
    def get_act_dim(self):
        # Return action space dimension
        return 2
    
    def reset(self):
        # Reset environment (called at start of each episode)
        # Must reset episode_step = 0
        self.episode_step = 0
        # ... reset logic ...
        return self.get_observation()
    
    def get_observation(self):
        # Return current observation
        pass
    
    def apply_action(self, action):
        # Apply action to environment
        pass
    
    def compute_reward(self, obs, action, next_obs, step):
        # Compute reward and done flag
        return reward, done
```

3. Import and use in `op3_ddpg_env.py`:

```python
from scenarios.my_scenario import MyScenario
SCENARIO_CLASS = MyScenario
```

## Running Training

Use `op3_ddpg_env.py` as the Webots controller (the filename must match the directory name). The training script will:
- Automatically call `reset()` at the start of each episode
- Save the trained model to `checkpoints/ddpg_model.pt`

## Testing Trained Models

Use the `test_policy` controller in the parent directory. Make sure to set the same scenario class that was used during training.

