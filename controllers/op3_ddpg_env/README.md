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
- Save the trained model to `checkpoints/{scenario_name}/ddpg_model.pt`

### Using main.py Script

The easiest way to run training is using the main script:

```bash
# Standard training
python main.py --train --scenario=pak_gembong
python main.py --train --scenario=yudhis
```

## Testing Trained Models

Use the `test_policy` controller in the parent directory. The controller automatically:
- Detects the scenario being used
- Looks for scenario-specific checkpoints first (e.g., `checkpoints/yudhis/ddpg_model.pt`)
- Falls back to default checkpoint if scenario-specific one doesn't exist
- Can accept custom checkpoint paths via `CHECKPOINT_PATH` environment variable

### Using main.py Script

```bash
# Test with default scenario-specific checkpoint
python main.py --test --scenario=yudhis

# Test with custom checkpoint path
python main.py --test --scenario=yudhis --checkpoint="yudhis/ddpg_model.pt"
python main.py --test --scenario=yudhis --checkpoint="yudhis/best_stage_5.pt"
```

### Checkpoint Directory Structure

Checkpoints are organized by scenario:
```
checkpoints/
├── yudhis/
│   ├── stage_1/
│   │   ├── agent_0.pt
│   │   ├── agent_1.pt
│   │   └── ...
│   ├── best_stage_1.pt
│   ├── best_stage_2.pt
│   └── ddpg_model.pt  # Final best model
└── pak_gembong/
    └── ...
```

## Genetic Algorithm Multi-Agent Training

A genetic algorithm training mode is available that trains multiple agents **in parallel** and uses evolutionary selection:

### Features:
- **Parallel multi-agent training**: Launches multiple Webots instances simultaneously for true parallel training
- **Genetic selection**: After each stage, selects top N performers based on performance metrics
- **Reproduction**: Creates new population from top performers with mutation
- **Staged training**: Runs M stages, improving population over time
- **Automatic Webots exit**: Webots instances close automatically after each stage completes
- **Scenario-specific checkpoints**: All checkpoints saved under `checkpoints/{scenario_name}/`

### Configuration:
Edit `genetic_config.py` to adjust:
- `POPULATION_SIZE`: Number of parallel agents (default: 8)
- `TOP_N`: Number of top performers to reproduce from (default: 3)
- `NUM_STAGES`: Number of training stages (default: 5)
- `EPISODES_PER_STAGE`: Episodes per agent per stage (default: 50)
- `MUTATION_RATE`: Probability of mutation during reproduction (default: 0.1)
- `MUTATION_STRENGTH`: Standard deviation of mutation noise (default: 0.05)
- `ELITE_COPY_RATE`: Fraction of population that are exact copies of elites (default: 0.3)
- `RANKING_METRIC`: How to rank agents ('avg_reward', 'max_reward', 'success_rate', 'final_distance')

### Usage:
```bash
# Parallel genetic training
python main.py --genetic --scenario=pak_gembong
python main.py --genetic --scenario=yudhis
```

### How It Works:

1. **Stage 1**: Initialize population of N random agents
2. **Parallel Training**: Launch N Webots instances simultaneously, each training one agent
3. **Result Collection**: Collect performance metrics from all agents
4. **Selection**: Rank agents and select top N performers
5. **Reproduction**: Create new population from elites:
   - Some are exact copies (elites)
   - Others are mutated copies for diversity
6. **Repeat**: Continue for specified number of stages

### Output:
- Individual agent checkpoints: `checkpoints/{scenario}/stage_{N}/agent_{id}.pt`
- Best agent per stage: `checkpoints/{scenario}/best_stage_{N}.pt`
- Final best agent: `checkpoints/{scenario}/ddpg_model.pt`

**Note**: The `main.py` script automatically handles controller file management. Webots instances exit automatically after training completes.

