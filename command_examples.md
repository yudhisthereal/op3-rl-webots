# Single-Agent Training
python main.py --train --controller=op3_ddpg # Uses config_train.json from controller dir

# Multi-Agent Evolutionary Training (NEW)
python main.py --train-multi --controller=op3_ddpg --population-size 8 # Overrides multi_agent_config.json
python main.py --train-multi --controller=op3_ddpg_abs --resume-from <run_dir>

# Multi-Agent Testing (NEW)
python main.py --test --controller=op3_ddpg_abs --run-dir <run_dir> # Tests specific run
python main.py --test --controller=op3_ppo # Tests all models in config_test.json

# Visualization (NEW)
python main.py --visualize --run-dir <run_dir>

# Global Options
--seed 42 # Random seed (default: 42)
