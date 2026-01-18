# Single-Agent (existing - preserved)
python main.py --train --alg=ddpg
python main.py --test --alg=ppo --checkpoint ppo_best.pt

# Multi-Agent Evolutionary Training (NEW)
python main.py --train-multi --alg=ddpg --population-size 8
python main.py --train-multi --alg=ddpg --resume-from <run_dir>

# Multi-Agent Testing (NEW)
python main.py --test-multi --alg=ddpg --run-dir <run_dir>
python main.py --test-multi-models --alg=ppo  # Tests all models in config

# Visualization (NEW)
python main.py --visualize --run-dir <run_dir>