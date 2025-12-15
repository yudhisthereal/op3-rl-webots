# genetic_config.py
# Configuration for genetic algorithm multi-agent training

# ================== EVOLUTIONARY ALGORITHM CONFIG ==================
POPULATION_SIZE = 2  # Number of agents in population
TOP_N = 2  # Number of top performers to reproduce from
NUM_STAGES = 3  # Number of training stages
EPISODES_PER_STAGE = 1000  # Episodes per agent per stage

# ================== REPRODUCTION CONFIG ==================
MUTATION_RATE = 0.1  # Probability of mutating each parameter during reproduction
MUTATION_STRENGTH = 0.05  # Standard deviation of mutation noise
ELITE_COPY_RATE = 0.3  # Fraction of population that are exact copies of elites

# ================== PERFORMANCE METRICS ==================
# How to rank agents (options: 'avg_reward', 'max_reward', 'success_rate', 'final_distance')
RANKING_METRIC = 'avg_reward'

