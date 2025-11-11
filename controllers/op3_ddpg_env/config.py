# config.py
# Configuration and hyperparameters

# ================== SIMULATION CONFIG ==================
TIMESTEP = 32
MAX_STEPS = 200
MAX_EPISODES = 300

# ================== JOINT CONFIG ==================
JOINT_NAMES = [
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR",
    "ArmLowerL", "PelvYR", "PelvYL", "PelvR", "PelvL",
    "LegUpperR", "LegUpperL", "LegLowerR", "LegLowerL", "AnkleR",
    "AnkleL", "FootR", "FootL", "Neck", "Head"
]

# Per-joint mechanical limits (Webots OP3)
ANGLE_LIMITS = {
    "ShoulderL": (-1.57, 1.57),
    "ArmUpperL": (-1.57, 1.57),
    "ShoulderR": (-1.57, 1.57),
    "ArmUpperR": (-1.57, 1.57),
}

# ================== DDPG HYPERPARAMETERS ==================
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
GAMMA = 0.98
TAU = 0.005
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EXPLORATION_NOISE = 0.1

# ================== MODEL SAVING ==================
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAME = "ddpg_model.pt"

