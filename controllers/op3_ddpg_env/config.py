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
# Default limits for all joints (can be overridden per scenario)
ANGLE_LIMITS = {
    "ShoulderR": (-1.57, 1.57),
    "ShoulderL": (-1.57, 1.57),
    "ArmUpperR": (-1.57, 1.57),
    "ArmUpperL": (-1.57, 1.57),
    "ArmLowerR": (-1.57, 1.57),
    "ArmLowerL": (-1.57, 1.57),
    "PelvYR": (-1.57, 1.57),
    "PelvYL": (-1.57, 1.57),
    "PelvR": (-1.57, 1.57),
    "PelvL": (-1.57, 1.57),
    "LegUpperR": (-1.57, 1.57),
    "LegUpperL": (-1.57, 1.57),
    "LegLowerR": (-1.57, 1.57),
    "LegLowerL": (-1.57, 1.57),
    "AnkleR": (-1.57, 1.57),
    "AnkleL": (-1.57, 1.57),
    "FootR": (-1.57, 1.57),
    "FootL": (-1.57, 1.57),
    "Neck": (-1.57, 1.57),
    "Head": (-1.57, 1.57),
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

# ================== SAFETY / COLLISION HEURISTICS ==================
# Threshold above which a joint angle is considered "extreme" for self-collision heuristics
EXTREME_JOINT_ANGLE_THRESHOLD = 2.0

