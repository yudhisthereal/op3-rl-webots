# config.py
# Configuration for SAC training

# ================== TRAINING CONFIG ==================
MAX_EPISODES = 10000
MAX_STEPS = 1000
TIMESTEP = 32  # Webots timestep in milliseconds

# ================== NETWORK CONFIG ==================
HIDDEN_SIZE = 256
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4

# ================== SAC SPECIFIC ==================
GAMMA = 0.99
TAU = 0.005  # Target network update rate
ALPHA = 0.2  # Temperature parameter
REPLAY_BUFFER_SIZE = 100000
BATCH_SIZE = 256

# ================== JOINT CONFIG ==================
# All 20 joint names
JOINT_NAMES = [
    "ShoulderR", "ShoulderL",
    "ArmUpperR", "ArmUpperL",
    "ArmLowerR", "ArmLowerL",
    "PelvYR", "PelvYL",
    "PelvR", "PelvL",
    "LegUpperR", "LegUpperL",
    "LegLowerR", "LegLowerL",
    "AnkleR", "AnkleL",
    "FootR", "FootL",
    "Neck", "Head"
]

# Angle limits for each joint (in radians)
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
    "Head": (-1.57, 1.57)
}

# ================== DIRECTORIES ==================
import os
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)