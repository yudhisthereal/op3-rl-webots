# common_config.py
# Common configuration shared across all algorithms

# ================== JOINT CONFIGURATION ==================
# All 20 joint names in OP3 robot
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

# Standard angle limits for each joint (in radians)
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

# ================== JOINT SAFETY THRESHOLDS ==================
# Extreme joint angle thresholds (radians)
EXTREME_ANGLE_THRESHOLD = 1.5  # 1.5 rad ≈ 86 degrees
CRITICAL_ANGLE_THRESHOLD = 2.0  # 2.0 rad ≈ 115 degrees
MAX_SAFE_ANGLE = 3.0  # 3.0 rad ≈ 172 degrees (absolute maximum)

# Joint velocity safety limits (rad/s)
MAX_JOINT_VELOCITY = 10.0  # Maximum safe joint velocity
CRITICAL_JOINT_VELOCITY = 15.0  # Critical joint velocity

# Joint acceleration safety limits (rad/s²)
MAX_JOINT_ACCELERATION = 50.0
CRITICAL_JOINT_ACCELERATION = 100.0

# ================== ROBOT PHYSICS PARAMETERS ==================
# Center of Mass (CoM) thresholds
MIN_COM_HEIGHT = 0.15  # Minimum CoM height before considered "fallen" (meters)
STANDING_COM_HEIGHT = 0.292665  # Default standing CoM height
LOW_COM_WARNING = 0.20  # Warning threshold for low CoM

# Falling detection
FALLEN_ANGLE_THRESHOLD = 0.8  # 0.8 rad ≈ 46 degrees torso tilt
CRITICAL_FALL_ANGLE = 1.2  # 1.2 rad ≈ 69 degrees (severe fall)

# Push parameters (for fall_control scenario)
PUSH_FORCE_MAGNITUDE = 500.0  # Newtons
PUSH_DELAY_SECONDS = 0.5  # Seconds before push is applied
PUSH_OFFSET = [0.0, 0.0, 0.3]  # Push offset from CoM (meters)

# ================== REWARD FUNCTION PARAMETERS ==================
# Joint error rewards
ERROR_REWARD_SCALE = 2.0
ERROR_EXPONENT = 3.0  # exp(-error * ERROR_EXPONENT)

# Progress bonus
PROGRESS_BONUS_SCALE = 0.5

# Smoothness penalties
SMOOTHNESS_PENALTY_SCALE = 0.1
ACTION_MAGNITUDE_PENALTY_SCALE = 0.05  # For SAC

# Extreme position penalties
EXTREME_POSITION_PENALTY = 0.1

# Success thresholds
SUCCESS_JOINT_ERROR_THRESHOLD = 0.2  # Average joint error < 0.2 rad for success
SUCCESS_COM_RATIO_THRESHOLD = 0.7  # CoM height > 70% of initial height

# Termination conditions
MAX_EXTREME_JOINTS = 5  # Maximum number of joints with extreme angles before termination

# ================== SIMULATION PARAMETERS ==================
DEFAULT_TIMESTEP = 32  # milliseconds
DEFAULT_MAX_STEPS = 1000
DEFAULT_MAX_EPISODES = 10000

# ================== COLLISION DETECTION ==================
# Self-collision detection parameters
COLLISION_DETECTION_ENABLED = True
MIN_COLLISION_DISTANCE = 0.05  # meters
COLLISION_PENALTY = -50.0

# ================== PLOTTING CONFIGURATION ==================
PLOT_WINDOW_SIZE = 10  # For moving averages
PLOT_DPI = 100
PLOT_FIGSIZE = (12, 10)

# ================== FILE PATHS ==================
# Relative path structure
CHECKPOINTS_DIR_NAME = "checkpoints"
PLOTS_DIR_NAME = "plots"
RESULTS_DIR_NAME = "results"
TEST_RESULTS_DIR_NAME = "test_results"

# ================== ALGORITHM DEFAULTS ==================
ALGORITHM_CONFIGS = {
    'ddpg': {
        'actor_lr': 1e-4,
        'critic_lr': 1e-3,
        'gamma': 0.99,
        'tau': 0.001,
        'buffer_size': 100000,
        'batch_size': 64,
        'exploration_noise': 0.1,
    },
    'ppo': {
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'gamma': 0.99,
        'clip_epsilon': 0.2,
        'gae_lambda': 0.95,
        'entropy_coeff': 0.01,
        'value_coeff': 0.5,
        'num_epochs': 10,
        'batch_size': 64,
    },
    'sac': {
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'alpha': 0.2,
        'buffer_size': 100000,
        'batch_size': 256,
    }
}

# ================== VALIDATION CONSTANTS ==================
# Joint groups for coordination analysis
ARM_JOINTS = ["ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR", "ArmLowerL"]
LEG_JOINTS = ["PelvYR", "PelvYL", "PelvR", "PelvL", "LegUpperR", "LegUpperL", "LegLowerR", "LegLowerL", "AnkleR", "AnkleL", "FootR", "FootL"]
NECK_JOINTS = ["Neck", "Head"]

# Symmetry tolerance (for checking left-right symmetry)
SYMMETRY_TOLERANCE = 0.1  # radians

# ================== DEBUG FLAGS ==================
DEBUG_MODE = False
PRINT_REWARD_DETAILS = False
LOG_JOINT_STATES = False
VISUALIZE_COM = False

# ================== EXPORT/IMPORT FORMATS ==================
CHECKPOINT_FORMAT = "pt"  # PyTorch format
RESULT_FORMAT = "json"
PLOT_FORMAT = "png"

# ================== PERFORMANCE OPTIMIZATION ==================
USE_VECTORIZED_OPS = True
PRECOMPUTE_GOAL_DISTANCES = True
CACHE_JOINT_LIMITS = True

# ================== VERSION INFORMATION ==================
CONFIG_VERSION = "1.0.0"
LAST_UPDATED = "2024-01-20"
AUTHOR = "OP3 RL Project"