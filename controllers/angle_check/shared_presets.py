# controllers/angle_check/shared_presets.py
"""
Shared presets for both angle_check.py and angle_check_cli.py
"""

PRESETS = {
    "standing": {
        "ShoulderR": 0.0, "ShoulderL": 0.0,
        "ArmUpperR": 0.0, "ArmUpperL": 0.0,
        "ArmLowerR": 0.0, "ArmLowerL": 0.0,
        "PelvYR": 0.0, "PelvYL": 0.0,
        "PelvR": 0.0, "PelvL": 0.0,
        "LegUpperR": 0.0, "LegUpperL": 0.0,
        "LegLowerR": 0.0, "LegLowerL": 0.0,
        "AnkleR": 0.0, "AnkleL": 0.0,
        "FootR": 0.0, "FootL": 0.0,
        "Neck": 0.0, "Head": 0.0,
    },
    "fall_forward": {
        "ShoulderR": 1.5, "ShoulderL": -1.5,
        "ArmUpperR": -1.25, "ArmUpperL": 1.25,
        "ArmLowerR": 0.3, "ArmLowerL": -0.3,
        "PelvR": 0.4, "PelvL": -0.4,
        "LegUpperR": 0.8, "LegUpperL": -0.8,
        "LegLowerR": -0.3, "LegLowerL": 0.3,
    },
    "fall_backward": {
        "ShoulderR": -0.6, "ShoulderL": 0.6,
        "ArmUpperR": 0.5, "ArmUpperL": 0.5,
        "PelvR": 0.3, "PelvL": 0.3,
        "LegUpperR": -0.3, "LegUpperL": -0.3,
    },
    "fall_right": {
        "ShoulderR": 0.0, "ShoulderL": 0.0,
        "ArmUpperR": -1.2, "ArmUpperL": 0.0,
        "PelvR": -0.6, "PelvL": 0.3,
        "LegUpperR": 0.5, "LegUpperL": 0.1,
    },
    "fall_left": {
        "ShoulderR": 0.0, "ShoulderL": 0.0,
        "ArmUpperR": 0.0, "ArmUpperL": -1.2,
        "PelvR": 0.3, "PelvL": -0.6,
        "LegUpperR": 0.1, "LegUpperL": 0.5,
    },
    "arms_up": {
        "ShoulderR": 1.57, "ShoulderL": -1.57,
        "ArmUpperR": 0.0, "ArmUpperL": 0.0,
    },
    "crouch": {
        "PelvR": -0.3, "PelvL": -0.3,
        "LegUpperR": 1.0, "LegUpperL": 1.0,
        "LegLowerR": -0.5, "LegLowerL": -0.5,
        "AnkleR": 0.3, "AnkleL": 0.3,
    }
}