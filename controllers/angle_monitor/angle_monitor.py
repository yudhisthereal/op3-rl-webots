#!/usr/bin/env python3
"""
Angle Monitor Controller for OP3 Robot
Monitors joint angles and applies presets from CLI.
"""

from controller import Supervisor
import json
import time
from pathlib import Path

def main():
    robot = Supervisor()
    timestep = 32  # ms
    
    # Path to shared state files
    script_dir = Path(__file__).parent
    STATE_FILE = script_dir / "angle_state.json"
    PRESETS_FILE = script_dir / "angle_presets.json"
    
    # Define all OP3 joint names
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
    
    # Get all motors and sensors
    motors = {}
    sensors = {}
    for name in JOINT_NAMES:
        try:
            motor = robot.getDevice(name)
            sensor = motor.getPositionSensor()
            sensor.enable(timestep)
            motor.setPosition(0.0)  # Start at zero position
            motors[name] = motor
            sensors[name] = sensor
        except:
            print(f"Warning: Could not find motor {name}")
    
    # Initialize or load presets
    def load_presets():
        """Load presets from file, template, or create default."""
        if PRESETS_FILE.exists():
            try:
                with open(PRESETS_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Check if template exists
        TEMPLATE_FILE = script_dir / "angle_presets.json.template"
        if TEMPLATE_FILE.exists():
            try:
                print(f"Using template to create presets: {TEMPLATE_FILE}")
                with open(TEMPLATE_FILE, 'r') as f:
                    presets = json.load(f)
                save_presets(presets)
                return presets
            except:
                pass
        
        # Default presets
        default_presets = {
            "All Zero": {name: 0.0 for name in JOINT_NAMES},
            "Standing": {
                "ShoulderR": 0.0, "ShoulderL": 0.0,
                "ArmUpperR": 0.0, "ArmUpperL": 0.0,
                "ArmLowerR": -1.0, "ArmLowerL": 1.0,
                "PelvYR": 0.0, "PelvYL": 0.0,
                "PelvR": 0.0, "PelvL": 0.0,
                "LegUpperR": 0.0, "LegUpperL": 0.0,
                "LegLowerR": 0.0, "LegLowerL": 0.0,
                "AnkleR": 0.0, "AnkleL": 0.0,
                "FootR": 0.0, "FootL": 0.0,
                "Neck": 0.0, "Head": 0.0
            },
            "Walking": {
                "ShoulderR": 0.0, "ShoulderL": 0.0,
                "ArmUpperR": 0.2, "ArmUpperL": -0.2,
                "ArmLowerR": -0.5, "ArmLowerL": 0.5,
                "PelvYR": 0.0, "PelvYL": 0.0,
                "PelvR": 0.0, "PelvL": 0.0,
                "LegUpperR": 0.3, "LegUpperL": -0.3,
                "LegLowerR": -0.6, "LegLowerL": 0.6,
                "AnkleR": 0.2, "AnkleL": -0.2,
                "FootR": 0.0, "FootL": 0.0,
                "Neck": 0.0, "Head": 0.0
            }
        }
        
        save_presets(default_presets)
        return default_presets
    
    def save_presets(presets):
        """Save presets to file."""
        with open(PRESETS_FILE, 'w') as f:
            json.dump(presets, f, indent=2)
    
    # Initialize state and presets
    def update_state():
        """Update state file with current joint angles."""
        state = {
            "joints": {},
            "connected": True,
            "last_update": time.time(),
            "status": "running"
        }
        
        for name, sensor in sensors.items():
            try:
                state["joints"][name] = sensor.getValue()
            except:
                state["joints"][name] = 0.0
        
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    
    def apply_preset(preset_name, presets_dict):
        """Apply a preset pose to the robot."""
        if preset_name in presets_dict:
            preset = presets_dict[preset_name]
            for name, angle in preset.items():
                if name in motors:
                    motors[name].setPosition(angle)
            print(f"Applied preset: '{preset_name}'")
            return True
        else:
            print(f"Error: Unknown preset '{preset_name}'")
            return False
    
    # Initialize state file
    with open(STATE_FILE, 'w') as f:
        json.dump({
            "joints": {name: 0.0 for name in JOINT_NAMES},
            "connected": True,
            "last_update": time.time(),
            "status": "starting"
        }, f)
    
    # Load presets
    presets = load_presets()
    
    print("🤖 OP3 Angle Monitor Controller Started")
    print("📁 Writing state to:", STATE_FILE)
    print("📁 Presets file:", PRESETS_FILE)
    print(f"📋 Loaded {len(presets)} presets:", ", ".join(presets.keys()))
    print("🟢 Ready...")
    
    # Main loop
    while robot.step(timestep) != -1:
        # Update state file
        update_state()
        
        # Check for preset changes
        if PRESETS_FILE.exists():
            try:
                with open(PRESETS_FILE, 'r') as f:
                    new_presets = json.load(f)
                
                # Check if current preset changed
                if "current_preset" in new_presets:
                    current_preset = new_presets["current_preset"]
                    if current_preset != presets.get("_current", ""):
                        if apply_preset(current_preset, new_presets):
                            presets["_current"] = current_preset
                
                # Update presets dictionary
                presets = new_presets
                
            except Exception as e:
                print(f"Error reading presets: {e}")
        
        # Small delay to prevent CPU overload
        time.sleep(0.01)

if __name__ == "__main__":
    main()