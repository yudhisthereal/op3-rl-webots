# controllers/angle_check/angle_check.py
#!/usr/bin/env python3
"""
Angle Check Controller for OP3 Robot
Reads commands from CLI via shared JSON file.
"""

from controller import Supervisor
import json
import time
from pathlib import Path
from shared_presets import PRESETS  # IMPORT FROM SHARED FILE

def main():
    robot = Supervisor()
    timestep = 32  # ms
    
    # Path to shared state files
    script_dir = Path(__file__).parent
    STATE_FILE = script_dir / "angle_state.json"
    COMMAND_FILE = script_dir / "angle_commands.json"
    
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
            motor.setPosition(0.0)
            motors[name] = motor
            sensors[name] = sensor
        except:
            print(f"Warning: Could not find motor {name}")
    
    # Initialize state file
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
    
    # Initialize files
    with open(STATE_FILE, 'w') as f:
        json.dump({"joints": {}, "connected": True, "status": "starting"}, f)
    
    with open(COMMAND_FILE, 'w') as f:
        json.dump({"commands": [], "last_command_id": 0}, f)
    
    print("🤖 OP3 Angle Check Controller Started")
    print("📁 Reading commands from:", COMMAND_FILE)
    print("📁 Writing state to:", STATE_FILE)
    print("🟢 Ready to receive commands from CLI...")
    
    # Main loop
    last_command_id = 0
    
    while robot.step(timestep) != -1:
        # Update state file
        update_state()
        
        # Check for new commands
        try:
            with open(COMMAND_FILE, 'r') as f:
                commands_data = json.load(f)
            
            # Process new commands
            for command in commands_data.get("commands", []):
                cmd_id = command.get("id", 0)
                if cmd_id > last_command_id:
                    last_command_id = cmd_id
                    cmd_type = command.get("type", "")
                    
                    if cmd_type == "set_joint":
                        idx = command.get("joint_index")
                        angle = command.get("angle", 0.0)
                        if 0 <= idx < len(JOINT_NAMES):
                            name = JOINT_NAMES[idx]
                            if name in motors:
                                motors[name].setPosition(angle)
                                print(f"Executed: Set {name} to {angle:.3f} rad")
                    
                    elif cmd_type == "set_all":
                        angle = command.get("angle", 0.0)
                        for name, motor in motors.items():
                            motor.setPosition(angle)
                        print(f"Executed: Set ALL joints to {angle:.3f} rad")
                    
                    elif cmd_type == "set_group":
                        indices = command.get("indices", [])
                        angle = command.get("angle", 0.0)
                        for idx in indices:
                            if 0 <= idx < len(JOINT_NAMES):
                                name = JOINT_NAMES[idx]
                                if name in motors:
                                    motors[name].setPosition(angle)
                        print(f"Executed: Set {len(indices)} joints to {angle:.3f} rad")
                    
                    elif cmd_type == "apply_preset":
                        preset_name = command.get("preset_name", "")
                        if preset_name in PRESETS:
                            preset = PRESETS[preset_name]
                            for name, angle in preset.items():
                                if name in motors:
                                    motors[name].setPosition(angle)
                            print(f"Executed: Applied preset '{preset_name}'")
                        else:
                            print(f"Error: Unknown preset '{preset_name}'")
                    
                    elif cmd_type == "reset_all":
                        for name, motor in motors.items():
                            motor.setPosition(0.0)
                        print("Executed: Reset all joints to 0 rad")
            
            # Clear processed commands
            commands_data["commands"] = [
                cmd for cmd in commands_data.get("commands", [])
                if cmd.get("id", 0) > last_command_id
            ]
            
            with open(COMMAND_FILE, 'w') as f:
                json.dump(commands_data, f, indent=2)
                
        except Exception as e:
            print(f"Error processing commands: {e}")
        
        # Small delay to prevent CPU overload
        time.sleep(0.01)

if __name__ == "__main__":
    main()