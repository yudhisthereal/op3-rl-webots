#!/usr/bin/env python3
"""
CLI for OP3 Joint Angle Check Mode
Runs in your terminal, communicates with Webots via shared state file.
Usage: python angle_check_cli.py
"""

import os
import sys
import json
import time
from pathlib import Path
from shared_presets import PRESETS  # IMPORT FROM SHARED FILE

# Project paths
ANGLE_CHECK_DIR = Path(__file__).parent

# Shared state file for communication
STATE_FILE = ANGLE_CHECK_DIR / "angle_state.json"
COMMAND_FILE = ANGLE_CHECK_DIR / "angle_commands.json"

# Joint names (same as in angle_check.py)
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

def init_state_files():
    """Initialize the shared state files."""
    # Create angle_check directory if it doesn't exist
    ANGLE_CHECK_DIR.mkdir(exist_ok=True)
    
    # Initialize state file
    if not STATE_FILE.exists():
        initial_state = {
            "joints": {name: 0.0 for name in JOINT_NAMES},
            "connected": False,
            "last_update": time.time(),
            "status": "waiting"
        }
        save_state(initial_state)
    
    # Initialize command file
    if not COMMAND_FILE.exists():
        initial_commands = {
            "commands": [],
            "last_command_id": 0
        }
        with open(COMMAND_FILE, 'w') as f:
            json.dump(initial_commands, f, indent=2)


def save_state(state):
    """Save state to file."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def load_state():
    """Load state from file."""
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"joints": {}, "connected": False}


def add_command(command_type, **kwargs):
    """Add a command to be executed by Webots."""
    try:
        with open(COMMAND_FILE, 'r') as f:
            commands_data = json.load(f)
    except:
        commands_data = {"commands": [], "last_command_id": 0}
    
    command_id = commands_data["last_command_id"] + 1
    command = {
        "id": command_id,
        "type": command_type,
        "timestamp": time.time(),
        **kwargs
    }
    
    commands_data["commands"].append(command)
    commands_data["last_command_id"] = command_id
    
    with open(COMMAND_FILE, 'w') as f:
        json.dump(commands_data, f, indent=2)
    
    return command_id


def clear_commands():
    """Clear all commands from the command file."""
    commands_data = {"commands": [], "last_command_id": 0}
    with open(COMMAND_FILE, 'w') as f:
        json.dump(commands_data, f, indent=2)


def print_joint_list():
    """Print the list of joints with indices."""
    print("\n📋 Joint List (with indices):")
    print("-" * 40)
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {i:2d}. {name}")
    print("-" * 40)


def print_current_angles():
    """Print current joint angles from state file."""
    state = load_state()
    joints = state.get("joints", {})
    
    if not joints:
        print("\n❌ No joint data available. Is Webots running?")
        return
    
    print("\n📊 Current Joint Angles:")
    print("-" * 60)
    for i, name in enumerate(JOINT_NAMES):
        angle = joints.get(name, 0.0)
        angle_deg = angle * 180 / 3.14159
        print(f"  {i:2d}. {name:15s}: {angle:7.3f} rad ({angle_deg:6.1f}°)")
    print("-" * 60)


def set_joint_cli():
    """CLI for setting a joint angle."""
    print_joint_list()
    
    try:
        joint_idx = int(input("\nEnter joint index: "))
        if joint_idx < 0 or joint_idx >= len(JOINT_NAMES):
            print(f"❌ Invalid index. Must be 0-{len(JOINT_NAMES)-1}")
            return
        
        angle_str = input(f"Enter angle in radians for {JOINT_NAMES[joint_idx]}: ")
        angle = float(angle_str)
        
        command_id = add_command("set_joint", joint_index=joint_idx, angle=angle)
        print(f"✅ Command sent (ID: {command_id}). Joint will update shortly...")
        
    except ValueError:
        print("❌ Invalid input. Please enter a number.")
    except Exception as e:
        print(f"❌ Error: {e}")


def set_all_joints_cli():
    """CLI for setting all joints to same angle."""
    try:
        angle_str = input("\nEnter angle in radians for ALL joints: ")
        angle = float(angle_str)
        
        command_id = add_command("set_all", angle=angle)
        print(f"✅ Command sent (ID: {command_id}). All joints will update shortly...")
        
    except ValueError:
        print("❌ Invalid input. Please enter a number.")


def set_group_joints_cli():
    """CLI for setting a group of joints."""
    print_joint_list()
    
    try:
        indices_str = input("\nEnter joint indices (e.g., '0,1,2' or '0-5'): ")
        
        # Parse indices
        indices = []
        for part in indices_str.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(part))
        
        # Validate indices
        for idx in indices:
            if idx < 0 or idx >= len(JOINT_NAMES):
                print(f"❌ Invalid index {idx}. Must be 0-{len(JOINT_NAMES)-1}")
                return
        
        angle_str = input(f"Enter angle in radians for {len(indices)} joints: ")
        angle = float(angle_str)
        
        command_id = add_command("set_group", indices=indices, angle=angle)
        print(f"✅ Command sent (ID: {command_id}). Joints will update shortly...")
        
    except ValueError:
        print("❌ Invalid input. Please enter valid numbers.")
    except Exception as e:
        print(f"❌ Error: {e}")


def apply_preset_cli():
    """CLI for applying a preset pose."""
    print("\n💡 Available presets:")
    for preset_name in PRESETS.keys():
        print(f"  - {preset_name}")
    
    preset_name = input("\nEnter preset name: ").strip().lower()
    
    if preset_name in PRESETS:
        command_id = add_command("apply_preset", preset_name=preset_name)
        print(f"✅ Command sent (ID: {command_id}). Applying {preset_name} pose...")
    else:
        print(f"❌ Unknown preset: {preset_name}")


def reset_joints_cli():
    """CLI for resetting all joints to zero."""
    confirm = input("\n⚠️  Reset ALL joints to 0 rad? (yes/no): ").strip().lower()
    if confirm in ['yes', 'y']:
        command_id = add_command("reset_all")
        print(f"✅ Command sent (ID: {command_id}). Resetting joints...")
    else:
        print("❌ Reset cancelled.")


def get_connection_status():
    """Get current connection status as a string."""
    state = load_state()
    connected = state.get("connected", False)
    status = state.get("status", "unknown")
    last_update = state.get("last_update", 0)
    
    current_time = time.time()
    time_since_update = current_time - last_update
    
    if connected and time_since_update < 5:
        connection_status = "🟢 CONNECTED"
    elif not connected:
        connection_status = "🔴 DISCONNECTED"
    else:
        connection_status = "🟡 STALE"
    
    return connection_status, status, time_since_update


def print_header():
    """Print the header with connection status."""
    connection_status, webots_status, time_since_update = get_connection_status()
    
    print("\n" + "=" * 70)
    print("🤖 OP3 JOINT ANGLE CHECK - CLI MODE")
    print("=" * 70)
    print(f"Status: {connection_status}")
    print(f"Webots: {webots_status}")
    if connection_status == "🟢 CONNECTED":
        print(f"Last update: {time_since_update:.1f}s ago")
    print("=" * 70)


def main_menu():
    """Display the main menu and handle user input."""
    # Clear screen at start
    os.system('cls' if os.name == 'nt' else 'clear')
    
    while True:
        # Print header with current status
        print_header()
        
        print("\n📱 MAIN MENU")
        print("=" * 40)
        print("  1. List joints & current angles")
        print("  2. Set individual joint")
        print("  3. Set all joints")
        print("  4. Set group of joints")
        print("  5. Apply preset pose")
        print("  6. Reset all joints to zero")
        print("  7. Clear command queue")
        print("  8. Refresh status")
        print("  9. Exit")
        print("=" * 40)
        
        try:
            choice = input("\nEnter choice (1-9): ").strip()
            
            if choice == '1':
                print_current_angles()
                input("\nPress Enter to continue...")
            elif choice == '2':
                os.system('cls' if os.name == 'nt' else 'clear')
                set_joint_cli()
                input("\nPress Enter to continue...")
            elif choice == '3':
                os.system('cls' if os.name == 'nt' else 'clear')
                set_all_joints_cli()
                input("\nPress Enter to continue...")
            elif choice == '4':
                os.system('cls' if os.name == 'nt' else 'clear')
                set_group_joints_cli()
                input("\nPress Enter to continue...")
            elif choice == '5':
                os.system('cls' if os.name == 'nt' else 'clear')
                apply_preset_cli()
                input("\nPress Enter to continue...")
            elif choice == '6':
                os.system('cls' if os.name == 'nt' else 'clear')
                reset_joints_cli()
                input("\nPress Enter to continue...")
            elif choice == '7':
                clear_commands()
                print("✅ Command queue cleared.")
                input("\nPress Enter to continue...")
            elif choice == '8':
                # Just refresh by continuing loop
                continue
            elif choice == '9':
                print("\n👋 Exiting...")
                # Clear commands on exit
                clear_commands()
                sys.exit(0)
            else:
                print("❌ Invalid choice. Please enter 1-9.")
                time.sleep(1)  # Brief pause before redrawing
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Exiting...")
            clear_commands()
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")
            input("\nPress Enter to continue...")


def print_instructions():
    """Print startup instructions."""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n" + "=" * 70)
    print("🤖 OP3 JOINT ANGLE CHECK - CLI MODE")
    print("=" * 70)
    print("INSTRUCTIONS:")
    print("1. First, launch Webots with angle check mode:")
    print("   python main.py --angle_check")
    print("2. Then run this CLI in a separate terminal:")
    print("   python angle_check_cli.py")
    print("3. Use this CLI to control the robot in Webots")
    print("=" * 70)
    print("\n⚠️  Make sure Webots is running before using this CLI!")
    print("=" * 70)
    input("\nPress Enter to start...")


def main():
    """Main function."""
    # Initialize shared files
    init_state_files()
    
    # Print instructions
    print_instructions()
    
    # Start main menu
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        clear_commands()
        sys.exit(0)


if __name__ == "__main__":
    main()