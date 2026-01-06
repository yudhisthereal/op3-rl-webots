#!/usr/bin/env python3
"""
Simple CLI for OP3 Angle Monitor
Usage: python angle_monitor_cli.py
"""

import os
import sys
import json
import time
from pathlib import Path

# Shared state files
STATE_FILE = Path("angle_state.json")
PRESETS_FILE = Path("angle_presets.json")

# Joint names
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

def load_state():
    """Load current state from Webots."""
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"joints": {}, "connected": False, "status": "disconnected"}

def load_presets():
    """Load presets from file."""
    if PRESETS_FILE.exists():
        try:
            with open(PRESETS_FILE, 'r') as f:
                presets = json.load(f)
                # Remove internal keys
                presets.pop("_current", None)
                return presets
        except:
            return {}
    return {}

def save_presets(presets):
    """Save presets to file."""
    with open(PRESETS_FILE, 'w') as f:
        json.dump(presets, f, indent=2)

def set_current_preset(preset_name, presets):
    """Set the current preset in the presets file."""
    if preset_name in presets:
        presets_with_current = presets.copy()
        presets_with_current["current_preset"] = preset_name
        save_presets(presets_with_current)
        return True
    return False

def print_joint_angles():
    """Print current joint angles."""
    state = load_state()
    joints = state.get("joints", {})
    
    print("\n" + "=" * 60)
    print("🤖 CURRENT JOINT ANGLES")
    print("=" * 60)
    
    if not joints:
        print("❌ No data - Is Webots running?")
        return
    
    print(f"{'Joint':<15} {'Radians':<10} {'Degrees':<10}")
    print("-" * 60)
    
    for name in JOINT_NAMES:
        angle_rad = joints.get(name, 0.0)
        angle_deg = angle_rad * 180 / 3.14159
        print(f"{name:<15} {angle_rad:<10.3f} {angle_deg:<10.1f}")
    
    print("=" * 60)
    
    # Show connection status
    connected = state.get("connected", False)
    status = state.get("status", "unknown")
    if connected:
        print(f"Status: 🟢 Connected ({status})")
    else:
        print(f"Status: 🔴 Disconnected")

def save_current_as_preset():
    """Save current joint angles as a new preset."""
    state = load_state()
    joints = state.get("joints", {})
    
    if not joints:
        print("❌ Cannot save - No joint data available")
        return
    
    # Get preset name
    preset_name = input("\nEnter name for new preset: ").strip()
    if not preset_name:
        print("❌ Preset name cannot be empty")
        return
    
    # Load existing presets
    presets = load_presets()
    
    # Check if name exists
    if preset_name in presets:
        overwrite = input(f"Preset '{preset_name}' already exists. Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("❌ Save cancelled")
            return
    
    # Create preset
    new_preset = {}
    for name in JOINT_NAMES:
        new_preset[name] = joints.get(name, 0.0)
    
    # Save to presets
    presets[preset_name] = new_preset
    save_presets(presets)
    
    print(f"✅ Saved as preset: '{preset_name}'")

def choose_preset():
    """Choose and apply a preset."""
    presets = load_presets()
    
    if not presets:
        print("❌ No presets available")
        print("   First save some joint angles as a preset")
        return
    
    print("\n💾 Available Presets:")
    print("-" * 40)
    for i, name in enumerate(presets.keys(), 1):
        print(f"  {i}. {name}")
    print("-" * 40)
    
    try:
        choice = input("\nEnter preset number or name: ").strip()
        
        # Try to parse as number
        if choice.isdigit():
            choice_num = int(choice)
            preset_names = list(presets.keys())
            if 1 <= choice_num <= len(preset_names):
                preset_name = preset_names[choice_num - 1]
            else:
                print("❌ Invalid number")
                return
        else:
            # Use as name
            preset_name = choice
        
        # Apply the preset
        if preset_name in presets:
            if set_current_preset(preset_name, presets):
                print(f"✅ Applying preset: '{preset_name}'")
                print("   Check Webots window to see the change...")
            else:
                print("❌ Failed to apply preset")
        else:
            print(f"❌ Preset '{preset_name}' not found")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def print_menu():
    """Print the main menu."""
    print("\n" + "=" * 40)
    print("🤖 OP3 ANGLE MONITOR - CLI")
    print("=" * 40)
    print("  1. Print joint angles")
    print("  2. Save current as preset")
    print("  3. Choose preset")
    print("  4. Refresh")
    print("  5. Exit")
    print("=" * 40)

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def check_webots_connection():
    """Check if Webots is connected."""
    state = load_state()
    connected = state.get("connected", False)
    last_update = state.get("last_update", 0)
    
    if not connected:
        print("\n⚠️  WARNING: Webots not connected")
        print("   Make sure Webots is running with angle_monitor controller")
        return False
    
    # Check if data is stale (older than 10 seconds)
    time_since_update = time.time() - last_update
    if time_since_update > 10:
        print(f"\n⚠️  WARNING: Last update was {time_since_update:.1f} seconds ago")
        print("   Webots might be paused or not responding")
    
    return True

def main():
    """Main CLI loop."""
    clear_screen()
    
    print("\n" + "=" * 50)
    print("🤖 OP3 ANGLE MONITOR CLI")
    print("=" * 50)
    print("Instructions:")
    print("1. First, launch Webots with the angle_monitor controller")
    print("2. Then run this CLI to monitor and control joint angles")
    print("3. Default preset 'All Zero' will be loaded automatically")
    print("=" * 50)
    
    # Initial connection check
    if not check_webots_connection():
        print("\nPress Enter to continue anyway...")
        input()
    
    while True:
        clear_screen()
        print_menu()
        
        try:
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                print_joint_angles()
                input("\nPress Enter to continue...")
            elif choice == '2':
                save_current_as_preset()
                input("\nPress Enter to continue...")
            elif choice == '3':
                choose_preset()
                input("\nPress Enter to continue...")
            elif choice == '4':
                # Just refresh
                continue
            elif choice == '5':
                print("\n👋 Exiting...")
                sys.exit(0)
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()