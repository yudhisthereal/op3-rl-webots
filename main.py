#!/usr/bin/env python3
"""
Main script to launch Webots training or testing with different scenarios.

Usage:
    python main.py --train --scenario=pak_gembong
    python main.py --train --scenario=yudhis
    python main.py --test --scenario=yudhis
    python main.py --test --scenario=yudhis --checkpoint=ddpg_model.pt
"""

import argparse
import subprocess
import sys
import os
import re
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
CONTROLLERS_DIR = PROJECT_ROOT / "controllers"
TRAIN_CONTROLLER = CONTROLLERS_DIR / "op3_ddpg_env" / "op3_ddpg_env.py"
TEST_CONTROLLER = CONTROLLERS_DIR / "test_policy" / "test_policy.py"
TRAIN_WORLD = PROJECT_ROOT / "worlds" / "robotis_op3_train.wbt"
TEST_WORLD = PROJECT_ROOT / "worlds" / "robotis_op3_test.wbt"

# Scenario mappings
SCENARIOS = {
    "pak_gembong": {
        "import": "from scenarios.arm_control_pak_gembong import ArmControlPakGembong",
        "class": "ArmControlPakGembong"
    },
    "yudhis": {
        "import": "from scenarios.arm_control_yudhis import ArmControlYudhis",
        "class": "ArmControlYudhis"
    }
}


def find_webots():
    """Find Webots executable."""
    # Check WEBOTS_HOME environment variable first
    webots_home = os.environ.get("WEBOTS_HOME")
    if webots_home:
        webots_path = os.path.join(webots_home, "webots")
        if os.path.exists(webots_path):
            return webots_path
    
    # Common Webots installation paths
    possible_paths = [
        "/usr/local/webots/webots",  # Linux default
        "/opt/webots/webots",  # Alternative Linux
        os.path.expanduser("~/webots/webots"),  # User installation
    ]
    
    # Check if webots is in PATH
    import shutil
    webots_path = shutil.which("webots")
    if webots_path:
        return webots_path
    
    # Check common installation paths
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def update_scenario_in_file(file_path, scenario_name):
    """Update scenario selection in a controller file."""
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(SCENARIOS.keys())}")
    
    scenario = SCENARIOS[scenario_name]
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Update lines
    updated_lines = []
    scenario_init_found = False
    for i, line in enumerate(lines):
        # Handle scenario import lines
        if 'from scenarios.arm_control' in line:
            # Comment out all, then uncomment the selected one
            if scenario["import"] in line:
                # This is our scenario - uncomment it
                updated_lines.append(scenario["import"] + '\n')
            else:
                # This is another scenario - comment it
                if not line.strip().startswith('#'):
                    updated_lines.append('# ' + line)
                else:
                    updated_lines.append(line)
        # Handle SCENARIO_CLASS assignment
        elif 'SCENARIO_CLASS' in line and '=' in line and 'scenario =' not in line:
            if scenario["class"] in line:
                # This is our scenario - uncomment and ensure correct
                updated_lines.append(f'SCENARIO_CLASS = {scenario["class"]}\n')
            else:
                # This is another scenario - comment it
                if not line.strip().startswith('#'):
                    updated_lines.append('# ' + line)
                else:
                    updated_lines.append(line)
        # Handle scenario initialization line (scenario = SCENARIO_CLASS(...))
        elif 'scenario =' in line and 'SCENARIO_CLASS' in line:
            # Ensure this line is uncommented and correct
            stripped = line.strip()
            if stripped.startswith('#'):
                # Uncomment it - preserve original indentation
                indent = len(line) - len(line.lstrip())
                # Create the correct line
                updated_lines.append(' ' * indent + f'scenario = SCENARIO_CLASS(robot, timestep=config.TIMESTEP)\n')
            else:
                # Already uncommented, keep it
                updated_lines.append(line)
            scenario_init_found = True
        else:
            updated_lines.append(line)
    
    # If scenario initialization line was not found, add it after robot initialization
    if not scenario_init_found:
        for i, line in enumerate(updated_lines):
            if 'robot = Supervisor()' in line:
                # Insert scenario initialization after robot line
                indent = len(line) - len(line.lstrip())
                scenario_line = ' ' * indent + f'scenario = SCENARIO_CLASS(robot, timestep=config.TIMESTEP)\n'
                updated_lines.insert(i + 1, scenario_line)
                break
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(updated_lines)


def update_checkpoint_in_file(file_path, checkpoint_name):
    """Update checkpoint name in test_policy.py."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find and update the checkpoint_path assignment (may span multiple lines)
    updated_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'checkpoint_path' in line and 'os.path.join' in line:
            # This is the start of checkpoint_path assignment
            # Check if it continues on next line
            if 'config.CHECKPOINT_NAME' in line:
                # Single line case
                line = re.sub(
                    r'config\.CHECKPOINT_NAME',
                    f'"{checkpoint_name}"',
                    line
                )
                updated_lines.append(line)
            else:
                # Multi-line case - add current line and check next
                updated_lines.append(line)
                i += 1
                if i < len(lines) and 'config.CHECKPOINT_NAME' in lines[i]:
                    # Replace on next line
                    next_line = re.sub(
                        r'config\.CHECKPOINT_NAME',
                        f'"{checkpoint_name}"',
                        lines[i]
                    )
                    updated_lines.append(next_line)
                else:
                    # Not found, keep original
                    if i < len(lines):
                        updated_lines.append(lines[i])
        else:
            updated_lines.append(line)
        i += 1
    
    with open(file_path, 'w') as f:
        f.writelines(updated_lines)


def launch_webots(world_file, mode="fast"):
    """Launch Webots with the specified world file."""
    webots_path = find_webots()
    
    if not webots_path:
        print("❌ Error: Webots executable not found!")
        print("Please install Webots or add it to your PATH.")
        print("You can also set WEBOTS_HOME environment variable.")
        sys.exit(1)
    
    if not world_file.exists():
        print(f"❌ Error: World file not found: {world_file}")
        sys.exit(1)
    
    # Launch Webots
    cmd = [webots_path, f"--mode={mode}", str(world_file.absolute())]
    print(f"🚀 Launching Webots: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Webots exited with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Launch Webots training or testing with different scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train --scenario=pak_gembong
  python main.py --train --scenario=yudhis
  python main.py --test --scenario=yudhis
  python main.py --test --scenario=yudhis --checkpoint=ddpg_model.pt
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Run training mode')
    mode_group.add_argument('--test', action='store_true', help='Run testing mode')
    
    # Scenario selection
    parser.add_argument(
        '--scenario',
        type=str,
        required=True,
        choices=list(SCENARIOS.keys()),
        help='Scenario to use (pak_gembong or yudhis)'
    )
    
    # Checkpoint (for test mode)
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint file name (default: uses config.CHECKPOINT_NAME)'
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint for test mode
    if args.test and args.checkpoint:
        checkpoint_path = CONTROLLERS_DIR / "op3_ddpg_env" / "checkpoints" / args.checkpoint
        if not checkpoint_path.exists():
            print(f"⚠️  Warning: Checkpoint file not found: {checkpoint_path}")
            print("Continuing anyway...")
    
    try:
        if args.train:
            print(f"📝 Setting up training with scenario: {args.scenario}")
            update_scenario_in_file(TRAIN_CONTROLLER, args.scenario)
            print(f"✅ Updated {TRAIN_CONTROLLER}")
            print(f"🎮 Launching training world...")
            launch_webots(TRAIN_WORLD)
            
        elif args.test:
            print(f"📝 Setting up testing with scenario: {args.scenario}")
            update_scenario_in_file(TEST_CONTROLLER, args.scenario)
            print(f"✅ Updated {TEST_CONTROLLER}")
            
            if args.checkpoint:
                print(f"📦 Setting checkpoint to: {args.checkpoint}")
                update_checkpoint_in_file(TEST_CONTROLLER, args.checkpoint)
                print(f"✅ Updated checkpoint in {TEST_CONTROLLER}")
            
            print(f"🎮 Launching test world...")
            launch_webots(TEST_WORLD)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

