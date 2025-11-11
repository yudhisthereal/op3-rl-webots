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

# Import genetic config for population size display
try:
    sys.path.insert(0, str(Path(__file__).parent / "controllers" / "op3_ddpg_env"))
    import genetic_config
except ImportError:
    genetic_config = None

# Project paths
PROJECT_ROOT = Path(__file__).parent
CONTROLLERS_DIR = PROJECT_ROOT / "controllers"
TRAIN_CONTROLLER = CONTROLLERS_DIR / "op3_ddpg_env" / "op3_ddpg_env.py"
GENETIC_CONTROLLER = CONTROLLERS_DIR / "op3_ddpg_env" / "op3_ddpg_genetic.py"
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
    },
    "fall_control": {
        "import": "from scenarios.fall_control import FallControl",
        "class": "FallControl"
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
        # NEVER comment out lines containing scenario_class_name or scenario_name assignments from SCENARIO_CLASS
        if 'scenario_class_name' in line or ('scenario_name' in line and 'SCENARIO_CLASS.scenario_name' in line):
            updated_lines.append(line)
            continue
        # Handle scenario import lines (match any scenario import, not just arm_control)
        if 'from scenarios.' in line and 'import' in line:
            # Check if this line matches our target scenario
            stripped_line = line.strip()
            is_commented = stripped_line.startswith('#')
            line_content = stripped_line.lstrip('#').strip()
            
            # Preserve original indentation
            indent = len(line) - len(line.lstrip())
            
            # Check if this line matches our target scenario import
            # Match by checking if the class name is in the import line
            target_class = scenario["class"]
            if target_class in line_content:
                # This is our scenario - uncomment it
                updated_lines.append(' ' * indent + scenario["import"] + '\n')
            else:
                # This is another scenario - comment it (if not already commented)
                if not is_commented:
                    updated_lines.append(' ' * indent + '# ' + line_content + '\n')
                else:
                    updated_lines.append(line)
        # Handle SCENARIO_CLASS assignment
        elif 'SCENARIO_CLASS' in line and '=' in line and 'scenario =' not in line:
            # Check if this line matches our target scenario class
            stripped_line = line.strip()
            is_commented = stripped_line.startswith('#')
            line_content = stripped_line.lstrip('#').strip()
            
            # Preserve original indentation
            indent = len(line) - len(line.lstrip())
            
            target_class = scenario["class"]
            
            # Check if this line assigns our target class
            # Pattern: "SCENARIO_CLASS = ClassName" or "SCENARIO_CLASS=ClassName"
            # Extract the class name from the assignment
            if '=' in line_content:
                assigned_class = line_content.split('=')[-1].strip()
                is_target_class = assigned_class == target_class
            else:
                is_target_class = False
            
            if is_target_class:
                # This is our scenario - ensure it's uncommented and correct
                if is_commented:
                    # Uncomment it
                    updated_lines.append(' ' * indent + f'SCENARIO_CLASS = {target_class}\n')
                else:
                    # Already uncommented and correct - keep it as-is
                    updated_lines.append(line)
            else:
                # This is another scenario - comment it (if not already commented)
                if not is_commented:
                    updated_lines.append(' ' * indent + '# ' + line_content + '\n')
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


def update_checkpoint_in_file(file_path, checkpoint_path):
    """Update checkpoint path in test_policy.py.
    
    Args:
        checkpoint_path: Full checkpoint path (e.g., "yudhis/ddpg_model.pt" or "ddpg_model.pt")
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if checkpoint_path contains a subdirectory
    has_subdir = '/' in checkpoint_path or '\\' in checkpoint_path
    
    if has_subdir:
        # Full path with subdirectory - replace entire checkpoint_path assignment
        # Split path and build os.path.join expression
        path_parts = checkpoint_path.replace('\\', '/').split('/')
        
        # Build the path parts for os.path.join
        path_args = ["os.path.dirname(__file__)", "'..'", "'op3_ddpg_env'", "'checkpoints'"]
        path_args.extend([f"'{part}'" for part in path_parts])
        path_join_expr = "os.path.join(" + ", ".join(path_args) + ")"
        
        # Replace the entire checkpoint_path assignment (handles multi-line)
        # Match both single-line and multi-line patterns
        multiline_pattern = r'checkpoint_path\s*=\s*os\.path\.join\([^)]*\n[^)]*\)'
        singleline_pattern = r'checkpoint_path\s*=\s*os\.path\.join\([^)]+\)'
        
        replacement = f'checkpoint_path = {path_join_expr}'
        
        if re.search(multiline_pattern, content, re.MULTILINE):
            content = re.sub(multiline_pattern, replacement, content, flags=re.MULTILINE)
        elif re.search(singleline_pattern, content):
            content = re.sub(singleline_pattern, replacement, content)
        else:
            # Fallback: try to find any checkpoint_path assignment
            pattern = r'checkpoint_path\s*=\s*[^\n]+'
            content = re.sub(pattern, replacement, content)
    else:
        # Just filename - replace config.CHECKPOINT_NAME or hardcoded "ddpg_model.pt"
        # Handle multi-line checkpoint_path assignment (test_policy.py uses multi-line)
        
        # Pattern for multi-line: matches across newlines (test_policy.py format)
        # Matches: checkpoint_path = os.path.join(..., \n ... config.CHECKPOINT_DIR, "ddpg_model.pt")
        multiline_pattern = r'(checkpoint_path\s*=\s*os\.path\.join\([^)]*\n\s*[^)]*,\s*)(?:config\.CHECKPOINT_NAME|"ddpg_model\.pt")'
        replacement_multiline = rf'\1"{checkpoint_path}"'
        
        # Pattern for single-line
        singleline_pattern = r'(checkpoint_path\s*=\s*os\.path\.join\([^)]+,\s*)(?:config\.CHECKPOINT_NAME|"ddpg_model\.pt")'
        replacement_singleline = rf'\1"{checkpoint_path}"'
        
        # Try multi-line first (most common case in test_policy.py)
        if re.search(multiline_pattern, content, re.MULTILINE | re.DOTALL):
            content = re.sub(multiline_pattern, replacement_multiline, content, flags=re.MULTILINE | re.DOTALL)
        elif re.search(singleline_pattern, content):
            content = re.sub(singleline_pattern, replacement_singleline, content)
        else:
            # Fallback: replace just the filename part anywhere it appears
            # This handles the case where config.CHECKPOINT_DIR is on a different line
            content = re.sub(
                r'"ddpg_model\.pt"',
                f'"{checkpoint_path}"',
                content
            )
            # Also try config.CHECKPOINT_NAME
            content = re.sub(
                r'config\.CHECKPOINT_NAME',
                f'"{checkpoint_path}"',
                content
            )
    
    with open(file_path, 'w') as f:
        f.write(content)


def launch_webots(world_file, mode="fast", env=None):
    """Launch Webots with the specified world file.
    
    Args:
        world_file: Path to world file
        mode: Webots mode (fast, run, etc.)
        env: Optional environment variables dict
    """
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
        subprocess.run(cmd, env=env, check=True)
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
  python main.py --genetic --scenario=pak_gembong
  python main.py --genetic --scenario=yudhis
  python main.py --test --scenario=yudhis
  python main.py --test --scenario=yudhis --checkpoint=ddpg_model.pt
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Run training mode')
    mode_group.add_argument('--genetic', action='store_true', help='Run genetic algorithm multi-agent training')
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
            
        elif args.genetic:
            print(f"📝 Setting up parallel genetic training with scenario: {args.scenario}")
            
            # Update genetic parallel controller scenario
            genetic_parallel_controller = CONTROLLERS_DIR / "op3_ddpg_env" / "op3_ddpg_genetic_parallel.py"
            update_scenario_in_file(genetic_parallel_controller, args.scenario)
            print(f"✅ Updated {genetic_parallel_controller}")
            
            # Run parallel genetic training script
            parallel_script = CONTROLLERS_DIR / "op3_ddpg_env" / "run_genetic_parallel.py"
            scenario_class = SCENARIOS[args.scenario]["class"]
            
            print(f"🚀 Launching parallel genetic training...")
            if genetic_config:
                print(f"   This will launch {genetic_config.POPULATION_SIZE} Webots instances in parallel")
            else:
                print(f"   This will launch multiple Webots instances in parallel")
            
            import subprocess
            result = subprocess.run(
                [sys.executable, str(parallel_script), scenario_class],
                cwd=str(CONTROLLERS_DIR / "op3_ddpg_env")
            )
            
            if result.returncode != 0:
                print(f"❌ Genetic training failed with exit code {result.returncode}")
                sys.exit(1)
            
        elif args.test:
            print(f"📝 Setting up testing with scenario: {args.scenario}")
            update_scenario_in_file(TEST_CONTROLLER, args.scenario)
            print(f"✅ Updated {TEST_CONTROLLER}")
            
            # Set checkpoint path via environment variable
            env = os.environ.copy()
            if args.checkpoint:
                env['CHECKPOINT_PATH'] = args.checkpoint
                print(f"📦 Using checkpoint: {args.checkpoint}")
            else:
                # Use scenario-specific default
                try:
                    sys.path.insert(0, str(CONTROLLERS_DIR / "op3_ddpg_env"))
                    import config as train_config
                    scenario_checkpoint = f"{args.scenario}/{train_config.CHECKPOINT_NAME}"
                    env['CHECKPOINT_PATH'] = scenario_checkpoint
                    print(f"📦 Using default checkpoint for scenario: {scenario_checkpoint}")
                except:
                    pass
            
            print(f"🎮 Launching test world...")
            launch_webots(TEST_WORLD, mode="fast", env=env)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

