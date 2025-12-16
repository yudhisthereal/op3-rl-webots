#!/usr/bin/env python3
"""
Main script to launch Webots training or testing with different scenarios and algorithms.

Usage:
    python main.py --train --scenario=fall_control --alg=ddpg
    python main.py --train --scenario=fall_control --alg=ppo
    python main.py --train --scenario=fall_control --alg=sac
    python main.py --test --scenario=fall_control --alg=ppo
    python main.py --test --scenario=fall_control --alg=sac --checkpoint=sac_final.pt
    python main.py --angle_check
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

# Algorithm configurations
ALGORITHMS = {
    "ddpg": {
        "controller_dir": "op3_ddpg_env",
        "train_controller": "op3_ddpg_env.py",
        "genetic_controller": "op3_ddpg_genetic.py",
        "genetic_parallel_controller": "op3_ddpg_genetic_parallel.py",
        "parallel_script": "run_genetic_parallel.py",
        "train_world": "robotis_op3_train.wbt",
        "test_controller_dir": "test_policy",
        "test_controller": "test_policy.py",
        "test_world": "robotis_op3_test.wbt",
        "config_module": "config"
    },
    "ppo": {
        "controller_dir": "op3_ppo_env",
        "train_controller": "op3_ppo_env.py",
        "train_world": "robotis_op3_train_ppo.wbt",
        "test_controller_dir": "test_ppo_env",
        "test_controller": "test_ppo_env.py",
        "test_world": "robotis_op3_test_ppo.wbt",
        "config_module": "config"
    },
    "sac": {
        "controller_dir": "op3_sac_env",
        "train_controller": "op3_sac_env.py",
        "train_world": "robotis_op3_train_sac.wbt",
        "test_controller_dir": "test_ppo_env",
        "test_controller": "test_ppo_env.py",
        "test_world": "robotis_op3_test_sac.wbt",
        "config_module": "config"
    }
}

# Angle check mode
ANGLE_CHECK_DIR = CONTROLLERS_DIR / "angle_check"
ANGLE_CHECK_CONTROLLER = ANGLE_CHECK_DIR / "angle_check.py"
ANGLE_CHECK_WORLD = PROJECT_ROOT / "worlds" / "robotis_op3_angle_check.wbt"

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
    webots_home = os.environ.get("WEBOTS_HOME")
    if webots_home:
        webots_path = os.path.join(webots_home, "webots")
        if os.path.exists(webots_path):
            return webots_path
    
    possible_paths = [
        "/usr/local/webots/webots",
        "/opt/webots/webots",
        os.path.expanduser("~/webots/webots"),
    ]
    
    import shutil
    webots_path = shutil.which("webots")
    if webots_path:
        return webots_path
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def update_scenario_in_file(file_path, scenario_name, algorithm="ddpg"):
    """Update scenario selection in a controller file."""
    if scenario_name not in SCENARIOS:
        # Try to find fall control variant for the algorithm
        fall_variant = f"fall_control_{algorithm}"
        if fall_variant in SCENARIOS:
            scenario_name = fall_variant
        elif scenario_name == "fall_control":
            # Use base fall control if variant not found
            pass
        else:
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
        # Handle scenario import lines
        if 'from scenarios.' in line and 'import' in line:
            stripped_line = line.strip()
            is_commented = stripped_line.startswith('#')
            line_content = stripped_line.lstrip('#').strip()
            
            indent = len(line) - len(line.lstrip())
            
            target_class = scenario["class"]
            if target_class in line_content:
                # This is our scenario - uncomment it
                updated_lines.append(' ' * indent + scenario["import"] + '\n')
            else:
                # This is another scenario - comment it
                if not is_commented:
                    updated_lines.append(' ' * indent + '# ' + line_content + '\n')
                else:
                    updated_lines.append(line)
        # Handle SCENARIO_CLASS assignment
        elif 'SCENARIO_CLASS' in line and '=' in line and 'scenario =' not in line:
            stripped_line = line.strip()
            is_commented = stripped_line.startswith('#')
            line_content = stripped_line.lstrip('#').strip()
            
            indent = len(line) - len(line.lstrip())
            
            target_class = scenario["class"]
            
            if '=' in line_content:
                assigned_class = line_content.split('=')[-1].strip()
                is_target_class = assigned_class == target_class
            else:
                is_target_class = False
            
            if is_target_class:
                if is_commented:
                    updated_lines.append(' ' * indent + f'SCENARIO_CLASS = {target_class}\n')
                else:
                    updated_lines.append(line)
            else:
                if not is_commented:
                    updated_lines.append(' ' * indent + '# ' + line_content + '\n')
                else:
                    updated_lines.append(line)
        # Handle scenario initialization line
        elif 'scenario =' in line and 'SCENARIO_CLASS' in line:
            stripped = line.strip()
            if stripped.startswith('#'):
                indent = len(line) - len(line.lstrip())
                updated_lines.append(' ' * indent + f'scenario = SCENARIO_CLASS(robot, timestep=config.TIMESTEP, algorithm="{algorithm}")\n')
            else:
                updated_lines.append(line)
            scenario_init_found = True
        else:
            updated_lines.append(line)
    
    # If scenario initialization line was not found, add it after robot initialization
    if not scenario_init_found:
        for i, line in enumerate(updated_lines):
            if 'robot = Supervisor()' in line:
                indent = len(line) - len(line.lstrip())
                scenario_line = ' ' * indent + f'scenario = SCENARIO_CLASS(robot, timestep=config.TIMESTEP, algorithm="{algorithm}")\n'
                updated_lines.insert(i + 1, scenario_line)
                break
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(updated_lines)


def update_checkpoint_in_file(file_path, checkpoint_path, algorithm="ddpg"):
    """Update checkpoint path in test controller file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    has_subdir = '/' in checkpoint_path or '\\' in checkpoint_path
    
    if has_subdir:
        path_parts = checkpoint_path.replace('\\', '/').split('/')
        
        # Build the path parts for os.path.join
        controller_dir = ALGORITHMS[algorithm]["controller_dir"]
        path_args = ["os.path.dirname(__file__)", "'..'", f"'{controller_dir}'", "'checkpoints'"]
        path_args.extend([f"'{part}'" for part in path_parts])
        path_join_expr = "os.path.join(" + ", ".join(path_args) + ")"
        
        # Replace the entire checkpoint_path assignment
        multiline_pattern = r'checkpoint_path\s*=\s*os\.path\.join\([^)]*\n[^)]*\)'
        singleline_pattern = r'checkpoint_path\s*=\s*os\.path\.join\([^)]+\)'
        
        replacement = f'checkpoint_path = {path_join_expr}'
        
        if re.search(multiline_pattern, content, re.MULTILINE):
            content = re.sub(multiline_pattern, replacement, content, flags=re.MULTILINE)
        elif re.search(singleline_pattern, content):
            content = re.sub(singleline_pattern, replacement, content)
        else:
            pattern = r'checkpoint_path\s*=\s*[^\n]+'
            content = re.sub(pattern, replacement, content)
    else:
        # Just filename
        controller_dir = ALGORITHMS[algorithm]["controller_dir"]
        
        # Pattern for multi-line
        multiline_pattern = r'(checkpoint_path\s*=\s*os\.path\.join\([^)]*\n\s*[^)]*,\s*)(?:config\.CHECKPOINT_NAME|"ddpg_model\.pt")'
        replacement_multiline = rf'\1"{checkpoint_path}"'
        
        # Pattern for single-line
        singleline_pattern = r'(checkpoint_path\s*=\s*os\.path\.join\([^)]+,\s*)(?:config\.CHECKPOINT_NAME|"ddpg_model\.pt")'
        replacement_singleline = rf'\1"{checkpoint_path}"'
        
        if re.search(multiline_pattern, content, re.MULTILINE | re.DOTALL):
            content = re.sub(multiline_pattern, replacement_multiline, content, flags=re.MULTILINE | re.DOTALL)
        elif re.search(singleline_pattern, content):
            content = re.sub(singleline_pattern, replacement_singleline, content)
        else:
            content = re.sub(
                r'"ddpg_model\.pt"',
                f'"{checkpoint_path}"',
                content
            )
            content = re.sub(
                r'config\.CHECKPOINT_NAME',
                f'"{checkpoint_path}"',
                content
            )
    
    with open(file_path, 'w') as f:
        f.write(content)


def launch_webots(world_file, mode="fast", env=None):
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


def validate_angle_check_setup():
    """Validate that angle check mode is properly set up."""
    if not ANGLE_CHECK_CONTROLLER.exists():
        print(f"❌ Angle check controller not found: {ANGLE_CHECK_CONTROLLER}")
        return False
    
    if not ANGLE_CHECK_WORLD.exists():
        print(f"❌ Angle check world file not found: {ANGLE_CHECK_WORLD}")
        return False
    
    return True


def validate_algorithm_config(algorithm):
    """Validate that the algorithm configuration exists."""
    if algorithm not in ALGORITHMS:
        print(f"❌ Unknown algorithm: {algorithm}")
        print(f"Available algorithms: {list(ALGORITHMS.keys())}")
        return False
    
    alg_config = ALGORITHMS[algorithm]
    
    # Check if train controller exists
    train_controller_path = CONTROLLERS_DIR / alg_config["controller_dir"] / alg_config["train_controller"]
    if not train_controller_path.exists():
        print(f"❌ Train controller not found: {train_controller_path}")
        return False
    
    # Check if world file exists
    train_world_path = PROJECT_ROOT / "worlds" / alg_config["train_world"]
    if not train_world_path.exists():
        print(f"❌ Train world file not found: {train_world_path}")
        return False
    
    return True


def create_test_controller_if_needed(algorithm):
    """Create test controller directory and files if they don't exist."""
    alg_config = ALGORITHMS[algorithm]
    
    # Test controller directory
    test_dir = CONTROLLERS_DIR / alg_config["test_controller_dir"]
    test_file = test_dir / alg_config["test_controller"]
    
    # If test directory doesn't exist, create it from template
    if not test_dir.exists():
        print(f"📁 Creating test controller directory for {algorithm}...")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py
        init_file = test_dir / "__init__.py"
        init_file.write_text("")
        
        # Create test controller file from DDPG template if it exists
        ddpg_test_dir = CONTROLLERS_DIR / "test_policy"
        ddpg_test_file = ddpg_test_dir / "test_policy.py"
        
        if ddpg_test_file.exists():
            with open(ddpg_test_file, 'r') as f:
                content = f.read()
            
            # Update imports for the specific algorithm
            controller_dir = alg_config["controller_dir"]
            content = content.replace(
                "from op3_ddpg_env.ddpg_agent import DDPG",
                f"from {controller_dir}.{algorithm}_agent import {algorithm.upper()}Agent"
            )
            
            # Update config import
            content = content.replace(
                "from op3_ddpg_env import config",
                f"from {controller_dir} import config"
            )
            
            # Save to new test controller
            with open(test_file, 'w') as f:
                f.write(content)
            print(f"✅ Created test controller: {test_file}")
        else:
            print(f"⚠️  Warning: Could not find DDPG test controller template")
    
    return test_file.exists()


def main():
    parser = argparse.ArgumentParser(
        description="Launch Webots training or testing with different scenarios and algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  python main.py --train --scenario=fall_control --alg=ddpg
  python main.py --train --scenario=fall_control --alg=ppo
  python main.py --train --scenario=fall_control --alg=sac
  
  # Genetic training (DDPG only)
  python main.py --genetic --scenario=fall_control --alg=ddpg
  
  # Testing
  python main.py --test --scenario=fall_control --alg=ppo
  python main.py --test --scenario=fall_control --alg=sac --checkpoint=sac_final.pt
  
  # Angle check
  python main.py --angle_check
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Run training mode')
    mode_group.add_argument('--genetic', action='store_true', help='Run genetic algorithm multi-agent training (DDPG only)')
    mode_group.add_argument('--test', action='store_true', help='Run testing mode')
    mode_group.add_argument('--angle_check', action='store_true', help='Manual joint angle testing mode')
    
    # Algorithm selection
    parser.add_argument(
        '--alg',
        type=str,
        default='ddpg',
        choices=['ddpg', 'ppo', 'sac'],
        help='Algorithm to use (default: ddpg)'
    )
    
    # Scenario selection
    parser.add_argument(
        '--scenario',
        type=str,
        default=None,
        choices=list(SCENARIOS.keys()) + [None],
        help='Scenario to use (not needed for --angle_check)'
    )
    
    # Checkpoint (for test mode)
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint file name (default: uses config.CHECKPOINT_NAME)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.angle_check:
        if args.scenario:
            print("⚠️  Warning: --scenario is ignored in --angle_check mode")
        if args.alg != 'ddpg':
            print("⚠️  Warning: --alg is ignored in --angle_check mode")
        
        if not validate_angle_check_setup():
            sys.exit(1)
    else:
        if not args.scenario:
            parser.error("--scenario is required for --train, --genetic, or --test modes")
        
        if not validate_algorithm_config(args.alg):
            sys.exit(1)
    
    try:
        if args.angle_check:
            print("🔧 Setting up joint angle check mode...")
            print(f"📁 Controller: {ANGLE_CHECK_CONTROLLER}")
            print(f"🌍 World file: {ANGLE_CHECK_WORLD}")
            
            print("\n" + "=" * 70)
            print("🤖 OP3 JOINT ANGLE CHECK MODE")
            print("=" * 70)
            
            # Launch Webots with angle check world
            launch_webots(ANGLE_CHECK_WORLD, mode="run")
            
        elif args.train:
            alg_config = ALGORITHMS[args.alg]
            
            print(f"📝 Setting up {args.alg.upper()} training with scenario: {args.scenario}")
            
            # Update scenario in train controller
            train_controller_path = CONTROLLERS_DIR / alg_config["controller_dir"] / alg_config["train_controller"]
            update_scenario_in_file(train_controller_path, args.scenario, args.alg)
            print(f"✅ Updated {train_controller_path}")
            
            # Launch training
            train_world_path = PROJECT_ROOT / "worlds" / alg_config["train_world"]
            print(f"🎮 Launching {args.alg.upper()} training world: {train_world_path.name}")
            launch_webots(train_world_path)
            
        elif args.genetic:
            if args.alg != 'ddpg':
                print(f"❌ Error: Genetic algorithm training is only supported for DDPG, not {args.alg}")
                print("Use --train mode for PPO and SAC training")
                sys.exit(1)
            
            alg_config = ALGORITHMS[args.alg]
            
            print(f"📝 Setting up parallel genetic training with scenario: {args.scenario}")
            
            # Update genetic parallel controller scenario
            genetic_parallel_controller = CONTROLLERS_DIR / alg_config["controller_dir"] / alg_config["genetic_parallel_controller"]
            update_scenario_in_file(genetic_parallel_controller, args.scenario, args.alg)
            print(f"✅ Updated {genetic_parallel_controller}")
            
            # Run parallel genetic training script
            parallel_script = CONTROLLERS_DIR / alg_config["controller_dir"] / alg_config["parallel_script"]
            scenario_class = SCENARIOS[args.scenario]["class"]
            
            print(f"🚀 Launching parallel genetic training for {args.alg.upper()}...")
            if genetic_config:
                print(f"   This will launch {genetic_config.POPULATION_SIZE} Webots instances in parallel")
            else:
                print(f"   This will launch multiple Webots instances in parallel")
            
            result = subprocess.run(
                [sys.executable, str(parallel_script), scenario_class],
                cwd=str(CONTROLLERS_DIR / alg_config["controller_dir"])
            )
            
            if result.returncode != 0:
                print(f"❌ Genetic training failed with exit code {result.returncode}")
                sys.exit(1)
            
        elif args.test:
            alg_config = ALGORITHMS[args.alg]
            
            print(f"📝 Setting up {args.alg.upper()} testing with scenario: {args.scenario}")
            
            # Create test controller if needed
            if not create_test_controller_if_needed(args.alg):
                print(f"❌ Test controller for {args.alg} could not be created")
                sys.exit(1)
            
            # Update scenario in test controller
            test_controller_path = CONTROLLERS_DIR / alg_config["test_controller_dir"] / alg_config["test_controller"]
            update_scenario_in_file(test_controller_path, args.scenario, args.alg)
            print(f"✅ Updated {test_controller_path}")
            
            # Update checkpoint path if specified
            if args.checkpoint:
                update_checkpoint_in_file(test_controller_path, args.checkpoint, args.alg)
                print(f"📦 Using checkpoint: {args.checkpoint}")
            
            # Set checkpoint path via environment variable
            env = os.environ.copy()
            if args.checkpoint:
                env['CHECKPOINT_PATH'] = args.checkpoint
            else:
                # Try to get default checkpoint from config
                try:
                    sys.path.insert(0, str(CONTROLLERS_DIR / alg_config["controller_dir"]))
                    import importlib
                    config_module = importlib.import_module(alg_config["config_module"])
                    
                    # Create checkpoint path
                    if hasattr(config_module, 'CHECKPOINT_NAME'):
                        scenario_checkpoint = f"{args.scenario}/{config_module.CHECKPOINT_NAME}"
                        env['CHECKPOINT_PATH'] = scenario_checkpoint
                        print(f"📦 Using default checkpoint for scenario: {scenario_checkpoint}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load config for {args.alg}: {e}")
            
            # Launch test world
            test_world_path = PROJECT_ROOT / "worlds" / alg_config["test_world"]
            print(f"🎮 Launching {args.alg.upper()} test world: {test_world_path.name}")
            launch_webots(test_world_path, mode="realtime", env=env)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()