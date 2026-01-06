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
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
CONTROLLERS_DIR = PROJECT_ROOT / "controllers"

# Algorithm configurations
ALGORITHMS = {
    "ddpg": {
        "controller_dir": "op3_ddpg_env",
        "train_controller": "op3_ddpg_env.py",
        "train_world": "robotis_op3_train.wbt",
        "test_controller_dir": "test_policy",
        "test_controller": "test_policy.py",
        "test_world": "robotis_op3_test.wbt",
    },
    "ppo": {
        "controller_dir": "op3_ppo_env",
        "train_controller": "op3_ppo_env.py",
        "train_world": "robotis_op3_train_ppo.wbt",
        "test_controller_dir": "test_ppo_env",
        "test_controller": "test_ppo_env.py",
        "test_world": "robotis_op3_test_ppo.wbt",
    },
    "sac": {
        "controller_dir": "op3_sac_env",
        "train_controller": "op3_sac_env.py",
        "train_world": "robotis_op3_train_sac.wbt",
        "test_controller_dir": "test_sac_env",
        "test_controller": "test_sac_env.py",
        "test_world": "robotis_op3_test_sac.wbt",
    }
}

# Angle check mode
ANGLE_CHECK_WORLD = PROJECT_ROOT / "worlds" / "robotis_op3_angle_check.wbt"

SCENARIOS = ["pak_gembong", "yudhis", "fall_control"]


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
        default='fall_control',
        choices=SCENARIOS,
        help='Scenario to use (default: fall_control)'
    )
    
    # Checkpoint (for test mode)
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint file name or path (e.g., ppo_final.pt or path/to/checkpoint.pt)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.angle_check:
        if not validate_angle_check_setup():
            sys.exit(1)
    else:
        if not validate_algorithm_config(args.alg):
            sys.exit(1)
    
    # Setup environment variables for controllers
    env = os.environ.copy()
    env['RL_SCENARIO'] = args.scenario
    env['RL_ALGORITHM'] = args.alg
    
    try:
        if args.angle_check:
            print("🔧 Setting up joint angle check mode...")
            print(f"🌍 World file: {ANGLE_CHECK_WORLD}")
            
            print("\n" + "=" * 70)
            print("🤖 OP3 JOINT ANGLE CHECK MODE")
            print("=" * 70)
            
            launch_webots(ANGLE_CHECK_WORLD, mode="run", env=env)
            
        elif args.train:
            alg_config = ALGORITHMS[args.alg]
            
            print(f"📝 Setting up {args.alg.upper()} training")
            print(f"   Algorithm: {args.alg}")
            print(f"   Scenario: {args.scenario}")
            
            # Launch training
            train_world_path = PROJECT_ROOT / "worlds" / alg_config["train_world"]
            print(f"🎮 Launching {args.alg.upper()} training world: {train_world_path.name}")
            launch_webots(train_world_path, mode="fast", env=env)
            
        elif args.test:
            alg_config = ALGORITHMS[args.alg]
            
            print(f"📝 Setting up {args.alg.upper()} testing")
            print(f"   Algorithm: {args.alg}")
            print(f"   Scenario: {args.scenario}")
            
            # Set checkpoint path via environment variable
            if args.checkpoint:
                env['CHECKPOINT_PATH'] = args.checkpoint
                print(f"📦 Using checkpoint: {args.checkpoint}")
            else:
                default_checkpoint = f"{args.alg}_final.pt"
                env['CHECKPOINT_PATH'] = default_checkpoint
                print(f"📦 Using default checkpoint: {default_checkpoint}")
            
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