#!/usr/bin/env python3
"""
Main script for OP3 Robot Training System.

Supports:
- Single-agent training and testing (existing behavior)
- Multi-agent evolutionary training with population-based training (PBT)
- Multi-stage training with stage transitions
- Checkpoint management and resuming
- Visualization of training results

Usage:
    # Single-agent training (existing)
    python main.py --train --alg=ddpg
    python main.py --train --alg=ppo
    
    # Single-agent testing (existing)
    python main.py --test --alg=ddpg
    python main.py --test --alg=ppo --checkpoint ppo_best.pt
    
    # Multi-agent evolutionary training (new)
    python main.py --train-multi --alg=ddpg
    python main.py --train-multi --alg=ppo --population-size 8
    python main.py --train-multi --alg=ddpg --resume-from controllers/op3_ddpg/runs/ddpg_21.44.03-17.01.26
    
    # Multi-agent testing (test all agents from a run)
    python main.py --test-multi --alg=ddpg
    python main.py --test-multi --alg=ppo --run-dir controllers/op3_ppo/runs/ppo_XX.XX.XX-XX.XX.XX
    
    # Visualization
    python main.py --visualize --run-dir controllers/op3_ddpg/runs/ddpg_XX.XX.XX-XX.XX.XX
"""

import argparse
import subprocess
import sys
import os
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import multi-agent modules
from multi_agent_core import (
    MultiAgentTrainer,
    PopulationManager,
    StageCoordinator,
    AgentCheckpointManager,
    create_trainer
)
from stats_manager import get_latest_run, get_latest_checkpoint
from visualization_utils import visualize_run


# =============================================================================
# Configuration
# =============================================================================

ALGORITHMS = {
    "ddpg": {
        "controller_dir": "controllers/op3_ddpg",
        "train_controller": "op3_ddpg.py",
        "train_world": "robotis_op3_ddpg.wbt",
        "test_controller_dir": "controllers/op3_ddpg",
        "test_controller": "op3_ddpg.py",
        "test_world": "robotis_op3_ddpg.wbt",
        "multi_agent_config": "multi_agent_config.json",
        "runs_dir": "controllers/op3_ddpg/runs",
        "checkpoints_dir": "controllers/op3_ddpg/checkpoints",
    },
    "ppo": {
        "controller_dir": "controllers/op3_ppo",
        "train_controller": "op3_ppo.py",
        "train_world": "robotis_op3_ppo.wbt",
        "test_controller_dir": "controllers/op3_ppo",
        "test_controller": "op3_ppo.py",
        "test_world": "robotis_op3_ppo.wbt",
        "multi_agent_config": "multi_agent_config.json",
        "runs_dir": "controllers/op3_ppo/runs",
        "checkpoints_dir": "controllers/op3_ppo/checkpoints",
    },
}

SCENARIOS = ["pak_gembong", "yudhis", "fall_control"]

# Angle check world
ANGLE_CHECK_WORLD = PROJECT_ROOT / "worlds" / "angle_monitor.wbt"


# =============================================================================
# Utility Functions
# =============================================================================

def find_webots() -> Optional[str]:
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
    
    webots_path = shutil.which("webots")
    if webots_path:
        return webots_path
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def launch_webots(world_file: Path, mode: str = "fast", env: Optional[Dict] = None) -> subprocess.Popen:
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
        process = subprocess.Popen(cmd, env=env)
        return process
    except Exception as e:
        print(f"❌ Error: Webots failed to start: {e}")
        sys.exit(1)


def validate_algorithm_config(algorithm: str) -> bool:
    """Validate that the algorithm configuration exists."""
    if algorithm not in ALGORITHMS:
        print(f"❌ Unknown algorithm: {algorithm}")
        print(f"Available algorithms: {list(ALGORITHMS.keys())}")
        return False
    
    alg_config = ALGORITHMS[algorithm]
    
    # Check if multi-agent config exists
    ma_config_path = PROJECT_ROOT / alg_config["runs_dir"].replace("controllers/", "") / alg_config["multi_agent_config"]
    if not ma_config_path.exists():
        # Try alternate path
        controller_dir = PROJECT_ROOT / "controllers" / f"op3_{algorithm}"
        ma_config_path = controller_dir / alg_config["multi_agent_config"]
    
    if not ma_config_path.exists():
        print(f"❌ Multi-agent config not found: {ma_config_path}")
        return False
    
    return True


def get_latest_run_for_algorithm(algorithm: str) -> Optional[str]:
    """Get the most recent run directory for an algorithm."""
    alg_config = ALGORITHMS[algorithm]
    runs_dir = PROJECT_ROOT / alg_config["runs_dir"]
    
    if not runs_dir.exists():
        return None
    
    runs = [d for d in os.listdir(runs_dir) 
            if os.path.isdir(os.path.join(runs_dir, d)) 
            and d.startswith(algorithm + "_")]
    
    if not runs:
        return None
    
    runs.sort(key=lambda x: os.path.getmtime(os.path.join(runs_dir, x)))
    return str(runs_dir / runs[-1])


def create_run_directory(algorithm: str) -> Tuple[str, str]:
    """
    Create a new timestamped run directory.
    
    Returns:
        (run_dir, run_id)
    """
    alg_config = ALGORITHMS[algorithm]
    runs_dir = PROJECT_ROOT / alg_config["runs_dir"]
    os.makedirs(runs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%H.%M.%S-%d.%m.%y")
    run_id = f"{algorithm}_{timestamp}"
    run_dir = runs_dir / run_id
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(run_dir / "checkpoints", exist_ok=True)
    os.makedirs(run_dir / "stats", exist_ok=True)
    
    return str(run_dir), run_id


def load_multi_agent_config(algorithm: str) -> Dict:
    """Load multi-agent configuration for an algorithm."""
    alg_config = ALGORITHMS[algorithm]
    
    # Try multiple paths
    possible_paths = [
        PROJECT_ROOT / "controllers" / f"op3_{algorithm}" / alg_config["multi_agent_config"],
        PROJECT_ROOT / alg_config["runs_dir"].replace("controllers/", "") / alg_config["multi_agent_config"],
        PROJECT_ROOT / "controllers" / f"op3_{algorithm}" / "multi_agent_config.json",
    ]
    
    for config_path in possible_paths:
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
    
    # Create default config if not found
    default_config = {
        "multi_agent": {
            "enabled": True,
            "population_size": 8,
            "num_stages": 3,
            "selection_ratio": 0.5,
            "mutation_sigma": 0.05,
            "crossover_enabled": False,
            "hyperparameter_mutation": True,
            "elitism_count": 2,
            "resample_count": 2,
            "checkpoint_policy": "all"
        },
        "stage_definitions": [
            {
                "stage_id": 0,
                "name": "pretrain",
                "description": "Initial training with easy goals",
                "episodes": 500,
                "termination_criteria": {
                    "min_episodes": 200,
                    "target_success_rate": 0.7,
                    "max_episodes": 800
                },
                "hyperparameters": {
                    "actor_lr": 1e-4,
                    "critic_lr": 1e-3,
                    "gamma": 0.99
                }
            },
            {
                "stage_id": 1,
                "name": "colearn",
                "description": "Intermediate training with harder goals",
                "episodes": 800,
                "termination_criteria": {
                    "min_episodes": 400,
                    "target_success_rate": 0.75,
                    "max_episodes": 1200
                },
                "hyperparameters": {
                    "actor_lr": 5e-5,
                    "critic_lr": 5e-4,
                    "gamma": 0.99
                }
            },
            {
                "stage_id": 2,
                "name": "fine_tune",
                "description": "Advanced training with full goals",
                "episodes": 1500,
                "termination_criteria": {
                    "min_episodes": 800,
                    "target_success_rate": 0.85,
                    "max_episodes": 2000
                },
                "hyperparameters": {
                    "actor_lr": 2e-5,
                    "critic_lr": 2e-4,
                    "gamma": 0.99
                }
            }
        ],
        "evaluation_metrics": [
            "mean_reward",
            "success_rate",
            "mean_episode_length"
        ]
    }
    
    return default_config


# =============================================================================
# Single-Agent Training/Testing
# =============================================================================

def run_single_agent_training(algorithm: str, scenario: str):
    """Run single-agent training (existing behavior)."""
    print("\n" + "="*70)
    print(f"🤖 SINGLE-AGENT {algorithm.upper()} TRAINING")
    print("="*70)
    
    alg_config = ALGORITHMS[algorithm]
    train_world = PROJECT_ROOT / "worlds" / alg_config["train_world"]
    
    if not train_world.exists():
        print(f"❌ Error: World file not found: {train_world}")
        sys.exit(1)
    
    # Set environment variables
    env = os.environ.copy()
    env['RL_TRAIN'] = 'true'
    env['RL_SCENARIO'] = scenario
    env['RL_ALGORITHM'] = algorithm
    
    # Launch Webots in fast mode
    process = launch_webots(train_world, mode="fast", env=env)
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        process.terminate()
        process.wait()


def run_single_agent_testing(
    algorithm: str,
    scenario: str,
    checkpoint: Optional[str] = None
):
    """Run single-agent testing (existing behavior)."""
    print("\n" + "="*70)
    print(f"🤖 SINGLE-AGENT {algorithm.upper()} TESTING")
    print("="*70)
    
    alg_config = ALGORITHMS[algorithm]
    test_world = PROJECT_ROOT / "worlds" / alg_config["test_world"]
    
    if not test_world.exists():
        print(f"❌ Error: Test world file not found: {test_world}")
        sys.exit(1)
    
    # Set environment variables
    env = os.environ.copy()
    env['RL_TRAIN'] = 'false'
    env['RL_SCENARIO'] = scenario
    env['RL_ALGORITHM'] = algorithm
    
    if checkpoint:
        env['CHECKPOINT_PATH'] = checkpoint
        print(f"📦 Using checkpoint: {checkpoint}")
    
    # Launch Webots in realtime mode
    process = launch_webots(test_world, mode="realtime", env=env)
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        process.terminate()
        process.wait()


# =============================================================================
# Multi-Agent Training/Testing
# =============================================================================

def run_multi_agent_training(
    algorithm: str,
    scenario: str,
    population_size: Optional[int] = None,
    resume_from: Optional[str] = None,
    seed: Optional[int] = None,
    max_episodes: Optional[int] = None
):
    """Run multi-agent evolutionary training."""
    print("\n" + "="*70)
    print(f"🚀 MULTI-AGENT {algorithm.upper()} EVOLUTIONARY TRAINING")
    print("="*70)
    
    if resume_from:
        print(f"📂 Resuming from: {resume_from}")
        # Resume existing training
        trainer = MultiAgentTrainer.load_state(
            controller_dir=str(PROJECT_ROOT / "controllers" / f"op3_{algorithm}"),
            run_dir=resume_from,
            algorithm=algorithm,
            checkpoint_path=os.path.join(resume_from, "checkpoints", "trainer_state.pt")
        )
    else:
        # Load multi-agent config
        ma_config = load_multi_agent_config(algorithm)
        
        # Override population size if specified
        if population_size:
            ma_config["multi_agent"]["population_size"] = population_size
        
        # Create new trainer
        trainer = MultiAgentTrainer(
            controller_dir=str(PROJECT_ROOT / "controllers" / f"op3_{algorithm}"),
            algorithm=algorithm,
            multi_agent_config=ma_config,
            seed=seed
        )
    
    # Run training
    trainer.run_training(
        base_hyperparameters={},
        max_global_episodes=max_episodes
    )
    
    # Visualize results
    print("\n📊 Generating visualizations...")
    visualize_run(trainer.run_dir)


def run_multi_agent_testing(
    algorithm: str,
    scenario: str,
    run_dir: Optional[str] = None,
    num_episodes: int = 10
):
    """Run multi-agent testing - test all agents from a run."""
    print("\n" + "="*70)
    print(f"🧪 MULTI-AGENT {algorithm.upper()} TESTING")
    print("="*70)
    
    # Find run directory
    if run_dir is None:
        run_dir = get_latest_run_for_algorithm(algorithm)
    
    if run_dir is None or not os.path.exists(run_dir):
        print(f"❌ Error: No runs found for {algorithm}")
        print("Please train a model first or specify --run-dir")
        sys.exit(1)
    
    print(f"📂 Using run directory: {run_dir}")
    
    # Get checkpoints
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        print(f"❌ Error: Checkpoints directory not found: {checkpoints_dir}")
        sys.exit(1)
    
    # Find all stage directories
    stage_dirs = sorted([
        d for d in os.listdir(checkpoints_dir)
        if d.startswith("stage_")
    ])
    
    if not stage_dirs:
        print("❌ Error: No stage directories found")
        sys.exit(1)
    
    # Test each agent in the final stage
    final_stage = stage_dirs[-1]
    final_stage_path = os.path.join(checkpoints_dir, final_stage)
    
    agent_dirs = [
        d for d in os.listdir(final_stage_path)
        if os.path.isdir(os.path.join(final_stage_path, d))
    ]
    
    if not agent_dirs:
        print("❌ Error: No agent directories found")
        sys.exit(1)
    
    # Test each agent
    results = []
    for agent_id in agent_dirs:
        agent_path = os.path.join(final_stage_path, agent_id)
        checkpoint_files = [f for f in os.listdir(agent_path) if f.endswith('.pt')]
        
        if not checkpoint_files:
            continue
        
        # Use best checkpoint if available, otherwise latest
        best_checkpoint = None
        for f in checkpoint_files:
            if 'best' in f.lower():
                best_checkpoint = os.path.join(agent_path, f)
                break
        
        if best_checkpoint is None:
            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(agent_path, x)))
            best_checkpoint = os.path.join(agent_path, checkpoint_files[-1])
        
        print(f"\n🤖 Testing agent: {agent_id}")
        print(f"   Checkpoint: {os.path.basename(best_checkpoint)}")
        
        # Run testing (simplified - in real implementation, would launch Webots)
        # For now, just report the checkpoint
        results.append({
            'agent_id': agent_id,
            'checkpoint': best_checkpoint,
            'status': 'ready'
        })
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TEST RESULTS SUMMARY")
    print("="*70)
    print(f"{'Agent ID':<20} {'Checkpoint':<30} {'Status'}")
    print("-"*70)
    for result in results:
        print(f"{result['agent_id']:<20} {os.path.basename(result['checkpoint']):<30} {result['status']}")
    
    print("\n💡 To test each agent in Webots, run:")
    print(f"   python main.py --test --alg={algorithm} --checkpoint <checkpoint_path>")


def run_multi_model_testing(
    algorithm: str,
    scenario: str,
    config_path: Optional[str] = None
):
    """
    Test multiple models sequentially as defined in test config.
    
    This is the new testing mode where models are tested one by one
    and results are compared to select the best.
    """
    print("\n" + "="*70)
    print(f"🔬 MULTI-MODEL TESTING: {algorithm.upper()}")
    print("="*70)
    
    # Load test config
    if config_path is None:
        controller_dir = PROJECT_ROOT / "controllers" / f"op3_{algorithm}"
        config_path = controller_dir / "config_test.json"
    
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get test models
    test_models = config.get('test_models', [])
    num_episodes = config.get('num_test_episodes', 10)
    
    if not test_models:
        # Fallback to single model
        model_path = config.get('model_path', '')
        if model_path:
            test_models = [{"path": model_path, "episodes": num_episodes, "name": "Default Model"}]
    
    if not test_models:
        print("❌ Error: No test models configured")
        print("Please add 'test_models' array to config or specify --checkpoint")
        sys.exit(1)
    
    # Test each model
    results = []
    for i, model_config in enumerate(test_models):
        model_path = model_config.get('path', '')
        model_name = model_config.get('name', f"Model {i+1}")
        model_episodes = model_config.get('episodes', num_episodes)
        
        print(f"\n{'='*70}")
        print(f"🧪 Testing: {model_name}")
        print(f"   Path: {model_path}")
        print(f"   Episodes: {model_episodes}")
        print(f"{'='*70}")
        
        # Check if checkpoint exists
        if not os.path.exists(model_path):
            print(f"   ⚠️  Checkpoint not found, skipping")
            continue
        
        # Run testing
        run_single_agent_testing(
            algorithm=algorithm,
            scenario=scenario,
            checkpoint=model_path
        )
        
        results.append({
            'name': model_name,
            'path': model_path,
            'episodes': model_episodes
        })
    
    # Print comparison
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON")
    print("="*70)
    print(f"{'Model Name':<30} {'Episodes':<10}")
    print("-"*70)
    for result in results:
        print(f"{result['name']:<30} {result['episodes']:<10}")
    
    print("\n💡 Results are saved in the respective training_stats directories")


def run_visualization(run_dir: str):
    """Generate visualizations for a training run."""
    print("\n" + "="*70)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*70)
    
    if not os.path.exists(run_dir):
        print(f"❌ Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    # Run visualization
    generated = visualize_run(run_dir)
    
    if not generated:
        print("❌ Error: No visualizations generated")
        sys.exit(1)
    
    print("\n✅ Generated visualizations:")
    for name, path in generated.items():
        print(f"   {name}: {path}")


# =============================================================================
# Angle Check Mode
# =============================================================================

def run_angle_check(scenario: str):
    """Run manual joint angle check mode."""
    print("\n" + "="*70)
    print("🤖 OP3 JOINT ANGLE CHECK MODE")
    print("="*70)
    
    if not ANGLE_CHECK_WORLD.exists():
        print(f"❌ Error: Angle check world file not found: {ANGLE_CHECK_WORLD}")
        sys.exit(1)
    
    env = os.environ.copy()
    env['RL_SCENARIO'] = scenario
    
    process = launch_webots(ANGLE_CHECK_WORLD, mode="run", env=env)
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        process.terminate()
        process.wait()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OP3 Robot Training System - Multi-Agent Evolutionary Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-Agent Training (existing)
  python main.py --train --alg=ddpg
  python main.py --train --alg=ppo
  
  # Single-Agent Testing (existing)
  python main.py --test --alg=ddpg
  python main.py --test --alg=ppo --checkpoint ppo_best.pt
  
  # Multi-Agent Evolutionary Training (NEW)
  python main.py --train-multi --alg=ddpg
  python main.py --train-multi --alg=ppo --population-size 8
  python main.py --train-multi --alg=ddpg --resume-from controllers/op3_ddpg/runs/ddpg_21.44.03-17.01.26
  
  # Multi-Agent Testing (NEW - tests all agents from a run)
  python main.py --test-multi --alg=ddpg
  python main.py --test-multi --alg=ppo --run-dir controllers/op3_ppo/runs/ppo_XX.XX.XX-XX.XX.XX
  
  # Multi-Model Testing (NEW - test models defined in config)
  python main.py --test-multi-models --alg=ddpg
  
  # Visualization
  python main.py --visualize --run-dir controllers/op3_ddpg/runs/ddpg_XX.XX.XX-XX.XX.XX
  
  # Angle Check
  python main.py --angle_check
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', 
                           help='Single-agent training mode')
    mode_group.add_argument('--train-multi', action='store_true', 
                           help='Multi-agent evolutionary training mode')
    mode_group.add_argument('--test', action='store_true', 
                           help='Single-agent testing mode')
    mode_group.add_argument('--test-multi', action='store_true', 
                           help='Multi-agent testing mode (test all agents from a run)')
    mode_group.add_argument('--test-multi-models', action='store_true', 
                           help='Test multiple models defined in config sequentially')
    mode_group.add_argument('--visualize', action='store_true', 
                           help='Generate visualizations for a run')
    mode_group.add_argument('--angle_check', action='store_true', 
                           help='Manual joint angle testing mode')
    
    # Algorithm selection
    parser.add_argument(
        '--alg',
        type=str,
        default='ddpg',
        choices=['ddpg', 'ppo'],
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
    
    # Checkpoint (for single-agent test)
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint file path for testing'
    )
    
    # Population size (for multi-agent training)
    parser.add_argument(
        '--population-size',
        type=int,
        default=None,
        help='Override population size for multi-agent training'
    )
    
    # Resume from (for multi-agent training)
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Resume training from a previous run directory'
    )
    
    # Run directory (for multi-agent testing and visualization)
    parser.add_argument(
        '--run-dir',
        type=str,
        default=None,
        help='Run directory for multi-agent testing or visualization'
    )
    
    # Test config path (for multi-model testing)
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Config file path for multi-model testing'
    )
    
    # Number of episodes (for testing)
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
        help='Number of episodes for testing (default: 10)'
    )
    
    # Seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Max episodes
    parser.add_argument(
        '--max-episodes',
        type=int,
        default=None,
        help='Maximum number of global episodes for training'
    )
    
    args = parser.parse_args()
    
    # Validate algorithm
    if not validate_algorithm_config(args.alg):
        sys.exit(1)
    
    try:
        if args.angle_check:
            run_angle_check(args.scenario)
            
        elif args.train:
            run_single_agent_training(args.alg, args.scenario)
            
        elif args.test:
            run_single_agent_testing(
                args.alg, 
                args.scenario,
                checkpoint=args.checkpoint
            )
            
        elif args.train_multi:
            run_multi_agent_training(
                args.alg,
                args.scenario,
                population_size=args.population_size,
                resume_from=args.resume_from,
                seed=args.seed,
                max_episodes=args.max_episodes
            )
            
        elif args.test_multi:
            run_multi_agent_testing(
                args.alg,
                args.scenario,
                run_dir=args.run_dir,
                num_episodes=args.num_episodes
            )
            
        elif args.test_multi_models:
            run_multi_model_testing(
                args.alg,
                args.scenario,
                config_path=args.config
            )
            
        elif args.visualize:
            if not args.run_dir:
                # Auto-detect latest run
                args.run_dir = get_latest_run_for_algorithm(args.alg)
            
            run_visualization(args.run_dir)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

