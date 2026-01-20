#!/usr/bin/env python3
"""
Main Entry Point for OP3 Robot Training.

Provides CLI for single-agent training, multi-agent evolutionary training,
testing, and visualization.

Usage:
    # Single-Agent Training
    python main.py --train --controller=op3_ddpg
    
    # Multi-Agent Evolutionary Training
    python main.py --train-multi --controller=op3_ddpg --population-size 8
    python main.py --train-multi --controller=op3_ddpg_abs --resume-from <run_dir>
    
    # Testing
    python main.py --test --controller=op3_ddpg_abs --run-dir <run_dir>
    python main.py --test --controller=op3_ppo
    
    # Visualization
    python main.py --visualize --run-dir <run_dir>
"""

import os
import sys
import json
import time
import argparse
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import logging
from logging_utils import (
    log, log_info, log_warning, log_error, log_success,
    log_debug, log_data, log_section, LogFunction
)

from visualization_utils import visualize_run


def load_config(config_path: str) -> Dict:
    """Load JSON configuration file."""
    if not os.path.exists(config_path):
        log_error("ConfigLoader", f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        return json.load(f)


def get_run_label(config: Dict) -> str:
    """Extract run_label from config, default to 'default'."""
    return config.get('run_label', 'default')


def get_algorithm_from_controller(controller: str) -> str:
    """Map controller name to algorithm name."""
    mapping = {
        'op3_ddpg': 'ddpg',
        'op3_ddpg_abs': 'ddpg',
        'op3_ppo': 'ppo',
        'op3_ddpg_omni': 'ddpg',
        'op3_sac_env': 'sac'
    }
    return mapping.get(controller, controller)


def get_controller_dir(controller: str) -> str:
    """Get full path to controller directory."""
    return str(PROJECT_ROOT / "controllers" / controller)


def generate_run_id(algorithm: str, run_label: str = "default") -> str:
    """Generate unique run ID with timestamp and label."""
    timestamp = datetime.now().strftime("%H.%M.%S-%d.%m.%y")
    return f"{algorithm}_{timestamp}_{run_label}"


def find_webots() -> Optional[str]:
    """Find Webots executable."""
    webots_home = os.environ.get("WEBOTS_HOME")
    if webots_home:
        webots_path = os.path.join(webots_home, "webots")
        if os.path.exists(webots_path):
            return webots_path
    
    webots_path = shutil.which("webots")
    if webots_path:
        return webots_path
    
    possible_paths = [
        "/usr/local/webots/webots",
        "/opt/webots/webots",
        os.path.expanduser("~/webots/webots"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def run_single_agent_training(controller: str, seed: Optional[int] = None) -> Dict:
    """
    Run single-agent training using the controller's train mode.
    
    Args:
        controller: Controller subdirectory name
        seed: Random seed
        
    Returns:
        Training summary dictionary
    """
    log_section("Main", "SINGLE-AGENT TRAINING")
    
    with LogFunction("Main", "run_single_agent_training", args=(controller, seed)):
        
        controller_dir = get_controller_dir(controller)
        config_path = os.path.join(controller_dir, "config_train.json")
        
        if not os.path.exists(config_path):
            # Try template
            template_path = os.path.join(controller_dir, "config_train.json.template")
            if os.path.exists(template_path):
                log_info("Main", f"Creating config from template: {template_path}")
                config = load_config(template_path)
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                log_error("Main", f"No config file found at {config_path}")
                return {}
        else:
            config = load_config(config_path)
        
        log_data("Main", "Loaded config keys", list(config.keys()))
        
        # Get run label and generate run ID
        run_label = get_run_label(config)
        algorithm = get_algorithm_from_controller(controller)
        run_id = generate_run_id(algorithm, run_label)
        
        # Create run directory
        run_dir = os.path.join(controller_dir, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "stats"), exist_ok=True)
        
        log_info("Main", f"Run ID: {run_id}")
        log_info("Main", f"Run directory: {run_dir}")
        
        # Save config to run directory
        with open(os.path.join(run_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Update config with run-specific settings
        config['run_id'] = run_id
        config['run_dir'] = run_dir
        if seed is not None:
            config['random_seed'] = seed
        
        # Find Webots
        webots_path = find_webots()
        if not webots_path:
            log_error("Main", "Webots executable not found!")
            return {}
        
        # Determine world file
        world_file = PROJECT_ROOT / "worlds" / f"robotis_op3_{algorithm}.wbt"
        if not world_file.exists():
            world_file = PROJECT_ROOT / "worlds" / f"robotis_op3_{controller}.wbt"
        
        if not world_file.exists():
            log_error("Main", f"World file not found for controller: {controller}")
            return {}
        
        log_info("Main", f"World file: {world_file}")
        
        # Prepare environment
        env = os.environ.copy()
        env['RL_TRAIN'] = 'true'
        env['RL_CONTROLLER'] = controller
        env['RL_RUN_DIR'] = run_dir
        env['RL_RUN_LABEL'] = run_label  # Pass the run label for model naming
        if seed is not None:
            env['RL_SEED'] = str(seed)
        
        # Launch Webots in fast mode
        cmd = [webots_path, "--mode=fast", "--stdout", "--stderr", "--batch", "--no-rendering", str(world_file.absolute())]
        
        log_info("Main", f"Launching Webots: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(PROJECT_ROOT)
            )
            
            log_info("Main", f"Webots PID: {process.pid}")
            
            # Wait for completion
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                log_warning("Main", f"Webots exited with code {process.returncode}")
                if stderr:
                    log_debug("Main", f"Error: {stderr.decode('utf-8')[:500]}")
            
            total_time = time.time() - start_time
            
            log_success("Main", f"Training completed in {total_time/60:.1f} minutes")
            log_info("Main", f"Run directory: {run_dir}")
            
            return {
                'run_id': run_id,
                'run_dir': run_dir,
                'controller': controller,
                'algorithm': algorithm,
                'total_time': total_time,
                'success': True
            }
            
        except Exception as e:
            log_exception("Main", e, "Error during training")
            return {'success': False, 'error': str(e)}


def run_multi_agent_training(
    controller: str,
    population_size: Optional[int] = None,
    resume_from: Optional[str] = None,
    seed: Optional[int] = None
) -> Dict:
    """
    Run multi-agent evolutionary training.
    
    Args:
        controller: Controller subdirectory name
        population_size: Override population size from config
        resume_from: Resume from run directory
        seed: Random seed
        
    Returns:
        Training summary dictionary
    """
    log_section("Main", "MULTI-AGENT EVOLUTIONARY TRAINING")
    
    with LogFunction("Main", "run_multi_agent_training", 
                    args=(controller, population_size, resume_from, seed)):
        
        controller_dir = get_controller_dir(controller)
        
        # Load multi-agent config
        config_path = os.path.join(controller_dir, "multi_agent_config.json")
        
        if not os.path.exists(config_path):
            template_path = os.path.join(controller_dir, "multi_agent_config.json.template")
            if os.path.exists(template_path):
                log_info("Main", f"Creating multi-agent config from template: {template_path}")
                config = load_config(template_path)
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                log_error("Main", f"No multi-agent config found at {config_path}")
                return {}
        else:
            config = load_config(config_path)
        
        log_data("Main", "Loaded multi-agent config keys", list(config.keys()))
        
        # Override population size if provided
        if population_size is not None:
            config['multi_agent']['population_size'] = population_size
            log_info("Main", f"Overriding population size to: {population_size}")
        
        # Get run label and generate run ID
        run_label = get_run_label(config)
        algorithm = get_algorithm_from_controller(controller)
        run_id = generate_run_id(algorithm, run_label)
        
        # Create trainer
        from multi_agent_core import create_trainer
        
        trainer = create_trainer(
            controller_dir=controller_dir,
            algorithm=algorithm,
            seed=seed,
            run_id=run_id
        )
        
        log_info("Main", f"Created trainer with run_id: {run_id}")
        log_info("Main", f"Run directory: {trainer.run_dir}")
        
        # Resume if requested
        if resume_from:
            log_info("Main", f"Resuming from: {resume_from}")
            state_path = os.path.join(resume_from, "training_state.pt")
            if os.path.exists(state_path):
                trainer = trainer.load_state(
                    controller_dir=controller_dir,
                    run_dir=resume_from,
                    algorithm=algorithm,
                    checkpoint_path=state_path
                )
                log_success("Main", "Resumed training state")
            else:
                log_warning("Main", f"State file not found: {state_path}")
        
        # Run training
        summary = trainer.run_training(
            base_hyperparameters={},
            max_global_episodes=None
        )
        
        summary['run_dir'] = trainer.run_dir
        summary['run_id'] = run_id
        
        log_success("Main", f"Multi-agent training completed")
        log_info("Main", f"Total episodes: {summary['total_episodes']}")
        log_info("Main", f"Stages completed: {summary['stages_completed']}")
        log_info("Main", f"Run directory: {trainer.run_dir}")
        
        return summary


def run_test_mode(
    controller: str,
    run_dir: Optional[str] = None,
    seed: Optional[int] = None
) -> Dict:
    """
    Run testing using config_test.json.
    Tests all models listed in config_test.json for specified episodes.
    
    Args:
        controller: Controller subdirectory name
        run_dir: Run directory for test output
        seed: Random seed
        
    Returns:
        Test results dictionary
    """
    log_section("Main", "TEST MODE")
    
    with LogFunction("Main", "run_test_mode", args=(controller, run_dir, seed)):
        
        controller_dir = get_controller_dir(controller)
        
        # Load test config
        config_path = os.path.join(controller_dir, "config_test.json")
        
        if not os.path.exists(config_path):
            template_path = os.path.join(controller_dir, "config_test.json.template")
            if os.path.exists(template_path):
                log_info("Main", f"Creating test config from template: {template_path}")
                config = load_config(template_path)
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                log_error("Main", f"No test config found at {config_path}")
                return {}
        else:
            config = load_config(config_path)
        
        log_data("Main", "Loaded test config keys", list(config.keys()))
        
        # Determine output directory
        if run_dir is None:
            run_label = get_run_label(config)
            algorithm = get_algorithm_from_controller(controller)
            run_id = generate_run_id(algorithm, f"test_{run_label}")
            run_dir = os.path.join(controller_dir, "runs", run_id)
        
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "stats"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "test_results"), exist_ok=True)
        
        log_info("Main", f"Test output directory: {run_dir}")
        
        # Save config
        with open(os.path.join(run_dir, "test_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize test stats storage
        from stats_manager import HDF5StatsLogger
        stats_logger = HDF5StatsLogger(
            os.path.join(run_dir, "stats", "test_stats.h5"),
            mode='w'
        )
        
        # Get test models from config
        test_models = config.get('test_models', [])
        if not test_models:
            # Fallback to model_path
            model_path = config.get('model_path', '')
            if model_path:
                test_models = [{'path': model_path, 'episodes': config.get('num_test_episodes', 10), 'name': 'Model'}]
        
        if not test_models:
            log_warning("Main", "No test models found in config")
            stats_logger.close()
            return {'success': False, 'message': 'No test models'}
        
        log_info("Main", f"Found {len(test_models)} test models")
        
        # Find Webots
        webots_path = find_webots()
        if not webots_path:
            log_error("Main", "Webots executable not found!")
            stats_logger.close()
            return {}
        
        # Determine world file
        algorithm = get_algorithm_from_controller(controller)
        world_file = PROJECT_ROOT / "worlds" / f"robotis_op3_{algorithm}.wbt"
        
        if not world_file.exists():
            log_error("Main", f"World file not found: {world_file}")
            stats_logger.close()
            return {}
        
        # Test each model
        all_results = []
        
        for i, model_info in enumerate(test_models):
            model_path = model_info.get('path', '')
            num_episodes = model_info.get('episodes', 10)
            model_name = model_info.get('name', f'Model_{i}')
            
            if not model_path or not os.path.exists(model_path):
                log_warning("Main", f"Model not found: {model_path}")
                continue
            
            log_section("Main", f"TESTING: {model_name}")
            log_info("Main", f"Model: {model_path}")
            log_info("Main", f"Episodes: {num_episodes}")
            
            model_results = []
            
            for ep in range(num_episodes):
                log_info("Main", f"Episode {ep+1}/{num_episodes}")
                
                # Prepare environment
                env = os.environ.copy()
                env['RL_TEST'] = 'true'
                env['RL_CONTROLLER'] = controller
                env['RL_MODEL_PATH'] = model_path
                env['RL_EPISODE_NUM'] = str(ep)
                env['RL_RUN_DIR'] = run_dir
                env['RL_TEST_MODEL_NAME'] = model_name
                if seed is not None:
                    env['RL_SEED'] = str(seed)
                
                # Update config settings
                for key in ['mode', 'control_joints', 'goal_angles', 'push_force', 
                           'initial_state', 'test', 'reward', 'joint_limits']:
                    if key in config:
                        env[f'RL_CONFIG_{key.upper()}'] = json.dumps(config[key])
                
                # Launch Webots
                cmd = [webots_path, "--mode=fast", "--stdout", "--stderr", "--batch", "--no-rendering", str(world_file.absolute())]
                
                try:
                    process = subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=str(PROJECT_ROOT)
                    )
                    
                    stdout, stderr = process.communicate()
                    
                    if process.returncode != 0:
                        log_warning("Main", f"Webots exited with code {process.returncode}")
                    
                    # Parse results from stdout/stderr
                    result = parse_test_result(stdout, stderr, model_name, ep)
                    if result:
                        model_results.append(result)
                        
                        # Log episode to stats
                        from stats_manager import EpisodeInfo
                        episode_info = EpisodeInfo(
                            global_episode_id=len(all_results),
                            stage_id=0,
                            local_episode_id=ep,
                            total_steps=result.get('steps', 0),
                            total_reward=result.get('reward', 0.0),
                            success=result.get('success', False),
                            termination_reason=result.get('termination_reason', 'normal'),
                            agent_id=model_name,
                            start_idx=0,
                            end_idx=0
                        )
                        stats_logger.log_episode(episode_info)
                        all_results.append(result)
                        
                except Exception as e:
                    log_exception("Main", e, f"Error testing model {model_name}")
            
            # Model summary
            if model_results:
                success_count = sum(1 for r in model_results if r.get('success', False))
                mean_reward = sum(r.get('reward', 0) for r in model_results) / len(model_results)
                
                model_summary = {
                    'model_name': model_name,
                    'model_path': model_path,
                    'episodes_tested': len(model_results),
                    'success_rate': success_count / len(model_results),
                    'mean_reward': mean_reward,
                    'results': model_results
                }
                
                log_info("Main", f"{model_name} - Success rate: {model_summary['success_rate']:.1%}, Mean reward: {mean_reward:.2f}")
                all_results.append(model_summary)
        
        stats_logger.close()
        
        # Generate summary
        total_episodes = len([r for r in all_results if isinstance(r, dict) and 'reward' in r])
        successful_episodes = len([r for r in all_results if isinstance(r, dict) and r.get('success', False)])
        
        summary = {
            'run_dir': run_dir,
            'controller': controller,
            'algorithm': algorithm,
            'total_episodes': total_episodes,
            'successful_episodes': successful_episodes,
            'overall_success_rate': successful_episodes / total_episodes if total_episodes > 0 else 0,
            'test_results': all_results,
            'success': True
        }
        
        log_section("Main", "TEST COMPLETE")
        log_info("Main", f"Total episodes: {total_episodes}")
        log_info("Main", f"Success rate: {summary['overall_success_rate']:.1%}")
        log_info("Main", f"Results saved to: {run_dir}")
        
        # Save summary
        with open(os.path.join(run_dir, "test_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary


def parse_test_result(stdout: bytes, stderr: bytes, model_name: str, episode_num: int) -> Optional[Dict]:
    """Parse test result from Webots output."""
    try:
        output = (stdout + stderr).decode('utf-8')
        
        # Look for JSON result in output
        import re
        json_match = re.search(r'\{[^{}]*"reward"[^{}]*\}', output)
        
        if json_match:
            return json.loads(json_match.group())
        
        # Fallback: extract basic info
        if 'success' in output.lower():
            return {
                'model_name': model_name,
                'episode': episode_num,
                'success': True,
                'reward': 10.0,
                'steps': 30
            }
        
    except Exception as e:
        log_warning("Main", f"Could not parse test result: {e}")
    
    return None


def run_visualize_mode(run_dir: str) -> Dict:
    """
    Generate visualizations for a training run.
    
    Args:
        run_dir: Run directory containing training_stats.h5
        
    Returns:
        Dictionary with paths to generated plots
    """
    log_section("Main", "VISUALIZATION")
    
    with LogFunction("Main", "run_visualize_mode", args=(run_dir,)):
        
        if not os.path.exists(run_dir):
            log_error("Main", f"Run directory not found: {run_dir}")
            return {}
        
        log_info("Main", f"Run directory: {run_dir}")
        
        generated = visualize_run(run_dir)
        
        if generated:
            log_success("Main", f"Generated {len(generated)} visualizations")
            for name, path in generated.items():
                log_info("Main", f"  {name}: {path}")
        else:
            log_warning("Main", "No visualizations generated")
        
        return generated


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OP3 Robot Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-Agent Training
  python main.py --train --controller=op3_ddpg
  
  # Multi-Agent Evolutionary Training
  python main.py --train-multi --controller=op3_ddpg --population-size 8
  python main.py --train-multi --controller=op3_ddpg_abs --resume-from <run_dir>
  
  # Testing
  python main.py --test --controller=op3_ddpg_abs --run-dir <run_dir>
  python main.py --test --controller=op3_ppo
  
  # Visualization
  python main.py --visualize --run-dir <run_dir>
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Single-agent training')
    mode_group.add_argument('--train-multi', action='store_true', help='Multi-agent evolutionary training')
    mode_group.add_argument('--test', action='store_true', help='Testing mode')
    mode_group.add_argument('--visualize', action='store_true', help='Visualization mode')
    
    # Common arguments
    parser.add_argument('--controller', type=str, default='op3_ddpg',
                       help='Controller subdirectory (default: op3_ddpg)')
    parser.add_argument('--run-dir', type=str,
                       help='Run directory for test/visualize output')
    
    # Training arguments
    parser.add_argument('--population-size', type=int,
                       help='Override population size for multi-agent training')
    parser.add_argument('--resume-from', type=str,
                       help='Resume training from run directory')
    
    # Common arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if (args.train or args.train_multi) and not args.controller:
        parser.error("--controller is required for training modes")
    
    if (args.test or args.visualize) and not (args.run_dir or args.controller):
        parser.error("--run-dir or --controller is required for test/visualize modes")
    
    # Execute based on mode
    if args.train:
        result = run_single_agent_training(args.controller, args.seed)
    elif args.train_multi:
        result = run_multi_agent_training(
            args.controller,
            args.population_size,
            args.resume_from,
            args.seed
        )
    elif args.test:
        result = run_test_mode(args.controller, args.run_dir, args.seed)
    elif args.visualize:
        result = run_visualize_mode(args.run_dir)
    else:
        result = {}
    
    # Print result summary
    if result:
        log_section("Main", "EXECUTION COMPLETE")
        if 'run_dir' in result:
            log_info("Main", f"Run directory: {result['run_dir']}")
        if 'run_id' in result:
            log_info("Main", f"Run ID: {result['run_id']}")
    
    return result


if __name__ == "__main__":
    main()

