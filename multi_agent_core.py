#!/usr/bin/env python3
"""
Multi-Agent Core - Main Orchestrator for Evolutionary Training.

Provides population management, stage coordination, and checkpoint management
for multi-agent, multi-stage evolutionary reinforcement learning.
"""

import os
import sys
import json
import time
import copy
import torch
import random
import hashlib
import argparse
import subprocess
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from stats_manager import HDF5StatsLogger, StageInfo, EpisodeInfo, AgentInfo
from evolutionary_operators import (
    SelectionOperator,
    MutationOperator,
    EvolutionaryAlgorithm,
    EvolutionResult
)


@dataclass
class Agent:
    """Represents an agent in the population."""
    agent_id: str
    agent_type: str  # 'ddpg' or 'ppo'
    model_state: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    fitness: float = 0.0
    episode_count: int = 0
    parent_id: Optional[str] = None
    creation_timestamp: str = ""
    lineage_depth: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'model_state': {k: v.cpu().tolist() if isinstance(v, torch.Tensor) else v 
                          for k, v in self.model_state.items()},
            'hyperparameters': self.hyperparameters,
            'fitness': self.fitness,
            'episode_count': self.episode_count,
            'parent_id': self.parent_id,
            'creation_timestamp': self.creation_timestamp,
            'lineage_depth': self.lineage_depth
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Agent':
        """Create Agent from dictionary."""
        # Convert lists back to tensors
        model_state = {}
        for k, v in data['model_state'].items():
            if isinstance(v, list):
                model_state[k] = torch.tensor(v)
            else:
                model_state[k] = v
        
        return cls(
            agent_id=data['agent_id'],
            agent_type=data['agent_type'],
            model_state=model_state,
            hyperparameters=data['hyperparameters'],
            fitness=data.get('fitness', 0.0),
            episode_count=data.get('episode_count', 0),
            parent_id=data.get('parent_id'),
            creation_timestamp=data.get('creation_timestamp', ''),
            lineage_depth=data.get('lineage_depth', 0)
        )
    
    def compute_parameters_hash(self) -> str:
        """Compute hash of model parameters for identification."""
        # Flatten and concatenate all parameters
        param_list = []
        for v in self.model_state.values():
            if isinstance(v, torch.Tensor):
                param_list.append(v.cpu().numpy().tobytes())
        
        combined = b''.join(param_list)
        return hashlib.sha256(combined).hexdigest()[:16]


class PopulationManager:
    """
    Manages a population of agents for evolutionary training.
    
    Responsibilities:
    - Initialize population
    - Track agent fitness
    - Select top performers
    - Perform repopulation
    """
    
    def __init__(
        self,
        config: Dict,
        algorithm: str,
        seed: Optional[int] = None
    ):
        """
        Initialize population manager.
        
        Args:
            config: Multi-agent configuration
            algorithm: 'ddpg' or 'ppo'
            seed: Random seed
        """
        self.config = config
        self.algorithm = algorithm
        self.seed = seed
        
        # Extract config
        ma_config = config.get('multi_agent', {})
        self.population_size = ma_config.get('population_size', 8)
        self.selection_ratio = ma_config.get('selection_ratio', 0.5)
        self.mutation_sigma = ma_config.get('mutation_sigma', 0.05)
        self.crossover_enabled = ma_config.get('crossover_enabled', False)
        self.elitism_count = ma_config.get('elitism_count', 2)
        self.resample_count = ma_config.get('resample_count', 2)  # Extra exact copies of top performers
        
        # Initialize operators
        self.evo_algo = EvolutionaryAlgorithm(config, seed=seed)
        
        # Population storage
        self.agents: List[Agent] = []
        self.fitness_scores: List[float] = []
        
        # Set seeds
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
    
    def initialize_population(
        self,
        base_hyperparameters: Dict,
        initial_model_state: Optional[Dict[str, Any]] = None
    ) -> List[Agent]:
        """
        Initialize population with new agents.
        
        Args:
            base_hyperparameters: Base hyperparameters for all agents
            initial_model_state: Optional initial model state (e.g., pretrained)
            
        Returns:
            List of initialized agents
        """
        self.agents = []
        self.fitness_scores = []
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for i in range(self.population_size):
            agent = Agent(
                agent_id=f"agent_{i}",
                agent_type=self.algorithm,
                model_state=copy.deepcopy(initial_model_state) if initial_model_state else {},
                hyperparameters=copy.deepcopy(base_hyperparameters),
                fitness=0.0,
                episode_count=0,
                parent_id=None,
                creation_timestamp=timestamp,
                lineage_depth=0
            )
            self.agents.append(agent)
            self.fitness_scores.append(0.0)
        
        return self.agents
    
    def update_fitness(self, agent_id: str, fitness: float):
        """Update fitness score for an agent."""
        for i, agent in enumerate(self.agents):
            if agent.agent_id == agent_id:
                self.fitness_scores[i] = fitness
                agent.fitness = fitness
                break
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None
    
    def get_all_agents(self) -> List[Agent]:
        """Get all agents."""
        return self.agents
    
    def get_fitness_scores(self) -> List[float]:
        """Get all fitness scores."""
        return self.fitness_scores
    
    def get_top_performers(
        self,
        num: Optional[int] = None
    ) -> Tuple[List[Agent], List[float]]:
        """
        Get top-performing agents.
        
        Args:
            num: Number of top agents to return
            
        Returns:
            (top_agents, top_scores)
        """
        if num is None:
            num = max(1, int(len(self.agents) * self.selection_ratio))
        
        # Sort by fitness
        sorted_indices = sorted(
            range(len(self.fitness_scores)),
            key=lambda i: self.fitness_scores[i],
            reverse=True
        )
        
        top_indices = sorted_indices[:num]
        
        top_agents = [self.agents[i] for i in top_indices]
        top_scores = [self.fitness_scores[i] for i in top_indices]
        
        return top_agents, top_scores
    
    def evolve(self) -> EvolutionResult:
        """
        Evolve the population through selection and mutation.
        
        Returns:
            EvolutionResult with details
        """
        # Convert agents to dict format for evolution
        agent_dicts = [a.to_dict() for a in self.agents]
        
        # Evolve using evolutionary algorithm
        new_dicts, result = self.evo_algo.evolve_population(
            agent_dicts,
            self.fitness_scores,
            algorithm=self.algorithm
        )
        
        if not result.success:
            return result
        
        # Update population
        # Keep elites unchanged
        num_elites = self.elitism_count
        new_agents = [self.agents[i] for i in range(num_elites)]
        
        # Create offspring from dicts
        for i, agent_dict in enumerate(new_dicts[num_elites:]):
            agent = Agent.from_dict(agent_dict)
            agent.agent_id = f"agent_{num_elites + i}"
            agent.lineage_depth += 1
            new_agents.append(agent)
        
        self.agents = new_agents
        self.fitness_scores = [a.fitness for a in self.agents]
        
        return result
    
    def repopulate_from_top(
        self,
        top_agents: List[Agent],
        top_scores: List[float]
    ) -> List[Agent]:
        """
        Repopulate using top performers as parents.
        
        Ensures:
        - Each top-K performer has at least one exact copy (to not lose best results)
        - Additional exact copies based on resample_count
        - Remaining slots filled with mutated clones
        
        Args:
            top_agents: List of top-performing agents
            top_scores: Corresponding fitness scores
            
        Returns:
            New population
        """
        mutator = MutationOperator(sigma=self.mutation_sigma, seed=self.seed)
        
        new_population = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Ensure each top-K performer has at least one exact copy (to not lose best results)
        num_top_k = len(top_agents)
        agent_id_counter = 0
        
        # 1. Exact copies: one per top-K performer (must exist for each top-K)
        for i, agent in enumerate(top_agents):
            clone = copy.deepcopy(agent)
            clone.agent_id = f"agent_{agent_id_counter}"
            clone.creation_timestamp = timestamp
            # Note: Keep original fitness for exact copies
            new_population.append(clone)
            agent_id_counter += 1
        
        # 2. Additional exact copies: resample_count extra exact copies from top performers
        for i in range(self.resample_count):
            if agent_id_counter >= self.population_size:
                break
            parent = random.choice(top_agents)
            clone = copy.deepcopy(parent)
            clone.agent_id = f"agent_{agent_id_counter}"
            clone.creation_timestamp = timestamp
            new_population.append(clone)
            agent_id_counter += 1
        
        # 3. Fill remaining slots with mutated clones (parameter cloning + noise/mutation)
        for i in range(self.population_size - agent_id_counter):
            parent = random.choice(top_agents)
            
            # Clone with noise (mutation)
            parent_dict = parent.to_dict()
            offspring_dict, _ = mutator.clone_with_noise(
                parent_dict,
                sigma=self.mutation_sigma,
                param_subset=0.5
            )
            
            offspring = Agent.from_dict(offspring_dict)
            offspring.agent_id = f"agent_{agent_id_counter}"
            offspring.parent_id = parent.agent_id
            offspring.creation_timestamp = timestamp
            offspring.lineage_depth = parent.lineage_depth + 1
            offspring.fitness = 0.0  # Reset fitness for new offspring
            offspring.episode_count = 0
            
            new_population.append(offspring)
            agent_id_counter += 1
        
        self.agents = new_population
        self.fitness_scores = [a.fitness for a in self.agents]
        
        return self.agents


class StageCoordinator:
    """
    Coordinates multi-stage training.
    
    Responsibilities:
    - Load stage definitions
    - Check termination criteria
    - Manage stage transitions
    """
    
    def __init__(self, config: Dict, seed: Optional[int] = None):
        """
        Initialize stage coordinator.
        
        Args:
            config: Multi-agent configuration
            seed: Random seed
        """
        self.config = config
        self.seed = seed
        self.stages = config.get('stage_definitions', [])
        self.num_stages = len(self.stages)
        
        self.current_stage_idx = 0
        self.episode_count = 0
        self.global_episode_offset = 0
        
        # Set seed
        if seed is not None:
            random.seed(seed)
    
    def load_stage(self, stage_idx: int) -> Optional[Dict]:
        """
        Load configuration for a specific stage.
        
        Args:
            stage_idx: Stage index
            
        Returns:
            Stage configuration or None
        """
        if stage_idx < 0 or stage_idx >= self.num_stages:
            return None
        
        self.current_stage_idx = stage_idx
        return self.stages[stage_idx]
    
    def get_current_stage(self) -> Optional[Dict]:
        """Get current stage configuration."""
        if self.current_stage_idx >= self.num_stages:
            return None
        return self.stages[self.current_stage_idx]
    
    def get_stage_hyperparameters(self) -> Dict:
        """Get hyperparameters for current stage."""
        stage = self.get_current_stage()
        if stage:
            return stage.get('hyperparameters', {})
        return {}
    
    def get_stage_environment(self) -> Dict:
        """Get environment configuration for current stage."""
        stage = self.get_current_stage()
        if stage:
            return stage.get('environment', {})
        return {}
    
    def check_termination(
        self,
        episode_count: int,
        success_rate: float
    ) -> Tuple[bool, str]:
        """
        Check if current stage should terminate.
        
        Args:
            episode_count: Number of episodes completed in stage
            success_rate: Recent success rate
            
        Returns:
            (should_terminate, reason)
        """
        stage = self.get_current_stage()
        if stage is None:
            return True, "no_more_stages"
        
        criteria = stage.get('termination_criteria', {})
        min_episodes = criteria.get('min_episodes', 0)
        max_episodes = criteria.get('max_episodes', float('inf'))
        target_success = criteria.get('target_success_rate', 0.0)
        
        # Check max episodes
        if episode_count >= max_episodes:
            return True, f"max_episodes_reached ({max_episodes})"
        
        # Check success rate (only after min episodes)
        if episode_count >= min_episodes and success_rate >= target_success:
            return True, f"success_rate_reached ({success_rate:.2%} >= {target_success:.2%})"
        
        return False, ""
    
    def should_advance(self, episode_count: int, success_rate: float) -> bool:
        """Check if should advance to next stage."""
        terminate, _ = self.check_termination(episode_count, success_rate)
        return terminate
    
    def advance_stage(self) -> bool:
        """
        Advance to next stage.
        
        Returns:
            True if advanced, False if no more stages
        """
        if self.current_stage_idx < self.num_stages - 1:
            self.current_stage_idx += 1
            self.episode_count = 0
            return True
        return False
    
    def reset(self):
        """Reset to first stage."""
        self.current_stage_idx = 0
        self.episode_count = 0
    
    def get_stage_info(self) -> Dict:
        """Get information about current stage."""
        stage = self.get_current_stage()
        if stage is None:
            return {}
        
        criteria = stage.get('termination_criteria', {})
        
        return {
            'stage_id': self.current_stage_idx,
            'stage_name': stage.get('name', f'stage_{self.current_stage_idx}'),
            'episodes_completed': self.episode_count,
            'target_episodes': stage.get('episodes', 0),
            'min_episodes': criteria.get('min_episodes', 0),
            'max_episodes': criteria.get('max_episodes', float('inf')),
            'target_success_rate': criteria.get('target_success_rate', 0.0)
        }


class AgentCheckpointManager:
    """
    Manages agent checkpoints with proper directory structure.
    
    Structure:
    <run_dir>/
    └── checkpoints/
        └── stage_<N>/
            └── agent_<ID>/
                └── model.pt
    """
    
    def __init__(self, run_dir: str):
        """
        Initialize checkpoint manager.
        
        Args:
            run_dir: Base run directory
        """
        self.run_dir = run_dir
        self.checkpoints_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
    
    def get_stage_dir(self, stage_id: int) -> str:
        """Get checkpoint directory for a stage."""
        stage_dir = os.path.join(self.checkpoints_dir, f"stage_{stage_id}")
        os.makedirs(stage_dir, exist_ok=True)
        return stage_dir
    
    def get_agent_dir(self, stage_id: int, agent_id: str) -> str:
        """Get checkpoint directory for an agent."""
        agent_dir = os.path.join(self.get_stage_dir(stage_id), agent_id)
        os.makedirs(agent_dir, exist_ok=True)
        return agent_dir
    
    def save_checkpoint(
        self,
        agent: Agent,
        stage_id: int,
        episode: int,
        is_best: bool = False
    ) -> str:
        """
        Save agent checkpoint.
        
        Args:
            agent: Agent to save
            stage_id: Current stage ID
            episode: Current episode
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        agent_dir = self.get_agent_dir(stage_id, agent.agent_id)
        
        if is_best:
            filename = "model_best.pt"
        else:
            filename = f"model_ep{episode}.pt"
        
        filepath = os.path.join(agent_dir, filename)
        
        checkpoint = {
            'agent_id': agent.agent_id,
            'agent_type': agent.agent_type,
            'model_state': agent.model_state,
            'hyperparameters': agent.hyperparameters,
            'fitness': agent.fitness,
            'episode_count': agent.episode_count,
            'parent_id': agent.parent_id,
            'lineage_depth': agent.lineage_depth,
            'stage_id': stage_id,
            'training_episode': episode,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save checkpoint using configured device
        # Note: Controllers run in Webots with CPU, so this is mainly for consistency
        torch.save(checkpoint, filepath)
        
        return filepath
    
    def load_checkpoint(self, filepath: str) -> Optional[Agent]:
        """
        Load agent from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Agent or None if load fails
        """
        if not os.path.exists(filepath):
            return None
        
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            return Agent(
                agent_id=checkpoint.get('agent_id', 'unknown'),
                agent_type=checkpoint.get('agent_type', 'unknown'),
                model_state=checkpoint.get('model_state', {}),
                hyperparameters=checkpoint.get('hyperparameters', {}),
                fitness=checkpoint.get('fitness', 0.0),
                episode_count=checkpoint.get('episode_count', 0),
                parent_id=checkpoint.get('parent_id'),
                creation_timestamp=checkpoint.get('timestamp', ''),
                lineage_depth=checkpoint.get('lineage_depth', 0)
            )
        except Exception as e:
            print(f"Error loading checkpoint {filepath}: {e}")
            return None
    
    def list_checkpoints(
        self,
        stage_id: Optional[int] = None,
        agent_id: Optional[str] = None
    ) -> List[str]:
        """
        List available checkpoints.
        
        Args:
            stage_id: Optional stage filter
            agent_id: Optional agent filter
            
        Returns:
            List of checkpoint paths
        """
        checkpoints = []
        
        for stage_dir in os.listdir(self.checkpoints_dir):
            if stage_id is not None:
                if not stage_dir.startswith(f"stage_{stage_id}"):
                    continue
            
            stage_path = os.path.join(self.checkpoints_dir, stage_dir)
            if not os.path.isdir(stage_path):
                continue
            
            for agent_dir in os.listdir(stage_path):
                if agent_id is not None:
                    if agent_dir != agent_id:
                        continue
                
                agent_path = os.path.join(stage_path, agent_dir)
                if not os.path.isdir(agent_path):
                    continue
                
                for f in os.listdir(agent_path):
                    if f.endswith('.pt'):
                        checkpoints.append(os.path.join(agent_path, f))
        
        return checkpoints
    
    def get_latest_checkpoint(
        self,
        stage_id: Optional[int] = None,
        agent_id: Optional[str] = None
    ) -> Optional[str]:
        """Get the most recent checkpoint."""
        checkpoints = self.list_checkpoints(stage_id, agent_id)
        
        if not checkpoints:
            return None
        
        checkpoints.sort(key=lambda x: os.path.getmtime(x))
        return checkpoints[-1]
    
    def get_best_checkpoint(
        self,
        stage_id: int,
        agent_id: str
    ) -> Optional[str]:
        """Get best checkpoint for an agent."""
        agent_dir = self.get_agent_dir(stage_id, agent_id)
        best_path = os.path.join(agent_dir, "model_best.pt")
        
        if os.path.exists(best_path):
            return best_path
        
        return None


class MultiAgentTrainer:
    """
    Main orchestrator for multi-agent evolutionary training.
    
    Responsibilities:
    - Setup directories and logging
    - Run training loop
    - Manage stage transitions
    - Handle selection and repopulation
    - Save/load checkpoints
    """
    
    def __init__(
        self,
        controller_dir: str,
        algorithm: str,
        multi_agent_config: Dict,
        seed: Optional[int] = None,
        run_id: Optional[str] = None
    ):
        """
        Initialize multi-agent trainer.
        
        Args:
            controller_dir: Controller directory
            algorithm: 'ddpg' or 'ppo'
            multi_agent_config: Multi-agent configuration
            seed: Random seed
            run_id: Optional run identifier (auto-generated if not provided)
        """
        self.controller_dir = controller_dir
        self.algorithm = algorithm
        self.config = multi_agent_config
        self.seed = seed
        
        # Set seeds
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed) if 'numpy' in sys.modules else None
        
        # Generate run ID if not provided
        if run_id is None:
            timestamp = datetime.now().strftime("%H.%M.%S-%d.%m.%y")
            run_id = f"{algorithm}_{timestamp}"
        
        self.run_id = run_id
        self.run_dir = os.path.join(controller_dir, "runs", run_id)
        
        # Setup directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "stats"), exist_ok=True)
        
        # Initialize components
        self.stage_coordinator = StageCoordinator(multi_agent_config, seed=seed)
        self.checkpoint_manager = AgentCheckpointManager(self.run_dir)
        
        # Get device configuration (use GPU if available and configured)
        device_str = multi_agent_config.get('device', 'cpu')
        if device_str == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Get checkpoint policy
        checkpoint_policy_config = multi_agent_config.get('checkpoint_policy', {})
        if isinstance(checkpoint_policy_config, dict):
            self.checkpoint_policy = checkpoint_policy_config
        else:
            # Fallback for string-based policy
            self.checkpoint_policy = {
                'save_best_only': checkpoint_policy_config == 'best_only',
                'save_frequency': 100,
                'save_all_stages': checkpoint_policy_config != 'best_only',
                'keep_top_k': 5
            }
        
        # Initialize stats logger
        self.stats_logger = HDF5StatsLogger(
            os.path.join(self.run_dir, "stats", "training_stats.h5"),
            mode='w'
        )
        
        # Initialize population manager
        self.population_manager = PopulationManager(
            multi_agent_config, algorithm, seed=seed
        )
        
        # Training state
        self.global_episode = 0
        self.global_timestep = 0
        self.current_stage_id = 0
        self.is_resumed = False
    
    def setup_initial_population(
        self,
        base_hyperparameters: Dict,
        initial_model_state: Optional[Dict[str, Any]] = None
    ):
        """Initialize the population for training."""
        self.population_manager.initialize_population(
            base_hyperparameters,
            initial_model_state
        )
        
        # Log agents
        for agent in self.population_manager.agents:
            agent_info = AgentInfo(
                agent_id=agent.agent_id,
                stage_id=self.current_stage_id,
                agent_type=agent.agent_type,
                parameters_hash=agent.compute_parameters_hash(),
                parent_id=agent.parent_id,
                creation_timestamp=agent.creation_timestamp,
                lineage_depth=agent.lineage_depth
            )
            self.stats_logger.log_agent(agent_info)
    
    def run_training(
        self,
        base_hyperparameters: Dict,
        initial_model_state: Optional[Dict[str, Any]] = None,
        eval_callback: Optional[Callable] = None,
        max_global_episodes: Optional[int] = None
    ) -> Dict:
        """
        Run the complete multi-stage training loop.
        
        Args:
            base_hyperparameters: Initial hyperparameters
            initial_model_state: Optional pretrained model
            eval_callback: Optional evaluation function(agent, stage_id) -> fitness
            max_global_episodes: Optional global episode limit
            
        Returns:
            Training summary dictionary
        """
        print("\n" + "="*70)
        print(f"🚀 MULTI-AGENT {self.algorithm.upper()} TRAINING")
        print(f"   Run ID: {self.run_id}")
        print(f"   Population Size: {self.population_manager.population_size}")
        print(f"   Stages: {self.stage_coordinator.num_stages}")
        print("="*70)
        
        # Initialize population
        self.setup_initial_population(base_hyperparameters, initial_model_state)
        
        # Log initial stage
        stage = self.stage_coordinator.get_current_stage()
        if stage:
            stage_info = StageInfo(
                stage_id=0,
                stage_name=stage.get('name', 'stage_0'),
                start_episode_global=self.global_episode,
                hyperparameters=stage.get('hyperparameters', {}),
                metrics={}
            )
            self.stats_logger.log_stage(stage_info)
        
        # Training loop
        start_time = time.time()
        summary = {
            'run_id': self.run_id,
            'total_episodes': 0,
            'stages_completed': 0,
            'total_time': 0.0,
            'stage_summaries': []
        }
        
        while True:
            # Check global episode limit
            if max_global_episodes is not None:
                if self.global_episode >= max_global_episodes:
                    print(f"\n🎯 Global episode limit reached: {max_global_episodes}")
                    break
            
            # Get current stage
            stage = self.stage_coordinator.get_current_stage()
            if stage is None:
                print("\n✅ All stages completed!")
                break
            
            # Run stage training
            stage_summary = self._train_stage(
                stage,
                eval_callback=eval_callback
            )
            
            summary['stage_summaries'].append(stage_summary)
            summary['total_episodes'] += stage_summary['episodes_completed']
            summary['stages_completed'] += 1
            
            # Log stage completion
            self.stats_logger.update_stage(
                stage_id=self.current_stage_id,
                end_episode_global=self.global_episode - 1,
                status='completed',
                metrics=stage_summary
            )
            
            # Advance to next stage
            if not self.stage_coordinator.advance_stage():
                break
            
            self.current_stage_id += 1
            
            # Log new stage
            next_stage = self.stage_coordinator.get_current_stage()
            if next_stage:
                stage_info = StageInfo(
                    stage_id=self.current_stage_id,
                    stage_name=next_stage.get('name', f'stage_{self.current_stage_id}'),
                    start_episode_global=self.global_episode,
                    hyperparameters=next_stage.get('hyperparameters', {}),
                    metrics={}
                )
                self.stats_logger.log_stage(stage_info)
        
        summary['total_time'] = time.time() - start_time
        
        # Print summary
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE")
        print(f"   Total Episodes: {summary['total_episodes']}")
        print(f"   Stages Completed: {summary['stages_completed']}")
        print(f"   Total Time: {summary['total_time']/60:.1f} minutes")
        print(f"   Run Directory: {self.run_dir}")
        print("="*70)
        
        # Save config
        config_path = os.path.join(self.run_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                'run_id': self.run_id,
                'algorithm': self.algorithm,
                'config': self.config,
                'seed': self.seed,
                'summary': summary
            }, f, indent=2, default=str)
        
        self.stats_logger.close()
        
        return summary
    
    def _train_stage(
        self,
        stage: Dict,
        eval_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Train for one stage.
        
        Args:
            stage: Stage configuration
            eval_callback: Optional evaluation function
            
        Returns:
            Stage summary
        """
        stage_id = self.current_stage_id
        stage_name = stage.get('name', f'stage_{stage_id}')
        target_episodes = stage.get('episodes', 1000)
        criteria = stage.get('termination_criteria', {})
        
        print(f"\n{'='*70}")
        print(f"📊 STAGE {stage_id}: {stage_name.upper()}")
        print(f"   Target Episodes: {target_episodes}")
        print(f"   Population: {len(self.population_manager.agents)}")
        print(f"{'='*70}")
        
        stage_episodes = 0
        best_fitness = -float('inf')
        best_agent_id = None
        
        # Episode tracking
        episode_rewards = []
        success_history = []
        
        # Create progress bar
        pbar = tqdm(total=target_episodes, desc=f"Stage {stage_id}")
        
        while stage_episodes < target_episodes:
            # Check termination criteria
            success_rate = np.mean(success_history[-100:]) if len(success_history) >= 100 else np.mean(success_history) if success_history else 0.0
            
            should_terminate, reason = self.stage_coordinator.check_termination(
                stage_episodes, success_rate
            )
            
            if should_terminate:
                print(f"\n🎯 Stage {stage_id} termination: {reason}")
                break
            
            # Launch all agents in parallel (one Webots instance per agent)
            print(f"\n🚀 Launching {len(self.population_manager.agents)} agents in parallel...")
            agent_processes = []
            
            for agent_idx, agent in enumerate(self.population_manager.agents):
                if eval_callback is not None:
                    # If callback provided, run synchronously
                    fitness = eval_callback(agent, stage_id)
                    self.population_manager.update_fitness(agent.agent_id, fitness)
                    episode_rewards.append(fitness)
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_agent_id = agent.agent_id
                    
                    self.global_episode += 1
                    stage_episodes += 1
                    pbar.update(1)
                else:
                    # Launch Webots process for this agent
                    process_info = self._launch_agent_episode(agent, stage_id)
                    agent_processes.append(process_info)
            
            # Wait for all parallel processes to complete
            if agent_processes:
                print(f"⏳ Waiting for {len(agent_processes)} agents to complete episodes...")
                
                # Wait for all processes and collect results
                agent_results = {}
                for agent_id, process, checkpoint_path, results_file in agent_processes:
                    try:
                        stdout, stderr = process.communicate()
                        
                        if process.returncode != 0:
                            print(f"⚠️  Webots returned non-zero exit code {process.returncode} for agent {agent_id}")
                            if stderr:
                                print(f"Error: {stderr.decode('utf-8')[:500]}")
                        
                        # Read results file
                        fitness = 0.0
                        if os.path.exists(results_file):
                            try:
                                with open(results_file, 'r') as f:
                                    results = json.load(f)
                                
                                fitness = float(results.get('total_reward', 0.0))
                                episode_success = results.get('success', False)
                                episode_steps = results.get('steps', 0)
                                
                                # Log episode to stats
                                episode_info = EpisodeInfo(
                                    global_episode_id=self.global_episode,
                                    stage_id=stage_id,
                                    local_episode_id=results.get('local_episode_id', 0),
                                    total_steps=episode_steps,
                                    total_reward=fitness,
                                    success=episode_success,
                                    termination_reason=results.get('termination_reason', 'normal'),
                                    agent_id=agent_id,
                                    start_idx=results.get('start_idx', 0),
                                    end_idx=results.get('end_idx', 0)
                                )
                                self.stats_logger.log_episode(episode_info)
                                
                                # Update global timestep counter
                                self.global_timestep += episode_steps
                                
                                # Load updated checkpoint if available
                                updated_checkpoint = checkpoint_path.replace('.pt', '_updated.pt')
                                if os.path.exists(updated_checkpoint):
                                    try:
                                        updated = torch.load(updated_checkpoint, map_location='cpu')
                                        agent = self.population_manager.get_agent(agent_id)
                                        if agent:
                                            agent.model_state = updated
                                    except Exception as e:
                                        print(f"⚠️  Could not load updated checkpoint for {agent_id}: {e}")
                                
                                agent_results[agent_id] = fitness
                                episode_rewards.append(fitness)
                                success_history.append(1 if episode_success else 0)
                                
                                if fitness > best_fitness:
                                    best_fitness = fitness
                                    best_agent_id = agent_id
                                
                                # Update fitness
                                self.population_manager.update_fitness(agent_id, fitness)
                                
                                # Increment counters
                                self.global_episode += 1
                                stage_episodes += 1
                                
                            except Exception as e:
                                print(f"⚠️  Could not read results file {results_file}: {e}")
                                agent_results[agent_id] = 0.0
                                self.population_manager.update_fitness(agent_id, 0.0)
                                self.global_episode += 1
                                stage_episodes += 1
                        else:
                            print(f"⚠️  Results file not found: {results_file} for agent {agent_id}")
                            agent_results[agent_id] = 0.0
                            self.population_manager.update_fitness(agent_id, 0.0)
                            self.global_episode += 1
                            stage_episodes += 1
                            
                    except Exception as e:
                        print(f"❌ Error waiting for agent {agent_id}: {e}")
                        agent_results[agent_id] = 0.0
                        self.population_manager.update_fitness(agent_id, 0.0)
                        self.global_episode += 1
                        stage_episodes += 1
                
                # Update progress bar
                pbar.update(len(agent_processes))
                pbar.set_postfix({
                    'ep': self.global_episode,
                    'best': f'{best_fitness:.2f}',
                    'rate': f'{success_rate:.1%}'
                })
                
                print(f"✅ Completed parallel training round: {len(agent_results)} agents finished")
            
            # Log stage metrics periodically
            if stage_episodes % 50 == 0:
                recent_rewards = episode_rewards[-100:]
                self.stats_logger.log_stage_summary(
                    stage_id=stage_id,
                    mean_reward=np.mean(recent_rewards),
                    success_rate=success_rate,
                    mean_episode_length=np.mean([30] * min(100, len(recent_rewards))),
                    window_size=100
                )
            
            # Perform evolution after each generation (all agents completed one episode)
            # This happens every population_size episodes
            if stage_episodes % len(self.population_manager.agents) == 0 and stage_episodes > 0:
                print(f"\n🔄 Evolving population after {stage_episodes} episodes...")
                evolution_result = self.population_manager.evolve()
                print(f"   {evolution_result.message}")
        
        pbar.close()
        
        # Save checkpoints based on policy
        save_best_only = self.checkpoint_policy.get('save_best_only', False)
        save_all_stages = self.checkpoint_policy.get('save_all_stages', True)
        
        if best_agent_id:
            best_agent = self.population_manager.get_agent(best_agent_id)
            if best_agent:
                # Always save best agent
                self.checkpoint_manager.save_checkpoint(
                    best_agent, stage_id, self.global_episode, is_best=True
                )
        
        # Save all agents if policy allows (not best_only)
        if not save_best_only and save_all_stages:
            keep_top_k = self.checkpoint_policy.get('keep_top_k', 5)
            top_agents, _ = self.population_manager.get_top_performers(num=keep_top_k)
            for agent in top_agents:
                if agent.agent_id != best_agent_id:  # Don't save best twice
                    self.checkpoint_manager.save_checkpoint(
                        agent, stage_id, self.global_episode, is_best=False
                    )
        
        # Final evolution after stage completion
        print(f"\n🔄 Final population evolution after stage {stage_id}...")
        result = self.population_manager.evolve()
        print(f"   {result.message}")
        
        # Return stage summary
        return {
            'stage_id': stage_id,
            'stage_name': stage_name,
            'episodes_completed': stage_episodes,
            'mean_reward': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            'best_reward': float(best_fitness),
            'final_success_rate': float(success_rate) if success_history else 0.0,
            'evolution_result': result.message
        }
    
    def _find_webots(self) -> Optional[str]:
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
    
    def _launch_agent_episode(
        self,
        agent: Agent,
        stage_id: int
    ) -> Tuple[str, subprocess.Popen, str, str]:
        """
        Launch a Webots instance for an agent to train one episode (non-blocking).
        
        Args:
            agent: Agent to evaluate
            stage_id: Current stage ID
            
        Returns:
            Tuple of (agent_id, process, checkpoint_path, results_file)
        """
        # Get stage configuration
        stage = self.stage_coordinator.get_current_stage()
        if not stage:
            # Return dummy process if stage not found
            dummy_process = subprocess.Popen(['true'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            dummy_process.communicate()
            temp_checkpoint_dir = os.path.join(self.run_dir, "temp_checkpoints", f"stage_{stage_id}", agent.agent_id)
            temp_checkpoint_path = os.path.join(temp_checkpoint_dir, "agent_checkpoint.pt")
            results_file = os.path.join(temp_checkpoint_dir, "episode_results.json")
            return (agent.agent_id, dummy_process, temp_checkpoint_path, results_file)
        
        # Save agent checkpoint to temp location
        temp_checkpoint_dir = os.path.join(self.run_dir, "temp_checkpoints", f"stage_{stage_id}", agent.agent_id)
        os.makedirs(temp_checkpoint_dir, exist_ok=True)
        temp_checkpoint_path = os.path.join(temp_checkpoint_dir, "agent_checkpoint.pt")
        
        # Convert Agent to checkpoint format compatible with controller
        # If model_state is empty (new agent), save a marker checkpoint
        # If model_state contains checkpoint data (has state_dicts), use it directly
        if agent.model_state and len(agent.model_state) > 0 and 'actor_state_dict' in agent.model_state:
            # Model state already in controller format, use directly
            agent_checkpoint = agent.model_state.copy()
            # Ensure required metadata is present
            agent_checkpoint['agent_id'] = agent.agent_id
            agent_checkpoint['parent_id'] = agent.parent_id
            agent_checkpoint['lineage_depth'] = agent.lineage_depth
            agent_checkpoint['creation_timestamp'] = agent.creation_timestamp
            torch.save(agent_checkpoint, temp_checkpoint_path)
        else:
            # New agent - save marker checkpoint that signals "create new agent"
            agent_checkpoint = {
                'obs_dim': 10,  # Default, will be updated by controller
                'act_dim': 10,  # Default, will be updated by controller
                'agent_id': agent.agent_id,
                'parent_id': agent.parent_id,
                'lineage_depth': agent.lineage_depth,
                'creation_timestamp': agent.creation_timestamp,
                'config': {},  # Empty config signals new agent
                'is_new_agent': True  # Explicit marker for new agent
            }
            torch.save(agent_checkpoint, temp_checkpoint_path)
        
        # Determine world file path
        world_file = PROJECT_ROOT / "worlds" / f"robotis_op3_{self.algorithm}.wbt"
        if not world_file.exists():
            print(f"❌ Error: World file not found: {world_file}")
            # Return dummy process if world file not found
            dummy_process = subprocess.Popen(['true'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            dummy_process.communicate()
            return (agent.agent_id, dummy_process, temp_checkpoint_path, results_file)
        
        # Create results file path
        results_file = os.path.join(temp_checkpoint_dir, "episode_results.json")
        if os.path.exists(results_file):
            os.remove(results_file)
        
        # Prepare environment variables for controller
        env = os.environ.copy()
        env['RL_TRAIN'] = 'true'
        env['RL_ALGORITHM'] = self.algorithm
        env['RL_MULTI_AGENT'] = 'true'
        env['RL_AGENT_ID'] = agent.agent_id
        env['RL_STAGE_ID'] = str(stage_id)
        env['RL_GLOBAL_EPISODE'] = str(self.global_episode)
        env['RL_CHECKPOINT_PATH'] = temp_checkpoint_path
        env['RL_RESULTS_FILE'] = results_file
        env['RL_EPISODES_PER_RUN'] = '1'  # Train 1 episode per Webots launch
        env['RL_RUN_DIR'] = str(self.run_dir)
        
        # Add stage hyperparameters
        stage_hparams = stage.get('hyperparameters', {})
        for key, value in stage_hparams.items():
            env[f'RL_HP_{key.upper()}'] = str(value)
        
        # Add stage environment config (goal angles, etc.)
        stage_env = stage.get('environment', {})
        if stage_env:
            env['RL_STAGE_ENV'] = json.dumps(stage_env)
        
        # Find Webots
        webots_path = self._find_webots()
        if not webots_path:
            print("❌ Error: Webots executable not found!")
            # Return dummy process if Webots not found
            dummy_process = subprocess.Popen(['true'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            dummy_process.communicate()
            return (agent.agent_id, dummy_process, temp_checkpoint_path, results_file)
        
        # Launch Webots in fast mode (no rendering) - non-blocking
        cmd = [webots_path, "--mode=fast", "--stdout", "--stderr", "--batch", "--no-rendering", str(world_file.absolute())]
        
        try:
            # Launch Webots process (non-blocking)
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(PROJECT_ROOT)
            )
            
            return (agent.agent_id, process, temp_checkpoint_path, results_file)
                
        except Exception as e:
            print(f"❌ Error launching Webots for agent {agent.agent_id}: {e}")
            import traceback
            traceback.print_exc()
            # Return a dummy process that has already completed with error
            import subprocess as sp
            dummy_process = sp.Popen(['true'], stdout=sp.PIPE, stderr=sp.PIPE)
            dummy_process.communicate()
            return (agent.agent_id, dummy_process, temp_checkpoint_path, results_file)
    
    def save_state(self, filepath: str):
        """Save complete training state for resuming."""
        state = {
            'run_id': self.run_id,
            'algorithm': self.algorithm,
            'config': self.config,
            'seed': self.seed,
            'global_episode': self.global_episode,
            'global_timestep': self.global_timestep,
            'current_stage_id': self.current_stage_id,
            'is_resumed': True,
            'agents': [a.to_dict() for a in self.population_manager.agents],
            'fitness_scores': self.population_manager.fitness_scores
        }
        
        torch.save(state, filepath)
    
    @classmethod
    def load_state(
        cls,
        controller_dir: str,
        run_dir: str,
        algorithm: str,
        checkpoint_path: str
    ) -> 'MultiAgentTrainer':
        """
        Load training state from checkpoint.
        
        Args:
            controller_dir: Controller directory
            run_dir: Run directory
            algorithm: 'ddpg' or 'ppo'
            checkpoint_path: Path to state checkpoint
            
        Returns:
            MultiAgentTrainer with loaded state
        """
        state = torch.load(checkpoint_path, map_location='cpu')
        
        # Create trainer
        trainer = cls(
            controller_dir=controller_dir,
            algorithm=algorithm,
            multi_agent_config=state['config'],
            seed=state['seed'],
            run_id=state['run_id']
        )
        
        trainer.run_dir = run_dir
        trainer.global_episode = state['global_episode']
        trainer.global_timestep = state['global_timestep']
        trainer.current_stage_id = state['current_stage_id']
        trainer.is_resumed = True
        
        # Load agents
        trainer.population_manager.agents = [
            Agent.from_dict(a) for a in state['agents']
        ]
        trainer.population_manager.fitness_scores = state['fitness_scores']
        
        return trainer
    
    def get_progress(self) -> Dict:
        """Get current training progress."""
        return {
            'run_id': self.run_id,
            'global_episode': self.global_episode,
            'current_stage': self.current_stage_id,
            'stage_info': self.stage_coordinator.get_stage_info(),
            'population_size': len(self.population_manager.agents),
            'best_fitness': max(self.population_manager.fitness_scores) if self.population_manager.fitness_scores else 0.0
        }


def create_trainer(
    controller_dir: str,
    algorithm: str,
    seed: Optional[int] = None,
    run_id: Optional[str] = None
) -> MultiAgentTrainer:
    """
    Factory function to create multi-agent trainer.
    
    Args:
        controller_dir: Controller directory
        algorithm: 'ddpg' or 'ppo'
        seed: Random seed
        run_id: Optional run identifier
        
    Returns:
        Configured MultiAgentTrainer
    """
    # Load multi-agent config
    config_path = os.path.join(controller_dir, f"multi_agent_config.json")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return MultiAgentTrainer(
        controller_dir=controller_dir,
        algorithm=algorithm,
        multi_agent_config=config,
        seed=seed,
        run_id=run_id
    )


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description="Multi-Agent Training")
    parser.add_argument('--controller', type=str, required=True,
                       choices=['controllers/op3_ddpg', 'controllers/op3_ppo'])
    parser.add_argument('--alg', type=str, required=True, choices=['ddpg', 'ppo'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=10000)
    
    args = parser.parse_args()
    
    # Create and run trainer
    trainer = create_trainer(args.controller, args.alg, seed=args.seed)
    trainer.run_training({}, max_global_episodes=args.episodes)

