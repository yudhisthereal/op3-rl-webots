#!/usr/bin/env python3
"""
Evolutionary Operators for Multi-Agent Evolutionary Training.

Provides selection, mutation, crossover, and hyperparameter mutation operators
for population-based reinforcement learning.
"""

import numpy as np
import torch
import copy
import random
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass


@dataclass
class EvolutionResult:
    """Result of an evolutionary operation."""
    success: bool
    message: str
    details: Optional[Dict] = None


class SelectionOperator:
    """
    Selection operators for evolutionary algorithms.
    
    Supports:
    - Tournament selection
    - Roulette wheel selection
    - Rank-based selection
    - Elitism (keep top performers)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize selection operator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def tournament_select(
        self,
        population: List[Dict],
        fitness_scores: List[float],
        tournament_size: int = 3,
        num_select: int = 1,
        maximize: bool = True
    ) -> List[Dict]:
        """
        Tournament selection: randomly select individuals and choose the best.
        
        Args:
            population: List of agent dictionaries
            fitness_scores: Corresponding fitness values
            tournament_size: Number of individuals per tournament
            num_select: Number of individuals to select
            maximize: True for maximizing fitness, False for minimizing
            
        Returns:
            List of selected individuals
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        selected = []
        indices = list(range(len(population)))
        
        for _ in range(num_select):
            # Randomly sample tournament participants
            if tournament_size >= len(population):
                participants = indices.copy()
            else:
                participants = random.sample(indices, tournament_size)
            
            # Get fitness scores for participants
            participant_fitness = [(i, fitness_scores[i]) for i in participants]
            
            # Select best (or worst if minimizing)
            if maximize:
                best_idx = max(participant_fitness, key=lambda x: x[1])[0]
            else:
                best_idx = min(participant_fitness, key=lambda x: x[1])[0]
            
            # Deep copy to avoid reference issues
            selected.append(copy.deepcopy(population[best_idx]))
        
        return selected
    
    def roulette_wheel_select(
        self,
        population: List[Dict],
        fitness_scores: List[float],
        num_select: int = 1,
        maximize: bool = True
    ) -> List[Dict]:
        """
        Roulette wheel (fitness proportionate) selection.
        
        Args:
            population: List of agent dictionaries
            fitness_scores: Corresponding fitness values
            num_select: Number of individuals to select
            maximize: True for maximizing fitness, False for minimizing
            
        Returns:
            List of selected individuals
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        if len(population) == 0:
            return []
        
        # Convert to positive values for roulette wheel
        if maximize:
            # Higher is better: use fitness directly
            min_fitness = min(fitness_scores)
            if min_fitness < 0:
                adjusted = [f - min_fitness + 1e-6 for f in fitness_scores]
            else:
                adjusted = [f + 1e-6 for f in fitness_scores]
        else:
            # Lower is better: invert
            max_fitness = max(fitness_scores)
            adjusted = [max_fitness - f + 1e-6 for f in fitness_scores]
        
        # Normalize to probabilities
        total = sum(adjusted)
        probabilities = [p / total for p in adjusted]
        
        selected = []
        for _ in range(num_select):
            idx = np.random.choice(len(population), p=probabilities)
            selected.append(copy.deepcopy(population[idx]))
        
        return selected
    
    def rank_select(
        self,
        population: List[Dict],
        fitness_scores: List[float],
        num_select: int = 1,
        maximize: bool = True
    ) -> List[Dict]:
        """
        Rank-based selection: select based on rank rather than raw fitness.
        
        Args:
            population: List of agent dictionaries
            fitness_scores: Corresponding fitness values
            num_select: Number of individuals to select
            maximize: True for maximizing fitness, False for minimizing
            
        Returns:
            List of selected individuals
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        if len(population) == 0:
            return []
        
        # Sort by fitness and assign ranks
        sorted_pairs = sorted(
            enumerate(fitness_scores),
            key=lambda x: x[1],
            reverse=maximize
        )
        
        ranks = [0] * len(population)
        for rank, (original_idx, _) in enumerate(sorted_pairs):
            ranks[original_idx] = rank + 1  # 1-indexed ranks
        
        # Use ranks as selection probabilities (higher rank = higher probability)
        total_rank = sum(ranks)
        probabilities = [r / total_rank for r in ranks]
        
        selected = []
        for _ in range(num_select):
            idx = np.random.choice(len(population), p=probabilities)
            selected.append(copy.deepcopy(population[idx]))
        
        return selected
    
    def elitism_select(
        self,
        population: List[Dict],
        fitness_scores: List[float],
        num_elites: int = 1,
        maximize: bool = True
    ) -> List[Dict]:
        """
        Elitism: directly select top-performing individuals.
        
        Args:
            population: List of agent dictionaries
            fitness_scores: Corresponding fitness values
            num_elites: Number of top individuals to select
            maximize: True for maximizing fitness, False for minimizing
            
        Returns:
            List of elite individuals
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        if len(population) == 0:
            return []
        
        # Sort by fitness
        sorted_pairs = sorted(
            enumerate(fitness_scores),
            key=lambda x: x[1],
            reverse=maximize
        )
        
        # Select top num_elites
        num_elites = min(num_elites, len(population))
        selected = []
        for i in range(num_elites):
            original_idx = sorted_pairs[i][0]
            selected.append(copy.deepcopy(population[original_idx]))
        
        return selected


class MutationOperator:
    """
    Mutation operators for evolutionary algorithms.
    
    Supports:
    - Gaussian noise mutation
    - Parameter cloning (copy without changes)
    - Clone with noise
    """
    
    def __init__(self, sigma: float = 0.05, seed: Optional[int] = None):
        """
        Initialize mutation operator.
        
        Args:
            sigma: Standard deviation for Gaussian noise
            seed: Random seed for reproducibility
        """
        self.sigma = sigma
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def gaussian_mutate(
        self,
        agent: Dict,
        sigma: Optional[float] = None,
        clip_value: float = 0.5
    ) -> Tuple[Dict, EvolutionResult]:
        """
        Apply Gaussian noise to agent parameters.
        
        Args:
            agent: Agent dictionary containing model state dict
            sigma: Noise standard deviation (uses default if None)
            clip_value: Maximum absolute value for clipping
            
        Returns:
            (mutated_agent, result)
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        sigma = sigma if sigma is not None else self.sigma
        
        try:
            mutated_agent = copy.deepcopy(agent)
            
            if 'model_state' not in mutated_agent:
                return agent, EvolutionResult(
                    success=False,
                    message="No model_state in agent",
                    details=None
                )
            
            model_state = mutated_agent['model_state']
            
            for key in model_state.keys():
                if isinstance(model_state[key], torch.Tensor):
                    # Skip if not a floating point tensor
                    if model_state[key].dtype.is_floating_point:
                        noise = torch.randn_like(model_state[key]) * sigma
                        model_state[key] = model_state[key] + noise
                        # Clip to prevent extreme values
                        if clip_value > 0:
                            model_state[key] = torch.clamp(
                                model_state[key], 
                                -clip_value, 
                                clip_value
                            )
            
            mutated_agent['mutation_info'] = {
                'type': 'gaussian',
                'sigma': sigma,
                'parent_id': agent.get('agent_id', 'unknown')
            }
            
            return mutated_agent, EvolutionResult(
                success=True,
                message=f"Applied Gaussian mutation with sigma={sigma}",
                details={'sigma': sigma}
            )
            
        except Exception as e:
            return agent, EvolutionResult(
                success=False,
                message=f"Mutation failed: {str(e)}",
                details=None
            )
    
    def clone(
        self,
        agent: Dict
    ) -> Tuple[Dict, EvolutionResult]:
        """
        Create an exact copy of an agent.
        
        Args:
            agent: Agent dictionary
            
        Returns:
            (cloned_agent, result)
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        try:
            cloned = copy.deepcopy(agent)
            
            # Generate new agent ID
            base_id = cloned.get('agent_id', 'unknown')
            cloned['agent_id'] = f"{base_id}_copy"
            cloned['parent_id'] = base_id
            cloned['mutation_info'] = {
                'type': 'clone',
                'parent_id': base_id
            }
            
            return cloned, EvolutionResult(
                success=True,
                message=f"Cloned agent from {base_id}",
                details={'parent_id': base_id}
            )
            
        except Exception as e:
            return agent, EvolutionResult(
                success=False,
                message=f"Clone failed: {str(e)}",
                details=None
            )
    
    def clone_with_noise(
        self,
        agent: Dict,
        sigma: Optional[float] = None,
        param_subset: float = 1.0
    ) -> Tuple[Dict, EvolutionResult]:
        """
        Clone agent and apply noise to a subset of parameters.
        
        Args:
            agent: Agent dictionary
            sigma: Noise standard deviation
            param_subset: Fraction of parameters to mutate (0.0 to 1.0)
            
        Returns:
            (mutated_clone, result)
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        sigma = sigma if sigma is not None else self.sigma
        
        try:
            # First clone
            cloned, _ = self.clone(agent)
            
            if 'model_state' not in cloned:
                return agent, EvolutionResult(
                    success=False,
                    message="No model_state in agent",
                    details=None
                )
            
            model_state = cloned['model_state']
            param_names = list(model_state.keys())
            
            # Select subset of parameters to mutate
            num_to_mutate = max(1, int(len(param_names) * param_subset))
            params_to_mutate = random.sample(param_names, num_to_mutate)
            
            for key in params_to_mutate:
                if isinstance(model_state[key], torch.Tensor):
                    if model_state[key].dtype.is_floating_point:
                        noise = torch.randn_like(model_state[key]) * sigma
                        model_state[key] = model_state[key] + noise
            
            cloned['mutation_info'] = {
                'type': 'clone_with_noise',
                'sigma': sigma,
                'params_mutated': len(params_to_mutate),
                'parent_id': agent.get('agent_id', 'unknown')
            }
            
            return cloned, EvolutionResult(
                success=True,
                message=f"Cloned with noise: {num_to_mutate} params mutated",
                details={'sigma': sigma, 'params_mutated': num_to_mutate}
            )
            
        except Exception as e:
            return agent, EvolutionResult(
                success=False,
                message=f"Clone with noise failed: {str(e)}",
                details=None
            )


class CrossoverOperator:
    """
    Crossover (recombination) operators.
    
    Note: Crossover is optional and disabled by default.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def uniform_crossover(
        self,
        agent1: Dict,
        agent2: Dict,
        crossover_rate: float = 0.5
    ) -> Tuple[Dict, EvolutionResult]:
        """
        Uniform crossover: randomly choose genes from each parent.
        
        Args:
            agent1: First parent agent
            agent2: Second parent agent
            crossover_rate: Probability of selecting from agent1
            
        Returns:
            (offspring, result)
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(seed)
        
        try:
            offspring = copy.deepcopy(agent1)
            
            if 'model_state' not in agent1 or 'model_state' not in agent2:
                return agent1, EvolutionResult(
                    success=False,
                    message="Both agents must have model_state",
                    details=None
                )
            
            state1 = agent1['model_state']
            state2 = agent2['model_state']
            
            for key in state1.keys():
                if key in state2:
                    if (isinstance(state1[key], torch.Tensor) and 
                        isinstance(state2[key], torch.Tensor) and
                        state1[key].shape == state2[key].shape):
                        
                        # Create crossover mask
                        mask = torch.rand_like(state1[key]) < crossover_rate
                        
                        # For tensors, blend values
                        if state1[key].dtype.is_floating_point:
                            # Weighted average
                            alpha = torch.rand_like(state1[key])
                            state1[key] = alpha * state1[key] + (1 - alpha) * state2[key]
            
            offspring['crossover_info'] = {
                'type': 'uniform',
                'parent1': agent1.get('agent_id', 'unknown'),
                'parent2': agent2.get('agent_id', 'unknown')
            }
            
            return offspring, EvolutionResult(
                success=True,
                message=f"Crossover between agents",
                details={'parent1': agent1.get('agent_id'), 'parent2': agent2.get('agent_id')}
            )
            
        except Exception as e:
            return agent1, EvolutionResult(
                success=False,
                message=f"Crossover failed: {str(e)}",
                details=None
            )
    
    def arithmetic_crossover(
        self,
        agent1: Dict,
        agent2: Dict,
        alpha: float = 0.5
    ) -> Tuple[Dict, EvolutionResult]:
        """
        Arithmetic crossover: weighted average of parameters.
        
        Args:
            agent1: First parent agent
            agent2: Second parent agent
            alpha: Weight for agent1 (1-alpha for agent2)
            
        Returns:
            (offspring, result)
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        try:
            offspring = copy.deepcopy(agent1)
            
            if 'model_state' not in agent1 or 'model_state' not in agent2:
                return agent1, EvolutionResult(
                    success=False,
                    message="Both agents must have model_state",
                    details=None
                )
            
            state1 = agent1['model_state']
            state2 = agent2['model_state']
            
            for key in state1.keys():
                if key in state2:
                    if (isinstance(state1[key], torch.Tensor) and 
                        isinstance(state2[key], torch.Tensor) and
                        state1[key].shape == state2[key].shape and
                        state1[key].dtype.is_floating_point):
                        
                        # Arithmetic mean
                        offspring['model_state'][key] = (
                            alpha * state1[key] + (1 - alpha) * state2[key]
                        )
            
            offspring['crossover_info'] = {
                'type': 'arithmetic',
                'alpha': alpha,
                'parent1': agent1.get('agent_id', 'unknown'),
                'parent2': agent2.get('agent_id', 'unknown')
            }
            
            return offspring, EvolutionResult(
                success=True,
                message=f"Arithmetic crossover with alpha={alpha}",
                details={'alpha': alpha}
            )
            
        except Exception as e:
            return agent1, EvolutionResult(
                success=False,
                message=f"Crossover failed: {str(e)}",
                details=None
            )


class HyperparameterMutator:
    """
    Mutate hyperparameters for evolutionary HPO.
    
    Supports:
    - Learning rate perturbation
    - Gamma/clip_epsilon perturbation
    - Tau perturbation (DDPG-specific)
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def mutate_learning_rate(
        self,
        config: Dict,
        sigma_factor: float = 0.1,
        bounds: Tuple[float, float] = (1e-5, 1e-2)
    ) -> Tuple[Dict, EvolutionResult]:
        """
        Mutate learning rate with log-normal perturbation.
        
        Args:
            config: Agent config with hyperparameters
            sigma_factor: Factor for multiplicative noise
            bounds: (min, max) bounds for learning rate
            
        Returns:
            (mutated_config, result)
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        try:
            mutated = copy.deepcopy(config)
            
            # Determine which learning rate to mutate
            lr_key = None
            if 'actor_lr' in mutated:
                lr_key = 'actor_lr'
            elif 'learning_rate' in mutated:
                lr_key = 'learning_rate'
            else:
                return config, EvolutionResult(
                    success=False,
                    message="No learning rate found in config",
                    details=None
                )
            
            current_lr = mutated['hyperparameters'].get(lr_key, mutated.get(lr_key))
            if current_lr is None:
                current_lr = mutated.get(lr_key, 0.001)
            
            # Log-normal perturbation
            log_lr = np.log(current_lr)
            new_log_lr = log_lr + np.random.randn() * sigma_factor
            new_lr = np.exp(new_log_lr)
            
            # Clamp to bounds
            new_lr = float(np.clip(new_lr, bounds[0], bounds[1]))
            
            # Update config
            if 'hyperparameters' in mutated:
                if lr_key in mutated['hyperparameters']:
                    mutated['hyperparameters'][lr_key] = new_lr
                else:
                    mutated['hyperparameters'][lr_key] = new_lr
            else:
                mutated[lr_key] = new_lr
            
            mutated['hyperparameter_mutation'] = {
                'type': 'learning_rate',
                'old_value': current_lr,
                'new_value': new_lr
            }
            
            return mutated, EvolutionResult(
                success=True,
                message=f"Mutated LR: {current_lr:.2e} -> {new_lr:.2e}",
                details={'old': current_lr, 'new': new_lr}
            )
            
        except Exception as e:
            return config, EvolutionResult(
                success=False,
                message=f"LR mutation failed: {str(e)}",
                details=None
            )
    
    def mutate_discount_factor(
        self,
        config: Dict,
        sigma: float = 0.01,
        bounds: Tuple[float, float] = (0.95, 0.999)
    ) -> Tuple[Dict, EvolutionResult]:
        """
        Mutate discount factor (gamma).
        
        Args:
            config: Agent config with hyperparameters
            sigma: Standard deviation for Gaussian noise
            bounds: (min, max) bounds for gamma
            
        Returns:
            (mutated_config, result)
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        try:
            mutated = copy.deepcopy(config)
            
            current_gamma = mutated.get('gamma', 0.99)
            if 'hyperparameters' in mutated:
                current_gamma = mutated['hyperparameters'].get('gamma', current_gamma)
            
            # Gaussian perturbation
            new_gamma = current_gamma + np.random.randn() * sigma
            new_gamma = float(np.clip(new_gamma, bounds[0], bounds[1]))
            
            if 'hyperparameters' in mutated:
                mutated['hyperparameters']['gamma'] = new_gamma
            else:
                mutated['gamma'] = new_gamma
            
            mutated['hyperparameter_mutation'] = {
                'type': 'gamma',
                'old_value': current_gamma,
                'new_value': new_gamma
            }
            
            return mutated, EvolutionResult(
                success=True,
                message=f"Mutated gamma: {current_gamma:.4f} -> {new_gamma:.4f}",
                details={'old': current_gamma, 'new': new_gamma}
            )
            
        except Exception as e:
            return config, EvolutionResult(
                success=False,
                message=f"Gamma mutation failed: {str(e)}",
                details=None
            )
    
    def mutate_clip_epsilon(
        self,
        config: Dict,
        sigma: float = 0.02,
        bounds: Tuple[float, float] = (0.05, 0.3)
    ) -> Tuple[Dict, EvolutionResult]:
        """
        Mutate PPO clip epsilon.
        
        Args:
            config: Agent config with hyperparameters
            sigma: Standard deviation for Gaussian noise
            bounds: (min, max) bounds
            
        Returns:
            (mutated_config, result)
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        try:
            mutated = copy.deepcopy(config)
            
            current_clip = 0.2  # Default
            if 'clip_epsilon' in config:
                current_clip = config['clip_epsilon']
            elif 'hyperparameters' in config:
                current_clip = config['hyperparameters'].get('clip_epsilon', 0.2)
            
            # Gaussian perturbation
            new_clip = current_clip + np.random.randn() * sigma
            new_clip = float(np.clip(new_clip, bounds[0], bounds[1]))
            
            if 'hyperparameters' in mutated:
                mutated['hyperparameters']['clip_epsilon'] = new_clip
            else:
                mutated['clip_epsilon'] = new_clip
            
            mutated['hyperparameter_mutation'] = {
                'type': 'clip_epsilon',
                'old_value': current_clip,
                'new_value': new_clip
            }
            
            return mutated, EvolutionResult(
                success=True,
                message=f"Mutated clip_epsilon: {current_clip:.4f} -> {new_clip:.4f}",
                details={'old': current_clip, 'new': new_clip}
            )
            
        except Exception as e:
            return config, EvolutionResult(
                success=False,
                message=f"Clip epsilon mutation failed: {str(e)}",
                details=None
            )
    
    def mutate_tau(
        self,
        config: Dict,
        sigma: float = 0.001,
        bounds: Tuple[float, float] = (0.001, 0.01)
    ) -> Tuple[Dict, EvolutionResult]:
        """
        Mutate DDPG soft update tau.
        
        Args:
            config: Agent config with hyperparameters
            sigma: Standard deviation for Gaussian noise
            bounds: (min, max) bounds
            
        Returns:
            (mutated_config, result)
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        try:
            mutated = copy.deepcopy(config)
            
            current_tau = 0.005  # Default
            if 'tau' in config:
                current_tau = config['tau']
            elif 'hyperparameters' in config:
                current_tau = config['hyperparameters'].get('tau', 0.005)
            
            # Gaussian perturbation
            new_tau = current_tau + np.random.randn() * sigma
            new_tau = float(np.clip(new_tau, bounds[0], bounds[1]))
            
            if 'hyperparameters' in mutated:
                mutated['hyperparameters']['tau'] = new_tau
            else:
                mutated['tau'] = new_tau
            
            mutated['hyperparameter_mutation'] = {
                'type': 'tau',
                'old_value': current_tau,
                'new_value': new_tau
            }
            
            return mutated, EvolutionResult(
                success=True,
                message=f"Mutated tau: {current_tau:.5f} -> {new_tau:.5f}",
                details={'old': current_tau, 'new': new_tau}
            )
            
        except Exception as e:
            return config, EvolutionResult(
                success=False,
                message=f"Tau mutation failed: {str(e)}",
                details=None
            )
    
    def mutate_all(
        self,
        config: Dict,
        sigma_lr: float = 0.1,
        sigma_gamma: float = 0.01,
        sigma_clip: float = 0.02,
        sigma_tau: float = 0.001,
        algorithm: str = 'ddpg'
    ) -> Tuple[Dict, EvolutionResult]:
        """
        Mutate all relevant hyperparameters.
        
        Args:
            config: Agent config
            sigma_*: Standard deviations for each parameter
            algorithm: 'ddpg' or 'ppo'
            
        Returns:
            (mutated_config, result)
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        mutated = copy.deepcopy(config)
        mutations = []
        
        # Learning rate
        new_config, result = self.mutate_learning_rate(mutated, sigma_lr)
        if result.success:
            mutated = new_config
            mutations.append(f"LR:{result.details['new']:.2e}")
        
        # Gamma (common to both)
        new_config, result = self.mutate_discount_factor(mutated, sigma_gamma)
        if result.success:
            mutated = new_config
            mutations.append(f"gamma:{result.details['new']:.4f}")
        
        if algorithm == 'ppo':
            # Clip epsilon for PPO
            new_config, result = self.mutate_clip_epsilon(mutated, sigma_clip)
            if result.success:
                mutated = new_config
                mutations.append(f"clip:{result.details['new']:.4f}")
        else:
            # Tau for DDPG
            new_config, result = self.mutate_tau(mutated, sigma_tau)
            if result.success:
                mutated = new_config
                mutations.append(f"tau:{result.details['new']:.5f}")
        
        return mutated, EvolutionResult(
            success=True,
            message=f"Mutated HPs: {', '.join(mutations)}",
            details={'mutations': mutations}
        )


class EvolutionaryAlgorithm:
    """
    Main evolutionary algorithm orchestrator.
    
    Combines selection, mutation, crossover, and hyperparameter mutation
    for population-based training.
    """
    
    def __init__(
        self,
        config: Dict,
        seed: Optional[int] = None
    ):
        """
        Initialize evolutionary algorithm.
        
        Args:
            config: Multi-agent configuration
            seed: Random seed
        """
        self.config = config
        self.seed = seed
        
        # Extract config parameters
        ma_config = config.get('multi_agent', {})
        self.population_size = ma_config.get('population_size', 8)
        self.selection_ratio = ma_config.get('selection_ratio', 0.5)
        self.mutation_sigma = ma_config.get('mutation_sigma', 0.05)
        self.crossover_enabled = ma_config.get('crossover_enabled', False)
        self.hp_mutation = ma_config.get('hyperparameter_mutation', True)
        self.elitism_count = ma_config.get('elitism_count', 2)
        
        # Initialize operators
        self.selector = SelectionOperator(seed=seed)
        self.mutator = MutationOperator(sigma=self.mutation_sigma, seed=seed)
        self.crossover = CrossoverOperator(seed=seed)
        self.hp_mutator = HyperparameterMutator(seed=seed)
    
    def evolve_population(
        self,
        population: List[Dict],
        fitness_scores: List[float],
        algorithm: str = 'ddpg'
    ) -> Tuple[List[Dict], EvolutionResult]:
        """
        Evolve a population through selection, mutation, and crossover.
        
        Args:
            population: List of agent dictionaries
            fitness_scores: Corresponding fitness values
            algorithm: 'ddpg' or 'ppo'
            
        Returns:
            (new_population, result)
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        try:
            new_population = []
            population_size = len(population)
            
            # 1. Elitism: keep top performers unchanged
            elites = self.selector.elitism_select(
                population, fitness_scores, num_elites=self.elitism_count
            )
            new_population.extend(elites)
            
            # 2. Selection: select parents for reproduction
            num_parents = population_size - self.elitism_count
            parents = self.selector.tournament_select(
                population, fitness_scores,
                tournament_size=3,
                num_select=num_parents
            )
            
            # 3. Reproduction: clone and mutate parents
            for i, parent in enumerate(parents):
                # Decide mutation type
                if random.random() < 0.3:
                    # Clone with noise (exploration)
                    offspring, result = self.mutator.clone_with_noise(
                        parent,
                        sigma=self.mutation_sigma,
                        param_subset=0.5
                    )
                else:
                    # Pure clone (exploitation)
                    offspring, result = self.mutator.clone(parent)
                
                # Optionally mutate hyperparameters
                if self.hp_mutation and random.random() < 0.2:
                    hp_config = offspring.get('hyperparameters', offspring)
                    new_config, hp_result = self.hp_mutator.mutate_all(
                        hp_config,
                        algorithm=algorithm
                    )
                    if hp_result.success:
                        offspring['hyperparameters'] = new_config.get('hyperparameters', {})
                
                # Update agent ID
                offspring['agent_id'] = f"gen_{i}"
                offspring['parent_id'] = parent.get('agent_id', 'unknown')
                
                new_population.append(offspring)
            
            # 4. Optional crossover (not implemented in v1)
            # Could add pairwise crossover here if enabled
            
            return new_population, EvolutionResult(
                success=True,
                message=f"Evolved population: {len(new_population)} agents",
                details={
                    'elites': len(elites),
                    'offspring': len(new_population) - len(elites)
                }
            )
            
        except Exception as e:
            return population, EvolutionResult(
                success=False,
                message=f"Evolution failed: {str(e)}",
                details=None
            )
    
    def select_top_performers(
        self,
        population: List[Dict],
        fitness_scores: List[float],
        num_select: Optional[int] = None
    ) -> Tuple[List[Dict], List[float]]:
        """
        Select top-performing agents.
        
        Args:
            population: List of agent dictionaries
            fitness_scores: Corresponding fitness values
            num_select: Number to select (default: selection_ratio * population)
            
        Returns:
            (selected_agents, selected_scores)
        """
        if num_select is None:
            num_select = max(1, int(len(population) * self.selection_ratio))
        
        # Use tournament selection for diversity
        selected = self.selector.tournament_select(
            population, fitness_scores,
            tournament_size=3,
            num_select=num_select
        )
        
        # Get corresponding fitness scores
        selected_scores = []
        for agent in selected:
            # Find matching score
            for i, p in enumerate(population):
                if p.get('agent_id') == agent.get('agent_id'):
                    selected_scores.append(fitness_scores[i])
                    break
            else:
                # Fallback to mean if not found
                selected_scores.append(np.mean(fitness_scores))
        
        return selected, selected_scores

