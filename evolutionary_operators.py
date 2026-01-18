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

# Import logging
from logging_utils import (
    log, log_info, log_warning, log_error, log_success, 
    log_debug, log_data, log_exception, log_section,
    start_timer, stop_timer, LogFunction
)


@dataclass
class EvolutionResult:
    """Result of an evolutionary operation."""
    success: bool
    message: str
    details: Optional[Dict] = None
    
    def __str__(self) -> str:
        """String representation of evolution result."""
        return f"EvolutionResult(success={self.success}, message={self.message})"


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
        log_section("SelectionOperator", "INITIALIZING SELECTION OPERATOR")
        
        with LogFunction("SelectionOperator", "__init__", args=(seed,)):
            self.seed = seed
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
            
            log_info("SelectionOperator", f"Initialized with seed: {seed}")
            log_success("SelectionOperator", "Selection operator initialized")
    
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
        with LogFunction("SelectionOperator", "tournament_select",
                        args=(tournament_size, num_select, maximize),
                        kwargs={'population_size': len(population),
                               'fitness_scores_length': len(fitness_scores)}):
            
            log_info("SelectionOperator", f"Tournament selection: size={tournament_size}, select={num_select}, maximize={maximize}")
            
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
            
            selected = []
            indices = list(range(len(population)))
            
            log_data("SelectionOperator", "Population indices", indices)
            log_data("SelectionOperator", "Fitness scores", fitness_scores)
            
            for select_idx in range(num_select):
                # Randomly sample tournament participants
                if tournament_size >= len(population):
                    participants = indices.copy()
                    log_debug("SelectionOperator", f"Tournament {select_idx}: using all {len(participants)} participants")
                else:
                    participants = random.sample(indices, tournament_size)
                    log_debug("SelectionOperator", f"Tournament {select_idx}: sampled {len(participants)} participants")
                
                # Get fitness scores for participants
                participant_fitness = [(i, fitness_scores[i]) for i in participants]
                log_data("SelectionOperator", f"Tournament {select_idx} participants", participant_fitness)
                
                # Select best (or worst if minimizing)
                if maximize:
                    best_idx = max(participant_fitness, key=lambda x: x[1])[0]
                    selection_type = "maximizing"
                else:
                    best_idx = min(participant_fitness, key=lambda x: x[1])[0]
                    selection_type = "minimizing"
                
                # Deep copy to avoid reference issues
                selected.append(copy.deepcopy(population[best_idx]))
                log_debug("SelectionOperator", f"Tournament {select_idx}: selected index {best_idx} (fitness={fitness_scores[best_idx]})")
            
            log_success("SelectionOperator", f"Selected {len(selected)} individuals via tournament selection")
            log_data("SelectionOperator", "Selected indices", [i for i, _ in enumerate(selected)])
            
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
        with LogFunction("SelectionOperator", "roulette_wheel_select",
                        args=(num_select, maximize),
                        kwargs={'population_size': len(population),
                               'fitness_scores_length': len(fitness_scores)}):
            
            log_info("SelectionOperator", f"Roulette wheel selection: select={num_select}, maximize={maximize}")
            
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
            
            if len(population) == 0:
                log_warning("SelectionOperator", "Empty population, returning empty list")
                return []
            
            # Convert to positive values for roulette wheel
            if maximize:
                # Higher is better: use fitness directly
                min_fitness = min(fitness_scores)
                if min_fitness < 0:
                    adjusted = [f - min_fitness + 1e-6 for f in fitness_scores]
                    log_debug("SelectionOperator", f"Adjusted negative fitness (min={min_fitness})")
                else:
                    adjusted = [f + 1e-6 for f in fitness_scores]
                adjustment_type = "maximizing"
            else:
                # Lower is better: invert
                max_fitness = max(fitness_scores)
                adjusted = [max_fitness - f + 1e-6 for f in fitness_scores]
                adjustment_type = "minimizing"
            
            log_data("SelectionOperator", f"Original fitness ({adjustment_type})", fitness_scores)
            log_data("SelectionOperator", "Adjusted fitness", adjusted)
            
            # Normalize to probabilities
            total = sum(adjusted)
            probabilities = [p / total for p in adjusted]
            
            log_data("SelectionOperator", "Selection probabilities", probabilities)
            log_data("SelectionOperator", "Probability sum", sum(probabilities))
            
            selected = []
            for select_idx in range(num_select):
                idx = np.random.choice(len(population), p=probabilities)
                selected.append(copy.deepcopy(population[idx]))
                log_debug("SelectionOperator", f"Selection {select_idx}: selected index {idx} (prob={probabilities[idx]:.4f})")
            
            log_success("SelectionOperator", f"Selected {len(selected)} individuals via roulette wheel")
            
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
        with LogFunction("SelectionOperator", "rank_select",
                        args=(num_select, maximize),
                        kwargs={'population_size': len(population),
                               'fitness_scores_length': len(fitness_scores)}):
            
            log_info("SelectionOperator", f"Rank-based selection: select={num_select}, maximize={maximize}")
            
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
            
            if len(population) == 0:
                log_warning("SelectionOperator", "Empty population, returning empty list")
                return []
            
            # Sort by fitness and assign ranks
            sorted_pairs = sorted(
                enumerate(fitness_scores),
                key=lambda x: x[1],
                reverse=maximize
            )
            
            log_data("SelectionOperator", "Sorted fitness pairs", [(i, f) for i, f in sorted_pairs[:5]])
            
            ranks = [0] * len(population)
            for rank, (original_idx, _) in enumerate(sorted_pairs):
                ranks[original_idx] = rank + 1  # 1-indexed ranks
            
            log_data("SelectionOperator", "Ranks assigned", ranks)
            
            # Use ranks as selection probabilities (higher rank = higher probability)
            total_rank = sum(ranks)
            probabilities = [r / total_rank for r in ranks]
            
            log_data("SelectionOperator", "Rank probabilities", probabilities)
            log_data("SelectionOperator", "Probability sum", sum(probabilities))
            
            selected = []
            for select_idx in range(num_select):
                idx = np.random.choice(len(population), p=probabilities)
                selected.append(copy.deepcopy(population[idx]))
                log_debug("SelectionOperator", f"Selection {select_idx}: selected index {idx} (rank={ranks[idx]}, prob={probabilities[idx]:.4f})")
            
            log_success("SelectionOperator", f"Selected {len(selected)} individuals via rank selection")
            
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
        with LogFunction("SelectionOperator", "elitism_select",
                        args=(num_elites, maximize),
                        kwargs={'population_size': len(population),
                               'fitness_scores_length': len(fitness_scores)}):
            
            log_info("SelectionOperator", f"Elitism selection: elites={num_elites}, maximize={maximize}")
            
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
            
            if len(population) == 0:
                log_warning("SelectionOperator", "Empty population, returning empty list")
                return []
            
            # Sort by fitness
            sorted_pairs = sorted(
                enumerate(fitness_scores),
                key=lambda x: x[1],
                reverse=maximize
            )
            
            log_data("SelectionOperator", "Top fitness scores", [(i, f) for i, f in sorted_pairs[:min(5, len(sorted_pairs))]])
            
            # Select top num_elites
            num_elites = min(num_elites, len(population))
            selected = []
            for i in range(num_elites):
                original_idx = sorted_pairs[i][0]
                selected.append(copy.deepcopy(population[original_idx]))
                log_debug("SelectionOperator", f"Elite {i}: selected index {original_idx} (fitness={fitness_scores[original_idx]})")
            
            log_success("SelectionOperator", f"Selected {len(selected)} elites")
            log_data("SelectionOperator", "Selected elite indices", [sorted_pairs[i][0] for i in range(num_elites)])
            
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
        log_section("MutationOperator", "INITIALIZING MUTATION OPERATOR")
        
        with LogFunction("MutationOperator", "__init__", args=(sigma, seed)):
            self.sigma = sigma
            self.seed = seed
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
            
            log_info("MutationOperator", f"Initialized with sigma={sigma}, seed={seed}")
            log_success("MutationOperator", "Mutation operator initialized")
    
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
        with LogFunction("MutationOperator", "gaussian_mutate",
                        args=(sigma, clip_value),
                        kwargs={'agent_keys': list(agent.keys())}):
            
            log_info("MutationOperator", f"Gaussian mutate called with sigma={sigma}, clip={clip_value}")
            log_data("MutationOperator", "Agent keys", list(agent.keys()))
            
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
            
            sigma = sigma if sigma is not None else self.sigma
            log_info("MutationOperator", f"Using sigma: {sigma}")
            
            try:
                log_debug("MutationOperator", "Creating deep copy of agent")
                mutated_agent = copy.deepcopy(agent)
                
                if 'model_state' not in mutated_agent:
                    log_error("MutationOperator", "No model_state in agent")
                    return agent, EvolutionResult(
                        success=False,
                        message="No model_state in agent",
                        details=None
                    )
                
                model_state = mutated_agent['model_state']
                param_count = len(model_state)
                log_info("MutationOperator", f"Processing {param_count} model parameters")
                
                mutated_params = 0
                total_params = 0
                
                for key, tensor in model_state.items():
                    if isinstance(tensor, torch.Tensor):
                        # Skip if not a floating point tensor
                        if tensor.dtype.is_floating_point:
                            log_debug("MutationOperator", f"Mutating parameter: {key} {tensor.shape}")
                            total_params += tensor.numel()
                            noise = torch.randn_like(tensor) * sigma
                            model_state[key] = tensor + noise
                            
                            # Clip to prevent extreme values
                            if clip_value > 0:
                                model_state[key] = torch.clamp(
                                    model_state[key], 
                                    -clip_value, 
                                    clip_value
                                )
                            
                            mutated_params += 1
                        else:
                            log_debug("MutationOperator", f"Skipping non-floating tensor: {key} {tensor.dtype}")
                    else:
                        log_debug("MutationOperator", f"Skipping non-tensor: {key} {type(tensor)}")
                
                mutated_agent['mutation_info'] = {
                    'type': 'gaussian',
                    'sigma': sigma,
                    'parent_id': agent.get('agent_id', 'unknown'),
                    'mutated_params': mutated_params,
                    'total_params': param_count
                }
                
                log_success("MutationOperator", 
                           f"Applied Gaussian mutation: {mutated_params}/{param_count} params")
                log_data("MutationOperator", "Mutation sigma", sigma)
                log_data("MutationOperator", "Total parameter count", total_params)
                
                return mutated_agent, EvolutionResult(
                    success=True,
                    message=f"Applied Gaussian mutation with sigma={sigma}",
                    details={'sigma': sigma, 'mutated_params': mutated_params, 'total_params': param_count}
                )
                
            except Exception as e:
                log_exception("MutationOperator", e, "During Gaussian mutation")
                return agent, EvolutionResult(
                    success=False,
                    message=f"Mutation failed: {str(e)}",
                    details={'error': str(e)}
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
        with LogFunction("MutationOperator", "clone",
                        args=(),
                        kwargs={'agent_keys': list(agent.keys())}):
            
            log_info("MutationOperator", "Cloning agent")
            
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
                
                log_success("MutationOperator", f"Cloned agent from {base_id}")
                log_data("MutationOperator", "New agent ID", cloned['agent_id'])
                
                return cloned, EvolutionResult(
                    success=True,
                    message=f"Cloned agent from {base_id}",
                    details={'parent_id': base_id, 'new_agent_id': cloned['agent_id']}
                )
                
            except Exception as e:
                log_exception("MutationOperator", e, "During cloning")
                return agent, EvolutionResult(
                    success=False,
                    message=f"Clone failed: {str(e)}",
                    details={'error': str(e)}
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
        with LogFunction("MutationOperator", "clone_with_noise",
                        args=(sigma, param_subset),
                        kwargs={'agent_keys': list(agent.keys())}):
            
            log_info("MutationOperator", f"Clone with noise: sigma={sigma}, subset={param_subset}")
            
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
            
            sigma = sigma if sigma is not None else self.sigma
            log_info("MutationOperator", f"Using sigma: {sigma}")
            
            try:
                # First clone
                log_debug("MutationOperator", "Creating base clone")
                cloned, clone_result = self.clone(agent)
                
                if not clone_result.success:
                    log_error("MutationOperator", f"Clone failed: {clone_result.message}")
                    return agent, clone_result
                
                if 'model_state' not in cloned:
                    log_error("MutationOperator", "No model_state in cloned agent")
                    return agent, EvolutionResult(
                        success=False,
                        message="No model_state in agent",
                        details=None
                    )
                
                model_state = cloned['model_state']
                param_names = list(model_state.keys())
                log_info("MutationOperator", f"Available parameters: {len(param_names)}")
                log_data("MutationOperator", "Parameter names", param_names)
                
                # Select subset of parameters to mutate
                num_to_mutate = max(1, int(len(param_names) * param_subset))
                params_to_mutate = random.sample(param_names, num_to_mutate)
                
                log_info("MutationOperator", f"Mutating {num_to_mutate} out of {len(param_names)} parameters")
                log_data("MutationOperator", "Parameters to mutate", params_to_mutate)
                
                mutated_count = 0
                for key in params_to_mutate:
                    if isinstance(model_state[key], torch.Tensor):
                        if model_state[key].dtype.is_floating_point:
                            log_debug("MutationOperator", f"Mutating parameter: {key} {model_state[key].shape}")
                            noise = torch.randn_like(model_state[key]) * sigma
                            model_state[key] = model_state[key] + noise
                            mutated_count += 1
                        else:
                            log_debug("MutationOperator", f"Skipping non-floating tensor: {key}")
                    else:
                        log_debug("MutationOperator", f"Skipping non-tensor: {key}")
                
                cloned['mutation_info'] = {
                    'type': 'clone_with_noise',
                    'sigma': sigma,
                    'params_mutated': mutated_count,
                    'params_total': len(param_names),
                    'param_subset': param_subset,
                    'parent_id': agent.get('agent_id', 'unknown')
                }
                
                log_success("MutationOperator", 
                           f"Cloned with noise: {mutated_count} params mutated")
                log_data("MutationOperator", "Mutation details", {
                    'sigma': sigma,
                    'mutated_count': mutated_count,
                    'total_params': len(param_names)
                })
                
                return cloned, EvolutionResult(
                    success=True,
                    message=f"Cloned with noise: {mutated_count} params mutated",
                    details={'sigma': sigma, 'params_mutated': mutated_count, 'total_params': len(param_names)}
                )
                
            except Exception as e:
                log_exception("MutationOperator", e, "During clone with noise")
                return agent, EvolutionResult(
                    success=False,
                    message=f"Clone with noise failed: {str(e)}",
                    details={'error': str(e)}
                )


class CrossoverOperator:
    """
    Crossover (recombination) operators.
    
    Note: Crossover is optional and disabled by default.
    """
    
    def __init__(self, seed: Optional[int] = None):
        log_section("CrossoverOperator", "INITIALIZING CROSSOVER OPERATOR")
        
        with LogFunction("CrossoverOperator", "__init__", args=(seed,)):
            self.seed = seed
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
            
            log_info("CrossoverOperator", f"Initialized with seed: {seed}")
            log_success("CrossoverOperator", "Crossover operator initialized")
    
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
        with LogFunction("CrossoverOperator", "uniform_crossover",
                        args=(crossover_rate,),
                        kwargs={'agent1_keys': list(agent1.keys()),
                               'agent2_keys': list(agent2.keys())}):
            
            log_info("CrossoverOperator", f"Uniform crossover: rate={crossover_rate}")
            
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
            
            try:
                offspring = copy.deepcopy(agent1)
                
                if 'model_state' not in agent1 or 'model_state' not in agent2:
                    log_error("CrossoverOperator", "Both agents must have model_state")
                    return agent1, EvolutionResult(
                        success=False,
                        message="Both agents must have model_state",
                        details=None
                    )
                
                state1 = agent1['model_state']
                state2 = agent2['model_state']
                
                log_info("CrossoverOperator", f"Agent1 parameters: {len(state1)}, Agent2 parameters: {len(state2)}")
                
                crossover_count = 0
                total_params = 0
                
                for key in state1.keys():
                    if key in state2:
                        if (isinstance(state1[key], torch.Tensor) and 
                            isinstance(state2[key], torch.Tensor) and
                            state1[key].shape == state2[key].shape):
                            
                            total_params += 1
                            
                            # Create crossover mask
                            mask = torch.rand_like(state1[key]) < crossover_rate
                            
                            # For tensors, blend values
                            if state1[key].dtype.is_floating_point:
                                # Weighted average
                                alpha = torch.rand_like(state1[key])
                                offspring['model_state'][key] = alpha * state1[key] + (1 - alpha) * state2[key]
                                crossover_count += 1
                                log_debug("CrossoverOperator", f"Crossed parameter: {key} {state1[key].shape}")
                            else:
                                log_debug("CrossoverOperator", f"Skipping non-floating tensor: {key}")
                        else:
                            log_debug("CrossoverOperator", f"Parameter mismatch: {key} (shapes: {state1[key].shape if isinstance(state1[key], torch.Tensor) else type(state1[key])} vs {state2[key].shape if isinstance(state2[key], torch.Tensor) else type(state2[key])})")
                    else:
                        log_debug("CrossoverOperator", f"Key {key} not in agent2")
                
                offspring['crossover_info'] = {
                    'type': 'uniform',
                    'parent1': agent1.get('agent_id', 'unknown'),
                    'parent2': agent2.get('agent_id', 'unknown'),
                    'crossover_rate': crossover_rate,
                    'crossed_params': crossover_count,
                    'total_params': total_params
                }
                
                log_success("CrossoverOperator", 
                           f"Uniform crossover: {crossover_count}/{total_params} parameters crossed")
                log_data("CrossoverOperator", "Crossover details", {
                    'parent1': agent1.get('agent_id'),
                    'parent2': agent2.get('agent_id'),
                    'crossover_rate': crossover_rate,
                    'crossed_params': crossover_count
                })
                
                return offspring, EvolutionResult(
                    success=True,
                    message=f"Crossover between agents",
                    details={'parent1': agent1.get('agent_id'), 'parent2': agent2.get('agent_id'),
                            'crossed_params': crossover_count, 'total_params': total_params}
                )
                
            except Exception as e:
                log_exception("CrossoverOperator", e, "During uniform crossover")
                return agent1, EvolutionResult(
                    success=False,
                    message=f"Crossover failed: {str(e)}",
                    details={'error': str(e)}
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
        with LogFunction("CrossoverOperator", "arithmetic_crossover",
                        args=(alpha,),
                        kwargs={'agent1_keys': list(agent1.keys()),
                               'agent2_keys': list(agent2.keys())}):
            
            log_info("CrossoverOperator", f"Arithmetic crossover: alpha={alpha}")
            
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
            
            try:
                offspring = copy.deepcopy(agent1)
                
                if 'model_state' not in agent1 or 'model_state' not in agent2:
                    log_error("CrossoverOperator", "Both agents must have model_state")
                    return agent1, EvolutionResult(
                        success=False,
                        message="Both agents must have model_state",
                        details=None
                    )
                
                state1 = agent1['model_state']
                state2 = agent2['model_state']
                
                log_info("CrossoverOperator", f"Agent1 parameters: {len(state1)}, Agent2 parameters: {len(state2)}")
                
                crossover_count = 0
                total_params = 0
                
                for key in state1.keys():
                    if key in state2:
                        if (isinstance(state1[key], torch.Tensor) and 
                            isinstance(state2[key], torch.Tensor) and
                            state1[key].shape == state2[key].shape and
                            state1[key].dtype.is_floating_point):
                            
                            total_params += 1
                            
                            # Arithmetic mean
                            offspring['model_state'][key] = (
                                alpha * state1[key] + (1 - alpha) * state2[key]
                            )
                            crossover_count += 1
                            log_debug("CrossoverOperator", f"Arithmetic crossover: {key} {state1[key].shape}")
                        else:
                            log_debug("CrossoverOperator", f"Skipping parameter: {key} (not compatible)")
                    else:
                        log_debug("CrossoverOperator", f"Key {key} not in agent2")
                
                offspring['crossover_info'] = {
                    'type': 'arithmetic',
                    'alpha': alpha,
                    'parent1': agent1.get('agent_id', 'unknown'),
                    'parent2': agent2.get('agent_id', 'unknown'),
                    'crossed_params': crossover_count,
                    'total_params': total_params
                }
                
                log_success("CrossoverOperator", 
                           f"Arithmetic crossover: {crossover_count}/{total_params} parameters crossed")
                log_data("CrossoverOperator", "Crossover details", {
                    'alpha': alpha,
                    'parent1': agent1.get('agent_id'),
                    'parent2': agent2.get('agent_id'),
                    'crossed_params': crossover_count
                })
                
                return offspring, EvolutionResult(
                    success=True,
                    message=f"Arithmetic crossover with alpha={alpha}",
                    details={'alpha': alpha, 'crossed_params': crossover_count, 'total_params': total_params}
                )
                
            except Exception as e:
                log_exception("CrossoverOperator", e, "During arithmetic crossover")
                return agent1, EvolutionResult(
                    success=False,
                    message=f"Crossover failed: {str(e)}",
                    details={'error': str(e)}
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
        log_section("HyperparameterMutator", "INITIALIZING HYPERPARAMETER MUTATOR")
        
        with LogFunction("HyperparameterMutator", "__init__", args=(seed,)):
            self.seed = seed
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
            
            log_info("HyperparameterMutator", f"Initialized with seed: {seed}")
            log_success("HyperparameterMutator", "Hyperparameter mutator initialized")
    
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
        with LogFunction("HyperparameterMutator", "mutate_learning_rate",
                        args=(sigma_factor, bounds)):
            
            log_info("HyperparameterMutator", f"Mutating learning rate: sigma={sigma_factor}, bounds={bounds}")
            
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
                    log_warning("HyperparameterMutator", "No learning rate found in config")
                    return config, EvolutionResult(
                        success=False,
                        message="No learning rate found in config",
                        details=None
                    )
                
                log_debug("HyperparameterMutator", f"Learning rate key: {lr_key}")
                
                current_lr = mutated['hyperparameters'].get(lr_key, mutated.get(lr_key))
                if current_lr is None:
                    current_lr = mutated.get(lr_key, 0.001)
                
                log_info("HyperparameterMutator", f"Current learning rate: {current_lr}")
                
                # Log-normal perturbation
                log_lr = np.log(current_lr)
                new_log_lr = log_lr + np.random.randn() * sigma_factor
                new_lr = np.exp(new_log_lr)
                
                # Clamp to bounds
                new_lr = float(np.clip(new_lr, bounds[0], bounds[1]))
                
                log_info("HyperparameterMutator", f"New learning rate: {new_lr}")
                
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
                    'new_value': new_lr,
                    'sigma_factor': sigma_factor,
                    'bounds': bounds
                }
                
                log_success("HyperparameterMutator", 
                           f"Mutated LR: {current_lr:.2e} -> {new_lr:.2e}")
                log_data("HyperparameterMutator", "LR mutation details", {
                    'old': current_lr,
                    'new': new_lr,
                    'change_factor': new_lr / current_lr
                })
                
                return mutated, EvolutionResult(
                    success=True,
                    message=f"Mutated LR: {current_lr:.2e} -> {new_lr:.2e}",
                    details={'old': current_lr, 'new': new_lr, 'change_factor': new_lr / current_lr}
                )
                
            except Exception as e:
                log_exception("HyperparameterMutator", e, "During LR mutation")
                return config, EvolutionResult(
                    success=False,
                    message=f"LR mutation failed: {str(e)}",
                    details={'error': str(e)}
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
        with LogFunction("HyperparameterMutator", "mutate_discount_factor",
                        args=(sigma, bounds)):
            
            log_info("HyperparameterMutator", f"Mutating discount factor: sigma={sigma}, bounds={bounds}")
            
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
            
            try:
                mutated = copy.deepcopy(config)
                
                current_gamma = mutated.get('gamma', 0.99)
                if 'hyperparameters' in mutated:
                    current_gamma = mutated['hyperparameters'].get('gamma', current_gamma)
                
                log_info("HyperparameterMutator", f"Current gamma: {current_gamma}")
                
                # Gaussian perturbation
                new_gamma = current_gamma + np.random.randn() * sigma
                new_gamma = float(np.clip(new_gamma, bounds[0], bounds[1]))
                
                log_info("HyperparameterMutator", f"New gamma: {new_gamma}")
                
                if 'hyperparameters' in mutated:
                    mutated['hyperparameters']['gamma'] = new_gamma
                else:
                    mutated['gamma'] = new_gamma
                
                mutated['hyperparameter_mutation'] = {
                    'type': 'gamma',
                    'old_value': current_gamma,
                    'new_value': new_gamma,
                    'sigma': sigma,
                    'bounds': bounds
                }
                
                log_success("HyperparameterMutator", 
                           f"Mutated gamma: {current_gamma:.4f} -> {new_gamma:.4f}")
                log_data("HyperparameterMutator", "Gamma mutation details", {
                    'old': current_gamma,
                    'new': new_gamma,
                    'delta': new_gamma - current_gamma
                })
                
                return mutated, EvolutionResult(
                    success=True,
                    message=f"Mutated gamma: {current_gamma:.4f} -> {new_gamma:.4f}",
                    details={'old': current_gamma, 'new': new_gamma, 'delta': new_gamma - current_gamma}
                )
                
            except Exception as e:
                log_exception("HyperparameterMutator", e, "During gamma mutation")
                return config, EvolutionResult(
                    success=False,
                    message=f"Gamma mutation failed: {str(e)}",
                    details={'error': str(e)}
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
        with LogFunction("HyperparameterMutator", "mutate_clip_epsilon",
                        args=(sigma, bounds)):
            
            log_info("HyperparameterMutator", f"Mutating clip epsilon: sigma={sigma}, bounds={bounds}")
            
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
                
                log_info("HyperparameterMutator", f"Current clip epsilon: {current_clip}")
                
                # Gaussian perturbation
                new_clip = current_clip + np.random.randn() * sigma
                new_clip = float(np.clip(new_clip, bounds[0], bounds[1]))
                
                log_info("HyperparameterMutator", f"New clip epsilon: {new_clip}")
                
                if 'hyperparameters' in mutated:
                    mutated['hyperparameters']['clip_epsilon'] = new_clip
                else:
                    mutated['clip_epsilon'] = new_clip
                
                mutated['hyperparameter_mutation'] = {
                    'type': 'clip_epsilon',
                    'old_value': current_clip,
                    'new_value': new_clip,
                    'sigma': sigma,
                    'bounds': bounds
                }
                
                log_success("HyperparameterMutator", 
                           f"Mutated clip_epsilon: {current_clip:.4f} -> {new_clip:.4f}")
                log_data("HyperparameterMutator", "Clip epsilon mutation details", {
                    'old': current_clip,
                    'new': new_clip,
                    'delta': new_clip - current_clip
                })
                
                return mutated, EvolutionResult(
                    success=True,
                    message=f"Mutated clip_epsilon: {current_clip:.4f} -> {new_clip:.4f}",
                    details={'old': current_clip, 'new': new_clip, 'delta': new_clip - current_clip}
                )
                
            except Exception as e:
                log_exception("HyperparameterMutator", e, "During clip epsilon mutation")
                return config, EvolutionResult(
                    success=False,
                    message=f"Clip epsilon mutation failed: {str(e)}",
                    details={'error': str(e)}
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
        with LogFunction("HyperparameterMutator", "mutate_tau",
                        args=(sigma, bounds)):
            
            log_info("HyperparameterMutator", f"Mutating tau: sigma={sigma}, bounds={bounds}")
            
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
                
                log_info("HyperparameterMutator", f"Current tau: {current_tau}")
                
                # Gaussian perturbation
                new_tau = current_tau + np.random.randn() * sigma
                new_tau = float(np.clip(new_tau, bounds[0], bounds[1]))
                
                log_info("HyperparameterMutator", f"New tau: {new_tau}")
                
                if 'hyperparameters' in mutated:
                    mutated['hyperparameters']['tau'] = new_tau
                else:
                    mutated['tau'] = new_tau
                
                mutated['hyperparameter_mutation'] = {
                    'type': 'tau',
                    'old_value': current_tau,
                    'new_value': new_tau,
                    'sigma': sigma,
                    'bounds': bounds
                }
                
                log_success("HyperparameterMutator", 
                           f"Mutated tau: {current_tau:.5f} -> {new_tau:.5f}")
                log_data("HyperparameterMutator", "Tau mutation details", {
                    'old': current_tau,
                    'new': new_tau,
                    'delta': new_tau - current_tau
                })
                
                return mutated, EvolutionResult(
                    success=True,
                    message=f"Mutated tau: {current_tau:.5f} -> {new_tau:.5f}",
                    details={'old': current_tau, 'new': new_tau, 'delta': new_tau - current_tau}
                )
                
            except Exception as e:
                log_exception("HyperparameterMutator", e, "During tau mutation")
                return config, EvolutionResult(
                    success=False,
                    message=f"Tau mutation failed: {str(e)}",
                    details={'error': str(e)}
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
        with LogFunction("HyperparameterMutator", "mutate_all",
                        args=(sigma_lr, sigma_gamma, sigma_clip, sigma_tau, algorithm)):
            
            log_info("HyperparameterMutator", f"Mutating all hyperparameters for algorithm: {algorithm}")
            log_data("HyperparameterMutator", "Mutation sigmas", {
                'sigma_lr': sigma_lr,
                'sigma_gamma': sigma_gamma,
                'sigma_clip': sigma_clip,
                'sigma_tau': sigma_tau
            })
            
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
            
            mutated = copy.deepcopy(config)
            mutations = []
            mutation_details = []
            
            # Learning rate
            log_debug("HyperparameterMutator", "Mutating learning rate")
            new_config, result = self.mutate_learning_rate(mutated, sigma_lr)
            if result.success:
                mutated = new_config
                mutations.append(f"LR:{result.details['new']:.2e}")
                mutation_details.append({
                    'type': 'learning_rate',
                    'old': result.details['old'],
                    'new': result.details['new']
                })
            
            # Gamma (common to both)
            log_debug("HyperparameterMutator", "Mutating discount factor")
            new_config, result = self.mutate_discount_factor(mutated, sigma_gamma)
            if result.success:
                mutated = new_config
                mutations.append(f"gamma:{result.details['new']:.4f}")
                mutation_details.append({
                    'type': 'gamma',
                    'old': result.details['old'],
                    'new': result.details['new']
                })
            
            if algorithm == 'ppo':
                # Clip epsilon for PPO
                log_debug("HyperparameterMutator", "Mutating clip epsilon (PPO)")
                new_config, result = self.mutate_clip_epsilon(mutated, sigma_clip)
                if result.success:
                    mutated = new_config
                    mutations.append(f"clip:{result.details['new']:.4f}")
                    mutation_details.append({
                        'type': 'clip_epsilon',
                        'old': result.details['old'],
                        'new': result.details['new']
                    })
            else:
                # Tau for DDPG
                log_debug("HyperparameterMutator", "Mutating tau (DDPG)")
                new_config, result = self.mutate_tau(mutated, sigma_tau)
                if result.success:
                    mutated = new_config
                    mutations.append(f"tau:{result.details['new']:.5f}")
                    mutation_details.append({
                        'type': 'tau',
                        'old': result.details['old'],
                        'new': result.details['new']
                    })
            
            log_success("HyperparameterMutator", f"Mutated {len(mutations)} hyperparameters")
            log_data("HyperparameterMutator", "Mutation summary", mutation_details)
            
            return mutated, EvolutionResult(
                success=True,
                message=f"Mutated HPs: {', '.join(mutations)}",
                details={'mutations': mutations, 'details': mutation_details}
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
        log_section("EvolutionaryAlgorithm", "INITIALIZING EVOLUTIONARY ALGORITHM")
        
        with LogFunction("EvolutionaryAlgorithm", "__init__", 
                        args=(seed,),
                        kwargs={'config_keys': list(config.keys())}):
            
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
            
            log_data("EvolutionaryAlgorithm", "Configuration", {
                "population_size": self.population_size,
                "selection_ratio": self.selection_ratio,
                "mutation_sigma": self.mutation_sigma,
                "crossover_enabled": self.crossover_enabled,
                "hp_mutation": self.hp_mutation,
                "elitism_count": self.elitism_count
            })
            
            # Initialize operators
            self.selector = SelectionOperator(seed=seed)
            self.mutator = MutationOperator(sigma=self.mutation_sigma, seed=seed)
            self.crossover = CrossoverOperator(seed=seed)
            self.hp_mutator = HyperparameterMutator(seed=seed)
            
            log_success("EvolutionaryAlgorithm", "Evolutionary algorithm initialized")
    
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
        with LogFunction("EvolutionaryAlgorithm", "evolve_population",
                        args=(algorithm,),
                        kwargs={'population_size': len(population),
                               'fitness_scores_length': len(fitness_scores)}):
            
            log_info("EvolutionaryAlgorithm", f"Evolving population for algorithm: {algorithm}")
            log_data("EvolutionaryAlgorithm", "Population size", len(population))
            log_data("EvolutionaryAlgorithm", "Fitness scores range", 
                    {'min': min(fitness_scores) if fitness_scores else 0,
                     'max': max(fitness_scores) if fitness_scores else 0,
                     'mean': np.mean(fitness_scores) if fitness_scores else 0})
            
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
            
            try:
                new_population = []
                population_size = len(population)
                
                # 1. Elitism: keep top performers unchanged
                log_info("EvolutionaryAlgorithm", f"Selecting {self.elitism_count} elites")
                elites = self.selector.elitism_select(
                    population, fitness_scores, num_elites=self.elitism_count
                )
                new_population.extend(elites)
                log_info("EvolutionaryAlgorithm", f"Selected {len(elites)} elites")
                
                # 2. Selection: select parents for reproduction
                num_parents = population_size - self.elitism_count
                log_info("EvolutionaryAlgorithm", f"Selecting {num_parents} parents via tournament")
                parents = self.selector.tournament_select(
                    population, fitness_scores,
                    tournament_size=3,
                    num_select=num_parents
                )
                log_info("EvolutionaryAlgorithm", f"Selected {len(parents)} parents")
                
                # 3. Reproduction: clone and mutate parents
                log_info("EvolutionaryAlgorithm", f"Reproducing {len(parents)} offspring")
                for i, parent in enumerate(parents):
                    # Decide mutation type
                    mutation_type = random.random()
                    if mutation_type < 0.3:
                        # Clone with noise (exploration)
                        log_debug("EvolutionaryAlgorithm", f"Parent {i}: clone with noise")
                        offspring, result = self.mutator.clone_with_noise(
                            parent,
                            sigma=self.mutation_sigma,
                            param_subset=0.5
                        )
                        mutation_desc = "clone_with_noise"
                    else:
                        # Pure clone (exploitation)
                        log_debug("EvolutionaryAlgorithm", f"Parent {i}: pure clone")
                        offspring, result = self.mutator.clone(parent)
                        mutation_desc = "pure_clone"
                    
                    if not result.success:
                        log_warning("EvolutionaryAlgorithm", f"Mutation failed for parent {i}: {result.message}")
                        # Fallback to parent
                        offspring = copy.deepcopy(parent)
                    
                    # Optionally mutate hyperparameters
                    if self.hp_mutation and random.random() < 0.2:
                        log_debug("EvolutionaryAlgorithm", f"Parent {i}: mutating hyperparameters")
                        hp_config = offspring.get('hyperparameters', offspring)
                        new_config, hp_result = self.hp_mutator.mutate_all(
                            hp_config,
                            algorithm=algorithm
                        )
                        if hp_result.success:
                            offspring['hyperparameters'] = new_config.get('hyperparameters', {})
                            log_debug("EvolutionaryAlgorithm", f"Parent {i}: hyperparameters mutated")
                    
                    # Update agent ID
                    offspring['agent_id'] = f"gen_{i}"
                    offspring['parent_id'] = parent.get('agent_id', 'unknown')
                    offspring['mutation_type'] = mutation_desc
                    
                    new_population.append(offspring)
                    log_debug("EvolutionaryAlgorithm", f"Created offspring {i}: {offspring['agent_id']}")
                
                # 4. Optional crossover (not implemented in v1)
                # Could add pairwise crossover here if enabled
                if self.crossover_enabled:
                    log_info("EvolutionaryAlgorithm", "Crossover is enabled but not implemented in this version")
                
                log_success("EvolutionaryAlgorithm", f"Evolved population: {len(new_population)} agents")
                log_data("EvolutionaryAlgorithm", "Population composition", {
                    'elites': len(elites),
                    'offspring': len(new_population) - len(elites),
                    'total': len(new_population)
                })
                
                return new_population, EvolutionResult(
                    success=True,
                    message=f"Evolved population: {len(new_population)} agents",
                    details={
                        'elites': len(elites),
                        'offspring': len(new_population) - len(elites),
                        'total': len(new_population)
                    }
                )
                
            except Exception as e:
                log_exception("EvolutionaryAlgorithm", e, "During population evolution")
                return population, EvolutionResult(
                    success=False,
                    message=f"Evolution failed: {str(e)}",
                    details={'error': str(e)}
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
        with LogFunction("EvolutionaryAlgorithm", "select_top_performers",
                        args=(),
                        kwargs={'population_size': len(population),
                               'num_select': num_select}):
            
            if num_select is None:
                num_select = max(1, int(len(population) * self.selection_ratio))
            
            log_info("EvolutionaryAlgorithm", f"Selecting top {num_select} performers")
            
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
                found = False
                for i, p in enumerate(population):
                    if p.get('agent_id') == agent.get('agent_id'):
                        selected_scores.append(fitness_scores[i])
                        found = True
                        break
                if not found:
                    # Fallback to mean if not found
                    fallback_score = np.mean(fitness_scores) if fitness_scores else 0.0
                    selected_scores.append(fallback_score)
                    log_warning("EvolutionaryAlgorithm", f"Agent {agent.get('agent_id')} not found in population, using mean score")
            
            log_success("EvolutionaryAlgorithm", f"Selected {len(selected)} top performers")
            log_data("EvolutionaryAlgorithm", "Selected scores", selected_scores)
            
            return selected, selected_scores