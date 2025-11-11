# base_scenario.py
# Base class for training scenarios

from abc import ABC, abstractmethod
import numpy as np


class BaseScenario(ABC):
    """Base class for all training scenarios."""
    
    def __init__(self, robot, timestep=32):
        """
        Initialize the scenario.
        
        Args:
            robot: Webots Supervisor instance
            timestep: Simulation timestep
        """
        self.robot = robot
        self.timestep = timestep
        self.episode_step = 0
        self._setup_environment()
    
    @abstractmethod
    def _setup_environment(self):
        """Setup motors, sensors, and other environment components."""
        pass
    
    @abstractmethod
    def get_obs_dim(self):
        """Return the observation/state space dimension."""
        pass
    
    @abstractmethod
    def get_act_dim(self):
        """Return the action space dimension."""
        pass
    
    @abstractmethod
    def reset(self):
        """
        Reset the environment for a new episode.
        Must be called at the start of each episode.
        
        Returns:
            Initial observation (numpy array)
        """
        pass
    
    @abstractmethod
    def get_observation(self):
        """
        Get the current observation/state.
        
        Returns:
            Current observation (numpy array)
        """
        pass
    
    @abstractmethod
    def apply_action(self, action):
        """
        Apply an action to the environment.
        
        Args:
            action: Action to apply (numpy array)
        """
        pass
    
    @abstractmethod
    def compute_reward(self, obs, action, next_obs, step):
        """
        Compute the reward for a transition.
        
        Args:
            obs: Previous observation
            action: Action taken
            next_obs: Next observation
            step: Current step in episode
            
        Returns:
            reward: Scalar reward
            done: Boolean indicating if episode is done
        """
        pass
    
    def step(self):
        """Advance the simulation by one timestep."""
        self.robot.step(self.timestep)
        self.episode_step += 1

