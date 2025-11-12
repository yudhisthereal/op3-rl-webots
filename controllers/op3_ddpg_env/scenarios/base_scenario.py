# base_scenario.py
# Base class for training scenarios

from abc import ABC, abstractmethod
import numpy as np
import config


class BaseScenario(ABC):
    """Base class for all training scenarios."""
    
    # Each scenario must define its scenario_name as a class attribute
    scenario_name = None  # Override in subclasses
    provides_acceleration = False  # Override in subclasses if acceleration data available
    
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
        self.robot_node = robot.getSelf() if hasattr(robot, 'getSelf') else None
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
    
    def check_self_collision(self):
        """
        Check if robot parts are colliding with each other (self-collision).
        Uses multiple heuristics to detect impossible configurations that cause physics errors.
        
        Returns:
            has_collision: Boolean indicating if self-collision detected
            collision_info: String describing collision (or None)
        """
        if not self.robot_node:
            return False, None
        
        try:
            # Method 1: Check contact points (if available in Webots API)
            try:
                contact_points = self.robot_node.getContactPoints()
                if contact_points and len(contact_points) > 6:
                    # Excessive contact points often indicate self-collision
                    return True, f"Excessive contact points ({len(contact_points)}), possible self-collision"
            except:
                # Contact points API might not be available or work differently
                pass
            
            # Method 2: Check for extreme joint configurations
            # If multiple joints are at their limits simultaneously, likely self-collision
            if hasattr(self, 'sensors') and self.sensors:
                extreme_joints = 0
                for sensor in self.sensors:
                    try:
                        value = sensor.getValue()
                        # Check if joint is at extreme position (near limits)
                        if abs(value) > config.EXTREME_JOINT_ANGLE_THRESHOLD:
                            extreme_joints += 1
                    except:
                        pass
                
                # If many joints are at extreme positions, likely self-collision
                if extreme_joints >= 1:
                    return True, f"Multiple joints at extreme positions ({extreme_joints}), possible self-collision"
            
            # Method 3: Check robot's velocity for sudden spikes (indicates physics error)
            try:
                velocity = self.robot_node.getVelocity()
                if velocity and len(velocity) >= 3:
                    linear_vel = np.array(velocity[:3])
                    vel_magnitude = np.linalg.norm(linear_vel)
                    # Sudden high velocity often indicates physics error from self-collision
                    if vel_magnitude > 10.0:  # Unrealistic velocity
                        return True, f"Unrealistic velocity detected ({vel_magnitude:.2f} m/s), possible physics error"
            except:
                pass
            
        except Exception as e:
            # If all methods fail, return no collision
            pass
        
        return False, None
    
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
            termination_reason: String describing why episode ended (or None if not done)
        """
        pass
    
    def get_episode_metric(self, obs):
        """
        Get a metric value for tracking episode progress (e.g., distance to target).
        This is used for logging and statistics, not for reward computation.
        
        Args:
            obs: Current observation
            
        Returns:
            metric: Float value representing progress (e.g., distance to target)
        """
        # Default: return 0.0 if scenario doesn't define a metric
        return 0.0
    
    def is_success(self, obs, done):
        """
        Check if the episode ended in success.
        
        Args:
            obs: Current observation
            done: Whether episode is done
            
        Returns:
            success: Boolean indicating if episode ended successfully
        """
        # Default: success if done (can be overridden by scenarios)
        return done
    
    def step(self):
        """Advance the simulation by one timestep."""
        self.robot.step(self.timestep)
        self.episode_step += 1

