# arm_control_yudhis.py
# Scenario: Rotate ShoulderL → +1.57 rad, ArmUpperL → -1.57 rad
# Reward function: Yudhis's version

import numpy as np
from scenarios.base_scenario import BaseScenario
import config


class ArmControlYudhis(BaseScenario):
    """Arm control scenario with Yudhis's reward function."""
    
    scenario_name = 'yudhis'
    
    def __init__(self, robot, timestep=32):
        self.CONTROL_JOINTS = ["ShoulderL", "ArmUpperL"]
        self.TARGET = np.array([1.57, -1.57], dtype=np.float32)
        super().__init__(robot, timestep)
    
    def _setup_environment(self):
        """Setup motors and sensors for controlled joints."""
        self.motors = []
        self.sensors = []
        
        for name in self.CONTROL_JOINTS:
            m = self.robot.getDevice(name)
            s = m.getPositionSensor()
            s.enable(self.timestep)
            m.setPosition(0.0)
            self.motors.append(m)
            self.sensors.append(s)
    
    def get_obs_dim(self):
        """Return observation dimension (2 joint positions)."""
        return 2
    
    def get_act_dim(self):
        """Return action dimension (2 joint commands)."""
        return 2
    
    def reset(self):
        """
        Reset environment to initial state (all joints at zero).
        Must be called at the start of each episode.
        """
        self.episode_step = 0
        
        # Reset all motors to zero
        for m in self.motors:
            m.setPosition(0.0)
        
        # Wait until sensor readings converge near zero
        while True:
            self.robot.step(self.timestep)
            obs = self.get_observation()
            if np.allclose(obs, [0.0, 0.0], atol=1e-3):
                break
        
        return self.get_observation()
    
    def get_observation(self):
        """Get current joint positions."""
        return np.array([s.getValue() for s in self.sensors], dtype=np.float32)
    
    def apply_action(self, action):
        """Apply action with joint-specific limits."""
        for i, (m, a) in enumerate(zip(self.motors, action)):
            jname = self.CONTROL_JOINTS[i]
            low, high = config.ANGLE_LIMITS[jname]
            a = float(np.clip(a, low, high))
            m.setPosition(a)
    
    def compute_reward(self, obs, action, next_obs, step):
        """
        Compute reward using Yudhis's method.
        
        Args:
            obs: Previous observation (not used in this reward)
            action: Action taken (not used in this reward)
            next_obs: Current observation after action
            step: Current step in episode
            
        Returns:
            reward: Scalar reward
            done: Boolean indicating if episode is done
        """
        dist = np.linalg.norm(next_obs - self.TARGET)
        reward = -dist**2
        
        # Check if done
        done = dist < 0.01 or step >= config.MAX_STEPS
        
        # Success bonus
        if dist < 0.01:
            reward += 5.0
        
        return reward, done
    
    def get_episode_metric(self, obs):
        """Get distance to target for tracking."""
        return np.linalg.norm(obs - self.TARGET)
    
    def is_success(self, obs, done):
        """Check if target is reached (distance < 0.01)."""
        if not done:
            return False
        dist = np.linalg.norm(obs - self.TARGET)
        return dist < 0.01

