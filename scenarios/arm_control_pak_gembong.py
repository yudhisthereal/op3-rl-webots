# arm_control_pak_gembong.py
# Scenario: Rotate ShoulderL → +1.57 rad, ArmUpperL → -1.57 rad
# Reward function: Pak Gembong's version

import numpy as np
from scenarios.base_scenario import BaseScenario
import config


class ArmControlPakGembong(BaseScenario):
    """Arm control scenario with Pak Gembong's reward function."""
    
    scenario_name = 'pak_gembong'
    
    def __init__(self, robot, timestep=32):
        self.CONTROL_JOINTS = ["ShoulderL", "ArmUpperL"]
        self.TARGET = np.array([1.57, -1.57], dtype=np.float32)
        self.dist_prev = -1
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
        self.dist_prev = -1
        
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
        Compute reward using Pak Gembong's method.
        
        Args:
            obs: Previous observation (not used in this reward)
            action: Action taken (not used in this reward)
            next_obs: Current observation after action
            step: Current step in episode
            
        Returns:
            reward: Scalar reward
            done: Boolean indicating if episode is done
            termination_reason: String describing why episode ended (or None if not done)
        """
        # Check for self-collision first
        has_collision, collision_info = self.check_self_collision()
        if has_collision:
            print(f"Self-collision detected: {collision_info}")
            return -100.0, True, collision_info
        
        reward = -0.1
        
        dist = np.linalg.norm(next_obs - self.TARGET)
        
        # Penalty if moving away from target
        if self.dist_prev != -1:
            if self.dist_prev < dist:
                reward = -1.0
        
        self.dist_prev = dist
        
        # Check if done
        done = (next_obs == self.TARGET).all() or step >= config.MAX_STEPS
        termination_reason = None
        
        if (next_obs == self.TARGET).all():
            reward = 1.0
            termination_reason = "target_reached"
        elif step >= config.MAX_STEPS:
            termination_reason = "max_steps"
        
        return reward, done, termination_reason
    
    def get_episode_metric(self, obs):
        """Get distance to target for tracking."""
        return np.linalg.norm(obs - self.TARGET)
    
    def is_success(self, obs, done):
        """Check if target is reached exactly or within threshold."""
        if not done:
            return False
        dist = np.linalg.norm(obs - self.TARGET)
        return (obs == self.TARGET).all()

