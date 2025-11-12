# fall_control.py
# Scenario: Fall control - minimize impact force during falls
# Controls all 20 joints to fall safely

import numpy as np
from scenarios.base_scenario import BaseScenario
import config


class FallControl(BaseScenario):
    """Fall control scenario - minimize impact force during falls."""
    
    scenario_name = 'fall_control'
    provides_acceleration = True
    
    def __init__(self, robot, timestep=32):
        self.CONTROL_JOINTS = config.JOINT_NAMES  # All 20 joints
        super().__init__(robot, timestep)
        
        # Get accelerometer for acceleration measurement (from proto: name "Accelerometer")
        self.accelerometer = None
        try:
            accel_device = self.robot.getDevice("Accelerometer")
            if accel_device:
                self.accelerometer = accel_device
                self.accelerometer.enable(self.timestep)
        except:
            pass
        
        # Get gyro for angular velocity (from proto: name "Gyro")
        self.gyro = None
        try:
            gyro_device = self.robot.getDevice("Gyro")
            if gyro_device:
                self.gyro = gyro_device
                self.gyro.enable(self.timestep)
        except:
            pass
        
        # Get IMU if available (from world file bodySlot: InertialUnit)
        self.imu = None
        try:
            imu_device = self.robot.getDevice("InertialUnit")
            if imu_device:
                self.imu = imu_device
                self.imu.enable(self.timestep)
        except:
            pass
        
        # Previous acceleration for detecting sudden changes
        self.prev_accel = np.zeros(3)
        self.prev_velocity = None
        
        # Standing position joint angles (approximate neutral/standing pose)
        self.STANDING_POSITIONS = {
            "ShoulderR": 0.0, "ShoulderL": 0.0,
            "ArmUpperR": 0.0, "ArmUpperL": 0.0,
            "ArmLowerR": 0.0, "ArmLowerL": 0.0,
            "PelvYR": 0.0, "PelvYL": 0.0,
            "PelvR": 0.0, "PelvL": 0.0,
            "LegUpperR": 0.0, "LegUpperL": 0.0,
            "LegLowerR": 0.0, "LegLowerL": 0.0,
            "AnkleR": 0.0, "AnkleL": 0.0,
            "FootR": 0.0, "FootL": 0.0,
            "Neck": 0.0, "Head": 0.0,
        }
    
    def _setup_environment(self):
        """Setup motors and sensors for all joints."""
        self.motors = []
        self.sensors = []
        
        for name in self.CONTROL_JOINTS:
            m = self.robot.getDevice(name)
            s = m.getPositionSensor()
            s.enable(self.timestep)
            m.setPosition(0.0)
            self.motors.append(m)
            self.sensors.append(s)
        
        # Get robot node for applying forces
        # Since robot is supervisor, we can use getSelf()
        self.robot_node = self.robot.getSelf()
    
    def get_obs_dim(self):
        """Return observation dimension (20 joint positions + 3 acceleration = 23)."""
        return 23  # 20 joint positions + 3D acceleration
    
    def get_act_dim(self):
        """Return action dimension (20 joint commands)."""
        return 20
    
    def get_acceleration(self):
        """Get current acceleration from accelerometer or estimate from velocity."""
        if self.accelerometer:
            accel = self.accelerometer.getValues()
            return np.array(accel, dtype=np.float32)
        else:
            # Fallback: estimate from velocity changes
            return np.zeros(3, dtype=np.float32)
    
    def reset(self):
        """
        Reset environment to standing position with random force applied to head.
        Must be called at the start of each episode.
        """
        self.episode_step = 0
        self.prev_accel = np.zeros(3)
        self.prev_velocity = None
        
        # Reset robot position and rotation (from proto: translation 0 0 0.279, rotation 0 0 1 0)
        # Using world file default: translation 0 0 0.292665
        if self.robot_node:
            try:
                # Reset translation to default standing position
                translation_field = self.robot_node.getField("translation")
                if translation_field:
                    translation_field.setSFVec3f([0.0, 0.0, 0.292665])
                
                # Reset rotation to default (0 0 1 0 means no rotation)
                rotation_field = self.robot_node.getField("rotation")
                if rotation_field:
                    rotation_field.setSFRotation([0.0, 0.0, 1.0, 0.0])
                
                # Reset physics state
                self.robot_node.resetPhysics()
            except Exception as e:
                pass
        
        # Reset all motors to standing position
        for i, name in enumerate(self.CONTROL_JOINTS):
            m = self.motors[i]
            standing_pos = self.STANDING_POSITIONS.get(name, 0.0)
            m.setPosition(standing_pos)
        
        # Wait for robot to settle into standing position
        for _ in range(50):  # Give it time to settle
            self.robot.step(self.timestep)
        
        # Apply random force to head from random direction
        # Random direction in 3D space (normalized)
        direction = np.random.randn(3)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        else:
            direction = np.array([1.0, 0.0, 0.0])  # Default to forward
        
        force_magnitude = 150.0
        force = direction * force_magnitude
        
        # Offset: 30cm above COM (head area) - in robot's local frame
        offset = [0.0, 0.3, 0.0]
        
        # Apply force with offset (relative to robot frame)
        if self.robot_node:
            try:
                # Use addForceWithOffset - force vector, offset vector, relative=True
                self.robot_node.addForceWithOffset(list(force), offset, True)
            except Exception as e:
                # Fallback: apply force at center of mass
                try:
                    self.robot_node.addForce(list(force), True)
                except:
                    pass
        
        # Step a few times to let force take effect
        for _ in range(10):
            self.robot.step(self.timestep)
        
        return self.get_observation()
    
    def get_observation(self):
        """Get current observation: joint positions + acceleration."""
        joint_positions = np.array([s.getValue() for s in self.sensors], dtype=np.float32)
        
        # Get acceleration
        if self.accelerometer:
            accel = self.accelerometer.getValues()
            accel_array = np.array(accel, dtype=np.float32)
        else:
            # Estimate acceleration from velocity changes
            try:
                if self.robot_node:
                    velocity = self.robot_node.getVelocity()
                    if velocity and len(velocity) >= 3:
                        current_velocity = np.array(velocity[:3], dtype=np.float32)  # Linear velocity
                        
                        if self.prev_velocity is not None:
                            # Estimate acceleration from velocity change
                            dt = self.timestep / 1000.0  # Convert to seconds
                            if dt > 0:
                                accel_array = (current_velocity - self.prev_velocity) / dt
                            else:
                                accel_array = np.zeros(3, dtype=np.float32)
                        else:
                            accel_array = np.zeros(3, dtype=np.float32)
                        
                        self.prev_velocity = current_velocity
                    else:
                        accel_array = np.zeros(3, dtype=np.float32)
                else:
                    accel_array = np.zeros(3, dtype=np.float32)
            except Exception as e:
                accel_array = np.zeros(3, dtype=np.float32)
        
        # Combine joint positions and acceleration
        obs = np.concatenate([joint_positions, accel_array])
        return obs
    
    def apply_action(self, action):
        """Apply action with joint-specific limits."""
        for i, (m, a) in enumerate(zip(self.motors, action)):
            jname = self.CONTROL_JOINTS[i]
            # Get limits from config, or use default
            if jname in config.ANGLE_LIMITS:
                low, high = config.ANGLE_LIMITS[jname]
            else:
                # Default limits for joints not in config
                low, high = -1.57, 1.57
            a = float(np.clip(a, low, high))
            m.setPosition(a)
    
    def compute_reward(self, obs, action, next_obs, step):
        """
        Compute reward based on impact force (acceleration magnitude).
        Lower acceleration = better (safer fall).
        Sudden acceleration changes = bad (impact).
        
        Args:
            obs: Previous observation (20 joint pos + 3 accel)
            action: Action taken
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
        
        # Extract acceleration from observation (last 3 elements)
        current_accel = next_obs[-3:]
        prev_accel = obs[-3:] if len(obs) >= 3 else np.zeros(3)
        
        # Acceleration magnitude (impact force indicator)
        accel_magnitude = np.linalg.norm(current_accel)
        
        # Sudden change in acceleration (impact detection)
        accel_change = np.linalg.norm(current_accel - prev_accel)
        
        # Reward: penalize high acceleration and sudden changes
        # Lower acceleration = better
        reward = -accel_magnitude * 0.1  # Base penalty for acceleration
        
        # Penalize sudden acceleration changes (impacts)
        reward -= accel_change * 0.5
        
        # Small survival bonus (encourage staying upright longer)
        reward += 0.01
        
        # Check if done
        # Episode ends if acceleration is very high (hard impact) or max steps reached
        done = accel_magnitude > 50.0 or step >= config.MAX_STEPS
        termination_reason = None
        
        # Large penalty for hard impact
        if accel_magnitude > 30.0:
            reward -= 10.0
        
        if accel_magnitude > 50.0:
            termination_reason = "hard_impact"
        elif step >= config.MAX_STEPS:
            termination_reason = "max_steps"
        
        return reward, done, termination_reason
    
    def get_episode_metric(self, obs):
        """Get acceleration magnitude for tracking (lower is better)."""
        if len(obs) >= 3:
            accel = obs[-3:]
            return np.linalg.norm(accel)
        return 0.0
    
    def is_success(self, obs, done):
        """Success is avoiding hard impact (acceleration < 30.0)."""
        if not done:
            return False
        if len(obs) >= 3:
            accel_magnitude = np.linalg.norm(obs[-3:])
            return accel_magnitude < 30.0
        return False

