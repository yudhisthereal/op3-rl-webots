# fall_control.py for SAC
# Modified with simplified reward based only on joint angle errors

import numpy as np
from scenarios.base_scenario import BaseScenario
import config


class FallControl(BaseScenario):
    """Fall control scenario with simplified joint error reward."""
    
    scenario_name = 'fall_control_sac'  # Different name for SAC
    provides_acceleration = False
    
    def __init__(self, robot, timestep=32):
        self.CONTROL_JOINTS = config.JOINT_NAMES  # All 20 joints
        
        # Goal joint positions for "protective falling pose"
        self.GOAL_POSITIONS = {
            "ShoulderR": 1.5, "ShoulderL": -1.5,
            "ArmUpperR": -1.25, "ArmUpperL": 1.25,
            "ArmLowerR": 0.3, "ArmLowerL": -0.3,
            "PelvR": 0.4, "PelvL": -0.4,
            "LegUpperR": 0.8, "LegUpperL": -0.8,
            "LegLowerR": -0.3, "LegLowerL": 0.3,
            "AnkleR": 0.0, "AnkleL": 0.0,
            "FootR": 0.0, "FootL": 0.0,
            "Neck": 0.0, "Head": 0.0,
        }
        
        # Standing position joint angles
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
        
        super().__init__(robot, timestep)
        
        # Push parameters
        self.push_delay_steps = int(0.5 * 1000 / timestep)
        self.push_applied = False
        self.push_step = 0
        
        # Get robot node for CoM tracking
        self.robot_node = self.robot.getSelf()
        
        # Initialize CoM tracking
        self.initial_com_height = 0.0
        self.current_com_height = 0.0
    
    def _setup_environment(self):
        """Setup motors and sensors for all joints."""
        self.motors = []
        self.sensors = []
        
        for name in self.CONTROL_JOINTS:
            m = self.robot.getDevice(name)
            s = m.getPositionSensor()
            s.enable(self.timestep)
            m.setPosition(self.STANDING_POSITIONS.get(name, 0.0))
            self.motors.append(m)
            self.sensors.append(s)
    
    def get_obs_dim(self):
        """Return observation dimension (20 joint positions + 1 CoM height)."""
        return 21  # 20 joint positions + CoM height
    
    def get_act_dim(self):
        """Return action dimension (20 joint commands)."""
        return 20
    
    def get_com_height(self):
        """Get current center of mass height from ground."""
        if self.robot_node:
            try:
                translation = self.robot_node.getField("translation").getSFVec3f()
                return translation[2]  # Z coordinate is height
            except:
                pass
        return 0.0
    
    def reset(self):
        """Reset environment to standing position."""
        self.episode_step = 0
        self.push_applied = False
        self.push_step = 0
        
        # Reset all motors to standing position
        for i, name in enumerate(self.CONTROL_JOINTS):
            if self.robot_node:
                try:
                    translation_field = self.robot_node.getField("translation")
                    if translation_field:
                        translation_field.setSFVec3f([0.0, 0.0, 0.292665])
                    
                    rotation_field = self.robot_node.getField("rotation")
                    if rotation_field:
                        rotation_field.setSFRotation([0.0, 0.0, 1.0, 0.0])
                    
                    self.robot_node.resetPhysics()
                except Exception as e:
                    pass
            
            m = self.motors[i]
            standing_pos = self.STANDING_POSITIONS.get(name, 0.0)
            m.setPosition(standing_pos)
        
        # Wait for robot to settle
        settle_steps = self.push_delay_steps
        for _ in range(settle_steps):
            self.robot.step(self.timestep)
            self.episode_step += 1
        
        # Record initial CoM height
        self.initial_com_height = self.get_com_height()
        self.current_com_height = self.initial_com_height
        
        return self.get_observation()
    
    def step(self):
        """Override step to apply push after delay."""
        if not self.push_applied and self.episode_step >= self.push_delay_steps:
            self.apply_forward_push()
            self.push_applied = True
            self.push_step = self.episode_step
        
        return super().step()
    
    def apply_forward_push(self):
        """Apply a forward push to the robot's head."""
        if not self.robot_node:
            return
        
        force_magnitude = 500.0
        force = [force_magnitude, 0.0, 0.0]
        offset = [0.0, 0.0, 0.3]
        
        try:
            self.robot_node.addForceWithOffset(force, offset, True)
            print(f"Applied forward push to head at step {self.episode_step}")
        except Exception as e:
            try:
                self.robot_node.addForce(force, True)
                print(f"Applied forward push to CoM at step {self.episode_step}")
            except Exception as e2:
                print(f"Failed to apply push: {e2}")
    
    def get_observation(self):
        """Get current observation: joint positions + CoM height."""
        joint_positions = np.array([s.getValue() for s in self.sensors], dtype=np.float32)
        self.current_com_height = self.get_com_height()
        com_height_array = np.array([self.current_com_height], dtype=np.float32)
        obs = np.concatenate([joint_positions, com_height_array])
        return obs
    
    def apply_action(self, action):
        """Apply action with joint-specific limits."""
        for i, (m, a) in enumerate(zip(self.motors, action)):
            jname = self.CONTROL_JOINTS[i]
            if jname in config.ANGLE_LIMITS:
                low, high = config.ANGLE_LIMITS[jname]
            else:
                low, high = -1.57, 1.57
            a = float(np.clip(a, low, high))
            m.setPosition(a)
    
    def compute_reward(self, obs, action, next_obs, step):
        """
        SIMPLIFIED REWARD FOR SAC:
        Based ONLY on joint angle errors (current - goal angles).
        
        Args:
            obs: Previous observation (20 joint pos + 1 CoM)
            action: Action taken
            next_obs: Current observation after action
            step: Current step in episode
            
        Returns:
            reward: Scalar reward based on joint angle errors
            done: Boolean indicating if episode is done
            termination_reason: String describing why episode ended
        """
        # Extract current joint positions (first 20 elements)
        current_joints = next_obs[:20] if len(next_obs) >= 20 else next_obs
        
        # Extract previous joint positions
        prev_joints = obs[:20] if len(obs) >= 20 else np.zeros(20)
        
        # 1. ERROR REWARD - how close to goal positions
        total_error = 0.0
        error_components = []
        
        for i, joint_name in enumerate(self.CONTROL_JOINTS[:len(current_joints)]):
            current_pos = current_joints[i]
            goal_pos = self.GOAL_POSITIONS.get(joint_name, 0.0)
            
            # Absolute distance from goal
            error = abs(current_pos - goal_pos)
            total_error += error
            error_components.append(error)
        
        # Average error across all joints
        avg_error = total_error / len(current_joints) if len(current_joints) > 0 else 1.0
        
        # Reward inversely proportional to error
        # Using negative exponential: higher reward for smaller error
        error_reward = np.exp(-avg_error * 3.0) * 2.0
        
        # 2. PROGRESS BONUS - reward for moving toward goal
        progress_bonus = 0.0
        if len(prev_joints) == len(current_joints):
            for i, joint_name in enumerate(self.CONTROL_JOINTS[:len(current_joints)]):
                current_pos = current_joints[i]
                goal_pos = self.GOAL_POSITIONS.get(joint_name, 0.0)
                prev_pos = prev_joints[i]
                
                # Check if we moved toward the goal
                prev_error = abs(prev_pos - goal_pos)
                current_error = abs(current_pos - goal_pos)
                
                if current_error < prev_error:
                    # We moved toward goal - give bonus
                    progress = prev_error - current_error
                    progress_bonus += progress * 0.5
        
        # 3. SMOOTHNESS PENALTY - small penalty for jerky movements
        smoothness_penalty = 0.0
        if len(prev_joints) == len(current_joints):
            for i in range(len(current_joints)):
                movement = abs(current_joints[i] - prev_joints[i])
                smoothness_penalty += movement * 0.1
        
        # 4. ACTION MAGNITUDE PENALTY - encourage smaller actions
        action_penalty = np.mean(np.abs(action)) * 0.05
        
        # 5. EXTREME POSITION PENALTY - penalize joints near limits
        extreme_penalty = 0.0
        for i, joint_name in enumerate(self.CONTROL_JOINTS[:len(current_joints)]):
            current_pos = current_joints[i]
            
            # Get joint limits
            if joint_name in config.ANGLE_LIMITS:
                low, high = config.ANGLE_LIMITS[joint_name]
                
                # Penalize if close to limits
                if abs(current_pos - low) < 0.1 or abs(current_pos - high) < 0.1:
                    extreme_penalty += 0.1
        
        # TOTAL REWARD
        total_reward = (
            error_reward +                # Main error-based reward
            progress_bonus -              # Bonus for progress
            smoothness_penalty -          # Penalty for jerky movements
            action_penalty -              # Penalty for large actions
            extreme_penalty               # Penalty for extreme positions
        )
        
        # Termination conditions
        done = False
        termination_reason = None
        
        # Max steps
        if step >= config.MAX_STEPS:
            done = True
            termination_reason = "max_steps"
            # Small bonus for completing episode with low error
            if avg_error < 0.3:
                total_reward += 1.0
        
        # Check if robot has fallen (CoM too low)
        if len(next_obs) >= 21:
            current_com = next_obs[-1]
            fallen_threshold = 0.15
            if current_com < fallen_threshold:
                done = True
                termination_reason = "fallen"
                total_reward -= 2.0
        
        # Check for extreme joint positions
        extreme_count = sum(1 for error in error_components if error > 1.0)
        if extreme_count > 5:
            done = True
            termination_reason = "extreme_joint_positions"
            total_reward -= 1.0
        
        # Debug output
        if step % 50 == 0:
            print(f"Step {step}: Avg error={avg_error:.3f}, Reward={total_reward:.3f}")
        
        return total_reward, done, termination_reason
    
    def get_episode_metric(self, obs):
        """Get metric for tracking (average joint error)."""
        if len(obs) >= 20:
            current_joints = obs[:20]
            total_error = 0.0
            
            for i, joint_name in enumerate(self.CONTROL_JOINTS[:20]):
                current_pos = current_joints[i]
                goal_pos = self.GOAL_POSITIONS.get(joint_name, 0.0)
                total_error += abs(current_pos - goal_pos)
            
            return total_error / 20
        return 1.0  # Default high error
    
    def is_success(self, obs, done):
        """Success is low joint error and not fallen."""
        if not done:
            return False
        
        # Check joint error
        if len(obs) >= 20:
            current_joints = obs[:20]
            total_error = 0.0
            
            for i, joint_name in enumerate(self.CONTROL_JOINTS[:20]):
                current_pos = current_joints[i]
                goal_pos = self.GOAL_POSITIONS.get(joint_name, 0.0)
                total_error += abs(current_pos - goal_pos)
            
            avg_error = total_error / 20
            return avg_error < 0.2  # Success if average error < 0.2 rad
        
        return False