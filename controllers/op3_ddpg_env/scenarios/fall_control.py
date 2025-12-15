# fall_control.py
# Modified Scenario: Fall control - focus on CoM maintenance and goal joint following
# Only pushes forward after settling

import numpy as np
from scenarios.base_scenario import BaseScenario
import config


class FallControl(BaseScenario):
    """Fall control scenario - maintain CoM height and follow goal joints."""
    
    scenario_name = 'fall_control'
    provides_acceleration = False  # No longer using acceleration
    
    def __init__(self, robot, timestep=32):
        self.CONTROL_JOINTS = config.JOINT_NAMES  # All 20 joints
        
        # Goal joint positions for "protective falling pose"
        # Arms extended forward, knees slightly bent, torso upright
        self.GOAL_POSITIONS = {
            "ShoulderR": 1.5, "ShoulderL": -1.5,
            "ArmUpperR": -1.25, "ArmUpperL": 1.25,
            "ArmLowerR": 0.3, "ArmLowerL": -0.3,
            "PelvR": 0.4, "PelvL": -0.4,
            "LegUpperR": 0.8, "LegUpperL": -0.8,
            "LegLowerR": -0.3, "LegLowerL": 0.3,
            
            # Feet flat
            "AnkleR": 0.0,
            "AnkleL": 0.0,
            "FootR": 0.0,
            "FootL": 0.0,
            
            # Head upright
            "Neck": 0.0,
            "Head": 0.0,
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
        self.push_delay_steps = int(0.5 * 1000 / timestep)  # 0.5 seconds delay
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
            # Start at standing position
            m.setPosition(self.STANDING_POSITIONS.get(name, 0.0))
            self.motors.append(m)
            self.sensors.append(s)
    
    def get_obs_dim(self):
        """Return observation dimension (14 joint positions + 1 CoM height)."""
        return 15  # 20 joint positions + CoM height
    
    def get_act_dim(self):
        """Return action dimension (14 joint commands)."""
        return 14
    
    def get_com_height(self):
        """Get current center of mass height from ground."""
        if self.robot_node:
            try:
                # Get robot's global position (CoM approximation)
                translation = self.robot_node.getField("translation").getSFVec3f()
                return translation[2]  # Z coordinate is height
            except:
                pass
        return 0.0
    
    def reset(self):
        """
        Reset environment to standing position.
        Push will be applied after delay.
        """
        self.episode_step = 0
        self.push_applied = False
        self.push_step = 0
        
        # Reset all motors to standing position
        for i, name in enumerate(self.CONTROL_JOINTS):
            if self.robot_node:
                try:
                    # Reset translation to default standing position
                    translation_field = self.robot_node.getField("translation")
                    if translation_field:
                        translation_field.setSFVec3f([0.0, 0.0, 0.292665])
                    
                    # Reset rotation to default
                    rotation_field = self.robot_node.getField("rotation")
                    if rotation_field:
                        rotation_field.setSFRotation([0.0, 0.0, 1.0, 0.0])
                    
                    # Reset physics state
                    self.robot_node.resetPhysics()
                except Exception as e:
                    pass
            m = self.motors[i]
            standing_pos = self.STANDING_POSITIONS.get(name, 0.0)
            m.setPosition(standing_pos)
        
        # Wait for robot to settle into standing position (0.5 seconds)
        settle_steps = self.push_delay_steps
        for _ in range(settle_steps):
            self.robot.step(self.timestep)
            # Update step counter
            self.episode_step += 1
        
        # Record initial CoM height after settling
        self.initial_com_height = self.get_com_height()
        self.current_com_height = self.initial_com_height
        
        return self.get_observation()
    
    def step(self):
        """Override step to apply push after delay."""
        # Apply forward push after 0.5 seconds delay
        if not self.push_applied and self.episode_step >= self.push_delay_steps:
            self.apply_forward_push()
            self.push_applied = True
            self.push_step = self.episode_step
        
        # Call parent step
        return super().step()
    
    def apply_forward_push(self):
        """Apply a forward push to the robot's head (Webots coordinates)."""
        if not self.robot_node:
            return
        
        # Apply forward force at head
        force_magnitude = 500.0  # Forward push force in Newtons
        force = [force_magnitude, 0.0, 0.0]  # Forward in +X direction
        
        try:
            # In Webots: X=forward, Y=left/right, Z=up/down
            # Head is approximately at [0, 0, 0.3] relative to robot's CoM
            # X=0 (centered), Y=0 (centered), Z=0.3m up
            offset = [0.0, 0.0, 0.3]  # CORRECT: Z is vertical
            
            # Apply force at head location (creates rotational torque)
            self.robot_node.addForceWithOffset(force, offset, True)
            print(f"Applied forward push to head at step {self.episode_step} (offset: {offset})")
            
        except Exception as e:
            # Fallback if addForceWithOffset fails
            try:
                # Try regular addForce as fallback (applies at CoM)
                self.robot_node.addForce(force, True)
                print(f"Applied forward push to CoM (fallback) at step {self.episode_step}")
            except Exception as e2:
                print(f"Failed to apply push: {e2}")
    
    def get_observation(self):
        """Get current observation: joint positions + CoM height."""
        # Get joint positions
        joint_positions = np.array([s.getValue() for s in self.sensors], dtype=np.float32)
        
        # Get current CoM height
        self.current_com_height = self.get_com_height()
        com_height_array = np.array([self.current_com_height], dtype=np.float32)
        
        # Combine joint positions and CoM height
        obs = np.concatenate([joint_positions, com_height_array])
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
        SIMPLIFIED REWARD: Only error penalty and smoothness penalty.
        
        Two components:
        1. Error penalty - penalize distance from goal positions (MAIN)
        2. Smoothness penalty - penalize jerky movements (SMALL)
        """
        # Extract joint positions
        current_joints = next_obs  # Current state after action
        prev_joints = obs          # Previous state before action
        
        # 1. ERROR PENALTY (MAIN - how far from goal positions)
        total_error = 0.0
        
        for i, joint_name in enumerate(self.CONTROL_JOINTS):
            current_pos = current_joints[i]
            goal_pos = self.GOAL_POSITIONS.get(joint_name, 0.0)
            
            # Distance from goal (absolute error)
            error = abs(current_pos - goal_pos)
            total_error += error
        
        # Average error across all joints
        avg_error = total_error / len(self.CONTROL_JOINTS)
        
        # Penalize error: higher error = more negative reward
        # Using squared error to emphasize large errors
        error_penalty = (avg_error * 10)**2
        
        # 2. SMOOTHNESS PENALTY (SMALL - penalize jerky movements)
        smoothness_penalty = 0.0
        
        for i in range(len(self.CONTROL_JOINTS)):
            # How much did the joint move?
            movement = abs(current_joints[i] - prev_joints[i])
            
            # Small penalty for movement
            smoothness_penalty += movement * 0.3  # Small weight
        
        # TOTAL REWARD: error penalty (main) + smoothness penalty (small)
        total_reward = - error_penalty - smoothness_penalty + self.current_com_height
        
        # Termination conditions (keep your existing termination logic)
        done = False
        termination_reason = None
        
        # Max steps
        if step >= config.MAX_STEPS:
            done = True
            termination_reason = "max_steps"
            # Small bonus for surviving entire episode
            if avg_error < 0.5:  # If close to goal at the end
                total_reward += 2.0
        
        # Check if robot has fallen (estimate from joint positions)
        # Simple heuristic: if average joint position is extreme
        extreme_positions = 0
        for i, pos in enumerate(current_joints):
            if abs(pos) > 2.0:  # Very extreme position
                extreme_positions += 1
        
        if extreme_positions > 5:  # Too many extreme positions
            done = True
            termination_reason = "unstable_position"
            total_reward -= 3.0
            
        # Check if robot has fallen (CoM too low)
        fallen_threshold = 0.15  # 15cm from ground
        if self.current_com_height < fallen_threshold:
            done = True
            termination_reason = "fallen"
            # total_reward -= 10.0  # Larger penalty for falling
        
        return total_reward, done, termination_reason
    
    # def compute_reward(self, obs, action, next_obs, step):
    #     """
    #     Compute reward with adaptive smoothness:
    #     - Fast movements when far from goal (to catch up)
    #     - Smooth movements when close to goal (to avoid overshoot)
        
    #     Args:
    #         obs: Previous observation (20 joint pos + 1 CoM)
    #         action: Action taken
    #         next_obs: Current observation after action
    #         step: Current step in episode
            
    #     Returns:
    #         reward: Scalar reward
    #         done: Boolean indicating if episode is done
    #         termination_reason: String describing why episode ended
    #     """
    #     # Check for self-collision
    #     has_collision, collision_info = self.check_self_collision()
    #     if has_collision:
    #         print(f"Self-collision detected: {collision_info}")
    #         return -50.0, True, collision_info
        
    #     # Extract current joint positions and CoM height
    #     current_joints = next_obs[:20]
    #     current_com = next_obs[-1]  # Last element is CoM height
        
    #     # Extract previous joint positions
    #     prev_joints = obs[:20] if len(obs) >= 20 else np.zeros(20)
        
    #     # 1. CoM HEIGHT REWARD (higher is better)
    #     # Normalize by initial height, penalize falling
    #     height_ratio = current_com / max(self.initial_com_height, 0.001)
    #     if height_ratio > 1.0:
    #         height_ratio = 1.0  # Cap at initial height
        
    #     height_reward = height_ratio * 2.0  # Scale factor
        
    #     # 2. ADAPTIVE GOAL FOLLOWING REWARD
    #     joint_error_sum = 0.0
    #     joint_velocity_sum = 0.0
    #     adaptive_smoothness_penalty = 0.0
        
    #     for i, joint_name in enumerate(self.CONTROL_JOINTS):
    #         current_pos = current_joints[i]
    #         goal_pos = self.GOAL_POSITIONS.get(joint_name, 0.0)
    #         prev_pos = prev_joints[i]
            
    #         # Compute distance to goal
    #         error = abs(current_pos - goal_pos)
    #         joint_error_sum += error
            
    #         # Compute joint velocity (change from previous position)
    #         velocity = abs(current_pos - prev_pos)
    #         joint_velocity_sum += velocity
            
    #         # ADAPTIVE SMOOTHNESS: penalty changes based on distance to goal
    #         # When far from goal: allow faster movements (low penalty)
    #         # When close to goal: encourage smoother movements (higher penalty)
            
    #         # Sigmoid function: transition between fast and smooth zones
    #         # error_threshold = 0.3 rad (~17 degrees)
    #         # When error > 0.3: fast movement zone (low penalty)
    #         # When error < 0.3: smooth movement zone (higher penalty)
            
    #         # Adaptive penalty factor (0 to 1)
    #         smoothness_factor = 1.0 / (1.0 + np.exp(10.0 * (error - 0.3)))
    #         # smoothness_factor ~0 when error > 0.5 (fast zone)
    #         # smoothness_factor ~1 when error < 0.1 (smooth zone)
            
    #         # Base velocity penalty
    #         base_penalty = velocity * 0.8
            
    #         # Apply adaptive factor
    #         adaptive_penalty = base_penalty * smoothness_factor
            
    #         # Add small constant penalty even in fast zone to prevent oscillations
    #         minimal_penalty = velocity * 0.1  # 10% of base penalty always
            
    #         adaptive_smoothness_penalty += (adaptive_penalty + minimal_penalty)
        
    #     # Average joint error and velocity
    #     avg_joint_error = joint_error_sum / len(self.CONTROL_JOINTS)
    #     avg_joint_velocity = joint_velocity_sum / len(self.CONTROL_JOINTS)
        
    #     # Reward for low joint error (follow goal positions)
    #     # Inverse exponential: smaller error = higher reward
    #     joint_reward = np.exp(-avg_joint_error * 3.0) * 1.5
        
    #     # 3. TIME-DEPENDENT AGGRESSION FACTOR
    #     # As robot starts falling, allow more aggressive movements
    #     aggression_factor = 1.0
        
    #     if self.push_applied:
    #         # Time since push
    #         time_since_push = step - self.push_step
            
    #         if time_since_push < 15:  # First 20 steps after push
    #             # Emergency phase: allow very fast movements
    #             aggression_factor = 0.5  # Reduce smoothness penalty by 50%
    #         elif time_since_push < 30:  # Next 30 steps
    #             # Recovery phase: moderate smoothness
    #             aggression_factor = 0.8
    #         # After 50 steps: normal smoothness (aggression_factor = 1.0)
        
    #     # Apply aggression factor to smoothness penalty
    #     adaptive_smoothness_penalty *= aggression_factor
        
    #     # 4. PROGRESS REWARD (bonus for moving toward goal)
    #     progress_reward = 0.0
    #     for i, joint_name in enumerate(self.CONTROL_JOINTS):
    #         current_pos = current_joints[i]
    #         goal_pos = self.GOAL_POSITIONS.get(joint_name, 0.0)
    #         prev_pos = prev_joints[i]
            
    #         # Check if we moved toward the goal
    #         prev_error = abs(prev_pos - goal_pos)
    #         current_error = abs(current_pos - goal_pos)
            
    #         if current_error < prev_error:
    #             # We moved toward goal - give bonus
    #             progress = prev_error - current_error
    #             # Scale bonus by how far we are from goal
    #             # Bigger bonus when far from goal to encourage fast progress
    #             distance_factor = min(prev_error * 2.0, 1.0)
    #             progress_reward += progress * distance_factor * 0.5
        
    #     # 5. COORDINATION BONUS (reward synchronized movements)
    #     coordination_bonus = 0.0
        
    #     # Check if arm joints are moving together
    #     arm_joint_names = ["ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL"]
    #     arm_indices = [i for i, name in enumerate(self.CONTROL_JOINTS) 
    #                 if name in arm_joint_names]
        
    #     if arm_indices:
    #         # Calculate variance in arm joint velocities
    #         arm_velocities = []
    #         for idx in arm_indices:
    #             vel = abs(current_joints[idx] - prev_joints[idx])
    #             arm_velocities.append(vel)
            
    #         if arm_velocities:
    #             vel_variance = np.var(arm_velocities)
    #             # Low variance = synchronized movements = good
    #             coordination_bonus = np.exp(-vel_variance * 5.0) * 0.3
        
    #     # Total reward
    #     total_reward = (
    #         height_reward +
    #         joint_reward +
    #         progress_reward +
    #         coordination_bonus -
    #         adaptive_smoothness_penalty  # Note: this is a penalty, so subtract
    #     )
        
    #     print(f"height_reward: {height_reward}\njoint_reward: {joint_reward}\nprogress_reward: {progress_reward}\ncoordination_bonus: {coordination_bonus}\nsmoothness_penalty: {adaptive_smoothness_penalty}")
        
    #     # 6. EMERGENCY OVERSHOOT PENALTY (prevent dangerous fast movements)
    #     emergency_overshoot_penalty = 0.0
    #     for i, joint_name in enumerate(self.CONTROL_JOINTS):
    #         current_pos = current_joints[i]
    #         goal_pos = self.GOAL_POSITIONS.get(joint_name, 0.0)
    #         prev_pos = prev_joints[i]
            
    #         # Check if we overshot the goal by a lot
    #         if abs(current_pos - goal_pos) > abs(prev_pos - goal_pos) + 0.5:
    #             # Overshot by more than 0.5 rad from previous error
    #             emergency_overshoot_penalty += 0.5
        
    #     total_reward -= emergency_overshoot_penalty
        
    #     # Termination conditions
    #     done = False
    #     termination_reason = None
        
    #     # Check if robot has fallen (CoM too low)
    #     fallen_threshold = 0.15  # 15cm from ground
    #     if current_com < fallen_threshold:
    #         done = True
    #         termination_reason = "fallen"
    #         total_reward -= 10.0  # Larger penalty for falling
        
    #     # Check if joints moved too aggressively (dangerous)
    #     max_safe_velocity = 15.0  # rad/s equivalent
    #     if avg_joint_velocity * (1000.0 / self.timestep) > max_safe_velocity:
    #         done = True
    #         termination_reason = f"dangerous_movement {avg_joint_velocity * (1000.0 / self.timestep)}"
    #         total_reward -= 8.0
        
    #     # Max steps
    #     elif step >= config.MAX_STEPS:
    #         done = True
    #         termination_reason = "max_steps"
    #         # Bonus scaled by how well we maintained height
    #         height_bonus = height_ratio * 8.0
    #         total_reward += height_bonus
        
    #     # Success: maintained height and followed goals
    #     if done and termination_reason == "max_steps" and height_ratio > 0.7:
    #         termination_reason = "success"
    #         # Additional success bonus
    #         success_bonus = (1.0 - avg_joint_error) * 5.0
    #         total_reward += success_bonus
        
    #     if termination_reason:
    #         print(f"TERMINATED FOR: {termination_reason}")
        
    #     return total_reward, done, termination_reason  
      
    def get_episode_metric(self, obs):
        """Get CoM height ratio for tracking (higher is better)."""
        if len(obs) >= 21:
            current_com = obs[-1]
            if self.initial_com_height > 0:
                return current_com / self.initial_com_height
        return 0.0
    
    def is_success(self, obs, done):
        """Success is maintaining CoM height > 70% of initial."""
        if not done:
            return False
        
        if len(obs) >= 21:
            current_com = obs[-1]
            if self.initial_com_height > 0:
                height_ratio = current_com / self.initial_com_height
                return height_ratio > 0.7
        
        return False