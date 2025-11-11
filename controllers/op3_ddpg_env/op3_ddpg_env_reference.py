# op3_ddpg_env.py
# Webots supervisor controller for ROBOTIS OP3 (supervisor = TRUE)
# Goal: rotate ShoulderL → +1.57 rad, ArmUpperL → -1.57 rad
# https://chat.deepseek.com/share/cxjo4yabx9ot4zi4ic

from controller import Supervisor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

# ================== CONFIG ==================
TIMESTEP = 32
JOINT_NAMES = [
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR",
    "ArmLowerL", "PelvYR", "PelvYL", "PelvR", "PelvL",
    "LegUpperR", "LegUpperL", "LegLowerR", "LegLowerL", "AnkleR",
    "AnkleL", "FootR", "FootL", "Neck", "Head"
]

CONTROL_JOINTS = ["ShoulderL", "ArmUpperL"]
TARGET = np.array([1.57, -1.57], dtype=np.float32)
MAX_STEPS = 200
MAX_EPISODES = 300

# Per-joint mechanical limits (Webots OP3)
ANGLE_LIMITS = {
    "ShoulderL": (-1.57, 1.57),
    "ArmUpperL": (-1.57, 1.57)
}

# DDPG hyperparams
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
GAMMA = 0.98
TAU = 0.005
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EXPLORATION_NOISE = 0.1

# ================== ENV SETUP ==================
robot = Supervisor()
motors, sensors = [], []

for name in CONTROL_JOINTS:
    m = robot.getDevice(name)
    s = m.getPositionSensor()
    s.enable(TIMESTEP)
    m.setPosition(0.0)
    motors.append(m)
    sensors.append(s)

episode_step = 0


def get_obs():
    return np.array([s.getValue() for s in sensors], dtype=np.float32)


def apply_action(action):
    """Clamp to joint-specific limits and apply."""
    for i, (m, a) in enumerate(zip(motors, action)):
        jname = CONTROL_JOINTS[i]
        low, high = ANGLE_LIMITS[jname]
        a = float(np.clip(a, low, high))
        m.setPosition(a)

dist_prev = -1
def compute_reward(obs):
    global dist_prev
    reward = -0.1
    
    dist = np.linalg.norm(obs - TARGET)
    if dist_prev != -1:
        if dist_prev < dist:
            reward = -1.0
    dist_prev = dist
    
    done = (obs == TARGET).all() or episode_step >= MAX_STEPS or dist < 0.01
    if (obs == TARGET).all():
        reward = 1.0
    
    return reward, done

def compute_reward_yudhis(obs):
    dist = np.linalg.norm(obs - TARGET)
    reward = -dist**2
    done = dist < 0.01 or episode_step >= MAX_STEPS
    if dist < 0.01:
        reward += 5.0
    return reward, done


def reset_env():
    """Forcefully reset both joints to zero."""
    global episode_step
    episode_step = 0
    for m in motors:
        m.setPosition(0.0)
    # Wait until sensor readings converge near zero
    while True:
        robot.step(TIMESTEP)
        obs = get_obs()
        if np.allclose(obs, [0.0, 0.0], atol=1e-3):
            break
    return get_obs()


# ================== DDPG COMPONENTS ==================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class DDPG:
    def __init__(self, obs_dim, act_dim):
        self.actor = MLP(obs_dim, act_dim)
        self.actor_target = MLP(obs_dim, act_dim)
        self.critic = MLP(obs_dim + act_dim, 1)
        self.critic_target = MLP(obs_dim + act_dim, 1)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.buffer = []
        self.device = torch.device("cpu")

    def store(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > BUFFER_SIZE:
            self.buffer.pop(0)

    def sample(self):
        idx = np.random.choice(len(self.buffer), BATCH_SIZE, replace=False)
        batch = [self.buffer[i] for i in idx]
        obs, act, rew, next_obs, done = map(np.stack, zip(*batch))
        return map(lambda x: torch.tensor(x, dtype=torch.float32, device=self.device),
                   (obs, act, rew, next_obs, done))

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        obs, act, rew, next_obs, done = self.sample()

        # Critic update
        with torch.no_grad():
            next_act = self.actor_target(next_obs)
            target_q = self.critic_target(torch.cat([next_obs, next_act], dim=1))
            y = rew.unsqueeze(1) + GAMMA * (1 - done.unsqueeze(1)) * target_q
        q = self.critic(torch.cat([obs, act], dim=1))
        critic_loss = nn.functional.mse_loss(q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        actor_loss = -self.critic(torch.cat([obs, self.actor(obs)], dim=1)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft target update
        for t, s in zip(self.actor_target.parameters(), self.actor.parameters()):
            t.data.copy_(t.data * (1 - TAU) + s.data * TAU)
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.copy_(t.data * (1 - TAU) + s.data * TAU)


# ================== TRAINING LOOP ==================
agent = DDPG(obs_dim=2, act_dim=2)
start_time = time.time()

for ep in range(1, MAX_EPISODES + 1):
    obs = reset_env()
    total_reward = 0.0
    for episode_step in range(MAX_STEPS):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = agent.actor(obs_t).numpy()
        action += np.random.normal(0, EXPLORATION_NOISE, size=2)
        apply_action(action)
        robot.step(TIMESTEP)
        next_obs = get_obs()
        reward, done = compute_reward(next_obs)
        agent.store((obs, action, reward, next_obs, float(done)))
        agent.update()
        obs = next_obs
        total_reward += reward
        if done:
            break

    elapsed = time.time() - start_time
    avg_time = elapsed / ep
    eta = avg_time * (MAX_EPISODES - ep)
    print(f"Ep {ep}/{MAX_EPISODES} | Steps: {episode_step:3d} | "
          f"Reward: {total_reward:7.3f} | ETA: {eta/60:.1f} min")

print("✅ Training finished.")

# Save the trained model
os.makedirs("checkpoints", exist_ok=True)
checkpoint_path = "checkpoints/ddpg_model1.pt"
torch.save({
    'actor_state_dict': agent.actor.state_dict(),
    'actor_target_state_dict': agent.actor_target.state_dict(),
    'critic_state_dict': agent.critic.state_dict(),
    'critic_target_state_dict': agent.critic_target.state_dict(),
    'obs_dim': 2,
    'act_dim': 2,
}, checkpoint_path)
print(f"✅ Model saved to {checkpoint_path}")
