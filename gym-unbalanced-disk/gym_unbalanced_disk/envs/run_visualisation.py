# run_visualisation_tracking.py
import time
import gymnasium as gym
import gym_unbalanced_disk
import UnbalancedDiskChristina
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
import numpy as np
import matplotlib.pyplot as plt
import UnbalancedDisk

DT = 0.025
UMAX = 3.0
EP_LEN = 300
MODEL_FILE = "models/sac_unbalanced_disk.zip"
USE_TRAJECTORY = True

class TrackingReward(gym.Wrapper):
    def __init__(self, env, use_trajectory=False):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1, -1, -40], dtype=np.float32),
            high=np.array([1, 1, 40], dtype=np.float32)
        )
        self.theta_target = 0.0
        self.use_trajectory = use_trajectory
        self.time_step = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.time_step = 0
        if not self.use_trajectory:
            self.theta_target = np.random.uniform(-np.pi/12, np.pi/12)
        return self.obs_fn(obs, step_count=self.time_step), info

    def _obs(self, obs):
        return self.obs_fn(obs, step_count=self.time_step)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        th, w = obs
        
        warmup_time = int(5 / DT)
        if self.use_trajectory:
            if self.time_step < warmup_time:
                self.theta_target = 0.0
            else:
                T = int(5 / DT)
                phase_t = self.time_step - warmup_time
                self.theta_target = (np.pi/12) * np.sin(2 * np.pi * phase_t / T)
        # info["theta_target"] = self.theta_target
        self.time_step += 1
        
        angle_error = th - self.theta_target
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        
        r = 5.0 * np.cos(angle_error)
        r += -0.1 * w**2
        r += -0.001 * float(action)**2

        obs = self._obs(obs)
        info["target_goal"] = self.target_goal
        return obs, r, term, trunc, info

    def obs_fn(self, obs, step_count=0, total_steps=1000):
        th, w = obs

        AMPLITUDE = np.pi / 12
        FREQ = 0.2

        if step_count < 50:
            target_shift = 0.0
        else:
            t_sec = (step_count - 50) * DT
            target_shift = AMPLITUDE * np.sin(2 * np.pi * FREQ * t_sec)
        self.target_goal = -np.pi -target_shift
        return np.array([
            np.sin(th + target_shift),
            np.cos(th + target_shift),
            w
        ], dtype=np.float32)

def make_env(render_mode=None, use_trajectory=False):
    base = UnbalancedDisk.UnbalancedDisk(umax=UMAX, dt=DT, render_mode=render_mode)
    base = TrackingReward(base, use_trajectory=use_trajectory)
    env = TimeLimit(base, max_episode_steps=EP_LEN)
    return env

# Load model
model = SAC.load(MODEL_FILE)
print(f"Loaded model from {MODEL_FILE}")

# Visualize with tracking
demo = make_env(render_mode="human", use_trajectory=USE_TRAJECTORY)
# obs, _ = demo.reset()

# target_goals = []
# ths = []
# for _ in range(500):
#     a, _ = model.predict(obs, deterministic=True)
#     obs, _, t, tc, info = demo.step(a)
#     target_goals.append(info["target_goal"])
#     ths.append(info["th"])
#     demo.render()
#     time.sleep(1/24)
#     # if t or tc:
#         # obs, _ = demo.reset()
# rmse = np.sqrt(np.mean((np.array(target_goals[50:]) - np.array(ths[50:])) ** 2))
# print("rmse ", rmse)
# demo.close()
timepoints = np.arange(500)
# fig, ax = plt.subplots()
# ax.plot(timepoints, target_goals, label='target angle')
# ax.plot(timepoints, ths, label = 'system angle')


# ymin = min(ths)
# ymax = max(ths)
# # Compute lower and upper bounds for odd multiples of π
# lower_top = int(np.ceil((ymin - np.pi) / (2 * np.pi)))
# upper_top = int(np.floor((ymax - np.pi) / (2 * np.pi)))

# # Generate odd multiples of π within the range
# top_positions = (2 * np.arange(lower_top, upper_top + 1) + 1) * np.pi

# bottom_multiples = np.arange(np.floor(ymin / (2 * np.pi)), np.ceil(ymax / (2*np.pi)) + 1)
# bottom_positions = bottom_multiples[1:-1] * 2*np.pi

# for pos in top_positions:
#     ax.axhline(y=pos, color='red', linestyle='--', linewidth=0.8)
#     ax.text(0, pos, 'top', color='red', fontsize=9, va='bottom', ha='left')

# for pos in bottom_positions:
#     ax.axhline(y=pos, color='blue', linestyle='--', linewidth=0.8)
#     ax.text(0, pos, 'bottom', color='blue', fontsize=9, va='bottom', ha='left')

# ax.set_xlim(0, 500)
# plt.ylabel("Angle (rad)")
# plt.xlabel("Time step")
# plt.legend()
# plt.title('angle $\\theta$ of the pendulum and the target angle per time step')
# plt.show()

import seaborn as sns
# plt.figure()
# sns.lineplot(timepoints, ths, ci=0.95)
# plt.show()
all_ci_ths = []
for i in range(30):
    obs, _ = demo.reset()
    ci_ths = []
    for _ in range(500):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, t, tc, info = demo.step(a)
        ci_ths.append(info["th"])
    all_ci_ths.append(ci_ths)
    print(i)
    print(info['th'])
import pandas as pd
all_ci_ths = np.array(all_ci_ths)
df = pd.DataFrame(all_ci_ths.T, index=timepoints)  # shape (time_steps, n_trajectories)
df = df.reset_index().melt(id_vars='index', var_name='trajectory', value_name='angle')
df.rename(columns={'index': 'time'}, inplace=True)
print(df.head())
print(df.groupby('time').size().min())

# Plot with seaborn: automatically calculates mean and 95% CI
sns.lineplot(data=df, x='time', y='angle', errorbar=('ci', 95))
plt.xlabel('Time step')
plt.ylabel('Angle θ (rad)')
plt.title('Mean angle of 30 iterations over time with 95% CI')
plt.show()

# Save trajectory + observation components
# save_env = make_env(use_trajectory=USE_TRAJECTORY)
# obs, _ = save_env.reset()
# th, th_target, u = [], [], []
# obs_sin, obs_cos, obs_w = [], [], []

# for _ in range(1000):
#     a, _ = model.predict(obs, deterministic=True)
#     obs, _, t, tc, _ = save_env.step(a)
    
#     th.append(save_env.unwrapped.th)
#     th_target.append(save_env.env.theta_target)
#     u.append(float(a))
    
#     obs_sin.append(obs[0])
#     obs_cos.append(obs[1])
#     obs_w.append(obs[2])
    
#     if t or tc:
#         obs, _ = save_env.reset()

# np.savez("models/sac_tracking_eval.npz", 
#          th=np.array(th), 
#          th_target=np.array(th_target), 
#          u=np.array(u),
#          obs_sin=np.array(obs_sin),
#          obs_cos=np.array(obs_cos),
#          obs_w=np.array(obs_w))

# print("Saved → models/sac_tracking_eval.npz")

# Load saved data
# data = np.load("models/sac_tracking_eval.npz")
# th = data["th"]  # Actual disk angle
# th_target = data["th_target"]  # Desired target angle
# u = data["u"]  # Action (torque)
# obs_sin = data["obs_sin"]
# obs_cos = data["obs_cos"]
# obs_w = data["obs_w"]

# # Time vector
# t = np.arange(len(th)) * DT

# # Compute θ_obs: the angle SAC "sees"
# theta_obs = np.arctan2(obs_sin, obs_cos)

# # Plot: Observed angle vs Actual angle vs Target
# plt.figure(figsize=(12, 6))
# plt.plot(t, theta_obs, label="Observed angle θ_obs (from obs)", linestyle=':')
# plt.plot(t, th, label="Actual angle θ (disk)", linestyle='-')
# plt.plot(t, th_target, label="Target angle θ_target", linestyle='--')
# plt.xlabel("Time (s)")
# plt.ylabel("Angle (rad)")
# plt.title("Observed angle vs Actual angle vs Target")
# plt.legend()
# plt.grid()

# # Plot control input
# plt.figure(figsize=(10, 3))
# plt.plot(t, u, label="Control input u (Nm)")
# plt.xlabel("Time (s)")
# plt.ylabel("Control input (Nm)")
# plt.title("Control Signal")
# plt.grid()

# plt.show()
