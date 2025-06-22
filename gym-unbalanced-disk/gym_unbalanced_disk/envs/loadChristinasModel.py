import time, numpy as np, gymnasium as gym, gym_unbalanced_disk
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit, TransformObservation
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import UnbalancedDiskChristina

DT           = 0.025
UMAX         = 3.0
EP_LEN       = 300
TOTAL_STEPS  = 120_000
EXPERIMENT   = False
N_ENVS       = 8 if not EXPERIMENT else 1
MODEL_FILE   = "models/sac_unbalanced_disk.zip"

def obs_fn(obs):
    th,w = obs
    return np.array([np.sin(th), np.cos(th), w], dtype=np.float32)

obs_space = spaces.Box(low=np.array([-1,-1,-40],dtype=np.float32),
                       high=np.array([ 1, 1, 40],dtype=np.float32))

class DenseReward(gym.Wrapper):
    def step(self, action):
        obs,r,term,trunc,info = self.env.step(action)
        th = self.env.unwrapped.th
        r  = r + 0.2*(1-np.cos(th)) - 0.001*float(action)**2
        return obs,r,term,trunc,info

def make_env(experiment=False, render_mode=None):
    if experiment:
        base = gym_unbalanced_disk.UnbalancedDisk_exp(umax=UMAX, dt=DT)
    else:
        base = UnbalancedDiskChristina.UnbalancedDisk(umax=UMAX, dt=DT, render_mode=render_mode)
        
        base = DenseReward(base) 
    env = TimeLimit(base, max_episode_steps=EP_LEN)
    env = TransformObservation(env, obs_fn, observation_space=obs_space)
    return env

class Printer(BaseCallback):
    def __init__(self, freq=500):
        super().__init__()
        self.f = freq
        self.t = 0
        self.buf = []

    def _on_step(self):
        self.t += 1
        for info in self.locals["infos"]:
            if "episode" in info:
                self.buf.append(info["episode"]["r"] / info["episode"]["l"])
                self.buf = self.buf[-50:]
        if self.t % self.f == 0 and self.buf:
            print(f"[{self.t:6}] mean50={np.mean(self.buf):.3f}")
        return True

vec_env = make_vec_env(lambda: make_env(EXPERIMENT), n_envs=N_ENVS)
vec_env = VecMonitor(vec_env)

model = SAC.load("sac model test", env=vec_env)
demo_env = make_env(experiment=False, render_mode="human")
obs, _ = demo_env.reset()


import time

for _ in range(300):  # or however long you want to run
    print("hi")
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = demo_env.step(action)
    demo_env.render()
    time.sleep(1/24)  # Control rendering speed
    if terminated or truncated:
        obs, _ = demo_env.reset()
demo_env.close()