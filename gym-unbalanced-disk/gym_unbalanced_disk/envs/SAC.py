import time, numpy as np, gymnasium as gym, gym_unbalanced_disk
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit, TransformObservation
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

DT           = 0.025
UMAX         = 3.0
EP_LEN       = 300
TOTAL_STEPS  = 120_000
EXPERIMENT   = False
N_ENVS       = 8 if not EXPERIMENT else 1
MODEL_FILE   = "models/sac_unbalanced_disk.zip"

def obs_fn(obs):
    obs = np.array(obs)
    if obs.ndim == 1:
        # Single obs: shape (2,) → (th, w)
        th, w = obs
        return np.array([np.sin(th), np.cos(th), w], dtype=np.float32)
    elif obs.ndim == 2:
        # Batch obs: shape (n_envs, 2)
        th = obs[:, 0]
        w  = obs[:, 1]
        return np.stack([np.sin(th), np.cos(th), w], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unexpected obs shape: {obs.shape}")


obs_space = spaces.Box(low=np.array([-1,-1,-40],dtype=np.float32),
                       high=np.array([ 1, 1, 40],dtype=np.float32))

class DenseReward(gym.Wrapper):
    def step(self, action):
        obs,r,term,trunc,info = self.env.step(action)
        th = self.env.unwrapped.th
        r  = r + 0.2*(1-np.cos(th)) - 0.001*float(action)**2
        return obs,r,term,trunc,info

    def seed(self, seed=None):
        return self.env.seed(seed)


def make_env(experiment=False, render_mode=None):
    if experiment:
        base = gym_unbalanced_disk.UnbalancedDisk_exp(umax=UMAX, dt=DT)
    else:
        base = gym_unbalanced_disk.UnbalancedDisk(umax=UMAX, dt=DT, render_mode=render_mode)
        base = DenseReward(base)
    base = ResetWrapper(base)   # <--- ADD THIS
    env = TimeLimit(base, max_episode_steps=EP_LEN)
    env = TransformObservation(env, obs_fn, observation_space=obs_space)
    return env

class ResetWrapper(gym.Wrapper):
    def reset(self, *, seed=None, options=None):
        try:
            obs, info = self.env.reset(seed=seed, options=options)
        except TypeError:
            # fallback for old environments
            if seed is not None:
                try:
                    self.env.seed(seed)
                except AttributeError:
                    pass
            obs = self.env.unwrapped.reset()   # <--- IMPORTANT
            info = {}
        return obs, info

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

model = SAC("MlpPolicy", vec_env,
            verbose=0, buffer_size=200_000, learning_starts=2_000,
            batch_size=512, gamma=0.99, tau=0.005, ent_coef="auto_0.2")

print("\nTraining…\n")
model.learn(total_timesteps=TOTAL_STEPS, callback=Printer())
model.save(MODEL_FILE)
print("\nSaved model →", MODEL_FILE, "\n")

if not EXPERIMENT:
    demo = make_env(False, render_mode="human")
    obs, _ = demo.reset()
    for _ in range(600):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, t, tc, _ = demo.step(a)
        demo.render(); time.sleep(1/24)
        if t or tc:
            obs, _ = demo.reset()
    demo.close()

save_env = make_env(EXPERIMENT)
obs, _ = save_env.reset(); th, u = [], []
for _ in range(500):
    a, _ = model.predict(obs, deterministic=True)
    obs, _, t, tc, _ = save_env.step(a)
    th.append(save_env.unwrapped.th); u.append(float(a))
    if t or tc:
        obs, _ = save_env.reset()

np.savez("models/sac_submission.npz", th=np.array(th), u=np.array(u))
print("Saved model → models/sac_submission.npz")