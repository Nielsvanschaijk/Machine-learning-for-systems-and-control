import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np
from collections import defaultdict
from scipy.integrate import solve_ivp
import time
from matplotlib import pyplot as plt
from os import path

class Discretize_obs(gym.Wrapper):
    def __init__(self, env, nvec=10):
        super(Discretize_obs, self).__init__(env) #sets self.env
        if isinstance(nvec,int): #nvec in each dimention
            self.nvec = [nvec]*np.prod(env.observation_space.shape,dtype=int)
        else:
            self.nvec = nvec
        self.nvec = np.array(nvec) #(Nobs,) array
        
        self.observation_space = gym.spaces.MultiDiscrete(self.nvec)#([self.nvec, self.nvec]) #b)
        self.olow, self.ohigh = np.array([-np.pi,-40]), np.array([np.pi,40])

    def discretize(self,observation):
        return tuple(((observation - self.olow)/(self.ohigh - self.olow)*self.nvec).astype(int)) #b)
        
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action) #b)
        return self.discretize(observation), reward, terminated, truncated, info #b)

    def reset(self):
        obs, info = self.env.reset()
        obs_dis = self.discretize(obs)  #b=)
        return obs_dis, info


class UnbalancedDisk(gym.Env):
    def __init__(self, nvec=40, umax=3., dt=0.025, render_mode='human'):
        # Initialize environment parameters
        self.omega0 = 11.339846957335382
        self.delta_th = 0
        self.gamma = 1.3328339309394384
        self.Ku = 28.136158407237073
        self.Fc = 6.062729509386865
        self.coulomb_omega = 0.001
        self.th_ref = np.pi
        self.umax = umax
        self.dt = dt
        # self.action_space = gym.spaces.Discrete(7)
        self.action_space = gym.spaces.Box(low=-3, high = 3)
        self.observation_space = gym.spaces.Box(low=np.array([-np.pi, -40], dtype=np.float32),
                                                high=np.array([np.pi, 40], dtype=np.float32), shape=(2,))
        l = np.array([-float('inf')]).astype(np.float32)
        # self.P = P
        self.th = 0
        self.omega = 0
        self.err = lambda self: abs(((self.th - np.pi + np.pi) % (2 * np.pi)) - np.pi)

        '''
        UnbalancedDisk
        th =            
                    +-pi
                        |
            pi/2   ----- -pi/2
                        |
                        0  = starting location
        '''
        ## Punish for overshooting
        self.reward_fun = lambda self: (
            # Big reward for being upright
            100 * np.cos(self.th - np.pi)

            # Reward for being upright for a long time
            + 5000 * abs(np.cos(self.th - np.pi)) * (self.dt / 0.025) if abs(self.th) > 3 else 0 # dt is the time step
            
            # Reward for swing amplitude: high when |th| is large (upside)
            + 100 * abs(np.sin(self.th / 2))  # peaks at th=±π
            
            # Reward fast motion near bottom to encourage energy build-up
            + 0.5 * (1 - np.cos(self.th)) * abs(self.omega) if abs(self.th) < np.pi/2 else 0
            
            # Penalize control effort
            - 0.001 * self.u**2

            # Penalize no swing angle at the bottom
            - 0.1 * abs(self.delta_th) if abs(self.delta_th) < np.pi/2 else 0

            # Pelanize large swing angle at the top
            - 1000 * abs(self.omega**4) if abs(self.th) >= 2.5 else 0

            # Penalize large swing angle at the top
            + 1/self.omega**2 if self.th >= 3 else 0
        )
        # - (self.th**2 + 0.1 * self.omega**2 + 0.001 * self.u**2))

        self.x = np.array([self.th, self.omega])
        self.r_matrix = np.array([[5, 0], [0, 0.1]])
        self.P = self.reward_fun
            # Big reward for being upright
            
        self.render_mode = render_mode
        self.viewer = None  # Initialize the viewer here
        self.reset()

    def step(self, action):
        self.u_last = self.u

        # action is continuous now
        self.u = np.clip(action.item(), -self.umax, self.umax)

        def f(t, y):
            th, omega = y
            dthdt = omega
            friction = self.gamma * omega + self.Fc * np.tanh(omega / self.coulomb_omega)
            domegadt = -self.omega0 ** 2 * np.sin(th + self.delta_th) - friction + self.Ku * self.u
            return np.array([dthdt, domegadt])

        y = np.array([self.th, self.omega])
        y_next = rk4_step(f, y, 0, self.dt)
        th, self.omega = y_next
        self.delta_th = np.arctan2(np.sin(th - self.th), np.cos(th - self.th))
        self.th = th
        self.costh = -np.cos(th)
        self.x = np.array([self.th, self.omega])

        reward = self.reward_fun(self)
        terminated = np.abs(self.th) >= 3.1415 and self.omega == 0.01
        if terminated:
            reward = 999999999
        return self.get_obs(), reward, terminated, False, {
            "th": self.th,
            "omega": self.omega,
            "delta_th": self.delta_th
        }


    def reset(self, seed=None, options=None):
        np.random.seed(42)
        self.th = np.random.normal(loc=0, scale=0.001)
        self.omega = np.random.normal(loc=0, scale=0.001)
        self.x = np.array([self.th, self.omega])
        self.u = 0
        self.delta_th = 0
        return self.get_obs(), {}

    def get_obs(self):
        self.th_noise = self.th + np.random.normal(loc=0, scale=0.001)
        self.omega_noise = self.omega + np.random.normal(loc=0, scale=0.001)
        return np.array([self.th_noise, self.omega_noise])

    def render(self):
        import pygame
        from pygame import gfxdraw

        screen_width = 500
        screen_height = 500

        th = self.th
        omega = self.omega

        # Initialize the viewer if it's None
        if self.viewer is None:
            pygame.init()
            pygame.display.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        gfxdraw.filled_circle(self.surf, screen_width // 2, screen_height // 2,
                            int(screen_width / 2 * 0.65 * 1.3), (32, 60, 92))
        gfxdraw.filled_circle(self.surf, screen_width // 2, screen_height // 2,
                            int(screen_width / 2 * 0.06 * 1.3), (132, 132, 126))

        r = screen_width // 2 * 0.40 * 1.3
        gfxdraw.filled_circle(self.surf, int(screen_width // 2 - np.sin(th) * r),
                            int(screen_height // 2 - np.cos(th) * r), int(screen_width / 2 * 0.22 * 1.3),
                            (155, 140, 108))
        gfxdraw.filled_circle(self.surf, int(screen_width // 2 - np.sin(th) * r),
                            int(screen_height // 2 - np.cos(th) * r), int(screen_width / 2 * 0.22 / 8 * 1.3),
                            (71, 63, 48))

        fname = path.join(path.dirname(__file__), "clockwise.png")
        self.arrow = pygame.image.load(fname)
        if self.u:
            if isinstance(self.u, (np.ndarray, list)):
                if self.u.ndim == 1:
                    u = self.u[0]
                elif self.u.ndim == 0:
                    u = self.u
                else:
                    raise ValueError(f'u={u} is not the correct shape')
            else:
                u = self.u
            arrow_size = abs(float(u) / self.umax * screen_height) * 0.25
            Z = (arrow_size, arrow_size)
            arrow_rot = pygame.transform.scale(self.arrow, Z)
            if self.u < 0:
                arrow_rot = pygame.transform.flip(arrow_rot, True, False)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.viewer.blit(self.surf, (0, 0))
        if self.u:
            self.viewer.blit(arrow_rot, (screen_width // 2 - arrow_size // 2, screen_height // 2 - arrow_size // 2))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()

        return True

    def close(self):
        if self.viewer is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            self.viewer = None

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from tqdm import tqdm

# PPO Hyperparameters
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
POLICY_CLIP = 0.2
EPOCHS = 3
BATCH_SIZE = 8

# PPO Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Sigmoid(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.BatchNorm1d(64),
            nn.Linear(64, act_dim)
        )
        # Log std parameter (learned)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Sigmoid(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        # Not used — separate actor/critic
        raise NotImplementedError

    def get_action(self, obs):
        mu = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob

    def evaluate_actions(self, obs, actions):
        mu = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value

# PPO Training Loop
def train_ppo(envs, total_steps=350):
    returns_history = []
    theta_history = []
    omega_history = []
    reward_history = []
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ac = ActorCritic(obs_dim, act_dim).to(device)
    optimizer_actor = optim.Adam(ac.actor.parameters(), lr=LR_ACTOR)
    optimizer_critic = optim.Adam(ac.critic.parameters(), lr=LR_CRITIC)

    ROLLOUT_STEPS = 4096
    num_envs = envs.num_envs
    steps_per_env = ROLLOUT_STEPS // num_envs

    obs, _ = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    pbar = tqdm(range(total_steps))
    for step in pbar:
        obs_batch = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
        # pbar.set_description(f"Return: {returns.mean().item():.2f}")
        # returns_history.append(returns.mean().item())
        # theta_history.append(envs.envs[0].th)
        # omega_history.append(envs.envs[0].omega)
        
        for _ in range(steps_per_env):
            obs_batch.append(obs.cpu().numpy())

            with torch.no_grad():
                action, log_prob = ac.get_action(obs)
                value = ac.critic(obs).squeeze(-1)

            actions.append(action.cpu().numpy())
            log_probs.append(log_prob.cpu().numpy())
            values.append(value.cpu().numpy())

            action_env = action.cpu().numpy()
            next_obs, reward, terminated, truncated, _ = envs.step(action_env)

            rewards.append(reward)
            dones.append(np.logical_or(terminated, truncated))

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        # Convert rollout to tensors
        obs_batch = torch.tensor(obs_batch, dtype=torch.float32, device=device)  # (steps, envs, obs_dim)
        actions = torch.tensor(actions, dtype=torch.float32, device=device)      # (steps, envs, act_dim)
        log_probs = torch.tensor(log_probs, dtype=torch.float32, device=device)  # (steps, envs)
        values = torch.tensor(values, dtype=torch.float32, device=device)        # (steps, envs)
        rewards = np.array(rewards)                                              # (steps, envs)
        dones = np.array(dones).astype(np.float32)                               # (steps, envs)

        # Compute returns and advantages
        with torch.no_grad():
            next_value = ac.critic(obs).squeeze(-1).cpu().numpy()

        returns = np.zeros_like(rewards)
        advs = np.zeros_like(rewards)
        gae = np.zeros(envs.num_envs)

        for t in reversed(range(steps_per_env)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + GAMMA * next_value * mask - values[t].cpu().numpy()
            gae = delta + GAMMA * GAE_LAMBDA * mask * gae
            advs[t] = gae
            returns[t] = advs[t] + values[t].cpu().numpy()
            next_value = values[t].cpu().numpy()

        # Flatten (steps * envs, ...)
        obs_batch = obs_batch.reshape(-1, obs_dim)
        actions = actions.reshape(-1, act_dim)
        log_probs = log_probs.reshape(-1)
        returns = torch.tensor(returns.reshape(-1), dtype=torch.float32, device=device)
        advs = torch.tensor(advs.reshape(-1), dtype=torch.float32, device=device)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # PPO update
        for _ in range(EPOCHS):
            indices = np.arange(len(returns))
            np.random.shuffle(indices)

            for start in range(0, len(returns), BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_idx = indices[start:end]

                mb_obs = obs_batch[mb_idx]
                mb_actions = actions[mb_idx]
                mb_log_probs = log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advs = advs[mb_idx]

                new_log_probs, entropy, values_pred = ac.evaluate_actions(mb_obs, mb_actions)
                ratio = (new_log_probs - mb_log_probs).exp()

                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1 - POLICY_CLIP, 1 + POLICY_CLIP) * mb_advs
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(values_pred, mb_returns)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                loss.backward()
                optimizer_actor.step()
                optimizer_critic.step()

        pbar.set_description(f"Return: {returns.mean().item():.2f}")

    torch.save(ac.state_dict(), "ppo_actor_critic.pth")
    # plt.figure(figsize=(8,5))
    # plt.plot(returns_history)
    # plt.xlabel("Training Iteration")
    # plt.ylabel("Average Return")
    # plt.title("Learning Curve")
    # plt.grid()
    # plt.show()

    # plt.figure(figsize=(8,5))
    # plt.hist(theta_history, bins=50)
    # plt.xlabel("Theta (rad)")
    # plt.ylabel("Count")
    # plt.title("Theta Distribution During Training")
    # plt.grid()
    # plt.show()

    # plt.figure(figsize=(8,5))
    # plt.hist(omega_history, bins=50)
    # plt.xlabel("Omega (rad/s)")
    # plt.ylabel("Count")
    # plt.title("Omega Distribution During Training")
    # plt.grid()
    # plt.show()

    plt.figure(figsize=(8,5))
    plt.hist(reward, bins=50)
    plt.xlabel("reward")
    plt.ylabel("Count")
    plt.title("reward over time")
    plt.grid()
    plt.show()

def rk4_step(f, y, t, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

# Evaluation
def evaluate_policy(env, render=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    ac = ActorCritic(obs_dim, act_dim).to(device)
    ac.load_state_dict(torch.load("ppo_actor_critic.pth"))
    ac.eval()

    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    total_reward = 0
    while True:
        with torch.no_grad():
            action, _ = ac.get_action(obs.unsqueeze(0))
        action_env = action.cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action_env)
        total_reward += reward
        time.sleep(0.025)  # Slow down rendering
        if render:
            env.render()

        if terminated or truncated:
            break
        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

    print(f"Episode return: {total_reward:.2f}")

def plot_reward_heatmap():
    # Create a new instance of UnbalancedDisk
    env = UnbalancedDisk(dt=0.025)

    # Create a grid of theta and omega values
    theta_values = np.linspace(-np.pi, np.pi, 100)
    omega_values = np.linspace(-40, 40, 100)

    # Create a 2D grid for theta and omega
    theta_grid, omega_grid = np.meshgrid(theta_values, omega_values)

    # Compute the reward for each (theta, omega) pair
    reward_grid = np.zeros_like(theta_grid)
    for i in range(theta_grid.shape[0]):
        for j in range(theta_grid.shape[1]):
            # Call the reward_function method of UnbalancedDisk instance
            reward_grid[i, j] = env.reward_function(theta_grid[i, j], omega_grid[i, j])

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.contourf(theta_grid, omega_grid, reward_grid, 50, cmap='viridis')
    plt.colorbar(label='Reward')
    plt.xlabel('Theta (rad)')
    plt.ylabel('Omega (rad/s)')
    plt.title('Reward Function Heatmap')
    plt.grid(True)
    plt.show()

def make_env():
    def _thunk():
        env = UnbalancedDisk(dt=0.025)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=350)
        return env
    return _thunk

if __name__ == "__main__":
    num_envs = 64
    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])

    # Train PPO agent
    train_ppo(envs, total_steps=1000)

    # Plot reward heatmap
    #plot_reward_heatmap()

    # Evaluate trained agent
    env_eval = UnbalancedDisk(dt=0.025)
    env_eval = gym.wrappers.TimeLimit(env_eval, max_episode_steps=350)
    evaluate_policy(env_eval, render=True)