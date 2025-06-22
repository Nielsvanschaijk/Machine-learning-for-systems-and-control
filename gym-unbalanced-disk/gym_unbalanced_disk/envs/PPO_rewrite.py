import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

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
        self.action_space = gym.spaces.Box(low=-3, high = 3, shape = (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.pi, -8], dtype=np.float32),
            high=np.array([np.pi, 8], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

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
        self.reward_fun = lambda self: (
            10 * np.cos(self.th - np.pi)
            - 0.1 * self.omega**2
            - 0.001 * self.u**2
        )

        # ## Punish for overshooting
        # self.reward_fun = lambda self: (
        #     -(self.th**2 + 0.1 * self.omega**2 + 0.001 * self.u**2)

        # )
        #     # Big reward for being upright
        #     100 * np.cos(self.th - np.pi)

        #     # Reward for being upright for a long time
        #     + 500 * abs(np.cos(self.th - np.pi)) * (self.dt / 0.025) if abs(self.th) > 3 else 0 # dt is the time step
            
        #     # Reward for swing amplitude: high when |th| is large (upside)
        #     + 100 * abs(np.sin(self.th / 2))  # peaks at th=±π
            
        #     # Reward fast motion near bottom to encourage energy build-up
        #     + 0.5 * (1 - np.cos(self.th)) * abs(self.omega) if abs(self.th) < np.pi/2 else 0
            
        #     # Penalize control effort
        #     - 0.001 * self.u**2

        #     # Penalize no swing angle at the bottom
        #     - 0.1 * abs(self.delta_th) if abs(self.delta_th) < np.pi/2 else 0

        #     # Pelanize large swing angle at the top
        #     - 500 * abs(self.omega**4) if abs(self.th) >= 2.5 else 0

        #     # Penalize large swing angle at the top
        #     + 1/self.omega**2 if self.th >= 3 else 0
        # )
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

        fname = os.path.join(os.path.dirname(__file__), "clockwise.png")
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

# Simple RK4 integrator
def rk4_step(f, y, t, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt * k1 / 2)
    k3 = f(t + dt/2, y + dt * k2 / 2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

# === Define PPO Actor-Critic Networks ===

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.alpha_head = nn.Linear(64, act_dim)
        self.beta_head = nn.Linear(64, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        alpha = F.softplus(self.alpha_head(x)) + 1.0
        beta = F.softplus(self.beta_head(x)) + 1.0
        return alpha, beta

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.v_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.v_head(x)
        return v

# === Utility: Sample actions from actor ===
def get_action(actor, obs):
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    alpha, beta = actor(obs_tensor)
    dist = torch.distributions.Beta(alpha, beta)
    sample = dist.sample()

    # Map Beta(0,1) → [-3, +3]
    action = 6.0 * sample - 3.0

    log_prob = dist.log_prob(sample).sum(axis=-1)

    return action.squeeze().detach().numpy(), log_prob.item()


# === Compute GAE returns ===
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    returns = []
    gae = 0
    next_value = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        next_value = values[step]
        returns.insert(0, gae + values[step])
    return returns

# === PPO Main Training Loop ===
def train_ppo(max_timesteps, batch_size, ppo_epochs, clip_ratio):

    env = gym.wrappers.TimeLimit(UnbalancedDisk(), max_episode_steps=350)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else 1

    actor = Actor(obs_dim, act_dim)
    critic = Critic(obs_dim)

    actor_optim = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optim = optim.Adam(critic.parameters(), lr=1e-3)

    max_timesteps = 200_000
    batch_size = 2048
    ppo_epochs = 10
    clip_ratio = 0.2

    obs_buf = []
    act_buf = []
    logp_buf = []
    rew_buf = []
    done_buf = []
    val_buf = []

    timesteps = 0
    episode_rewards = []

    while timesteps < max_timesteps:

        obs, info = env.reset()
        done = False
        ep_reward = 0

        while not done and timesteps < max_timesteps:
            action, logp = get_action(actor, obs)
            action_np = np.array(action).reshape(-1)  # ensure 1D array

            value = critic(torch.FloatTensor(obs).unsqueeze(0)).item()

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done_flag = terminated or truncated

            # Store experience
            obs_buf.append(obs)
            act_buf.append(action_np)
            logp_buf.append(logp)
            rew_buf.append(reward)
            done_buf.append(done_flag)
            val_buf.append(value)

            obs = next_obs
            ep_reward += reward
            timesteps += 1

            if len(obs_buf) >= batch_size:
                # === Compute targets ===
                values = val_buf + [critic(torch.FloatTensor(obs).unsqueeze(0)).item()]
                returns = compute_gae(rew_buf, values[:-1], done_buf)
                returns = torch.FloatTensor(returns)
                obs_tensor = torch.FloatTensor(obs_buf)
                act_tensor = torch.FloatTensor(act_buf)
                old_logp_tensor = torch.FloatTensor(logp_buf)
                val_tensor = torch.FloatTensor(val_buf)

                # === PPO update ===
                for _ in range(ppo_epochs):
                    mu, std = actor(obs_tensor)
                    dist = torch.distributions.Normal(mu, std)
                    new_logp = dist.log_prob(act_tensor).sum(axis=-1)
                    ratio = torch.exp(new_logp - old_logp_tensor)

                    advantage = returns - val_tensor
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                    # Policy loss
                    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
                    loss_pi = -(torch.min(ratio * advantage, clip_adv)).mean()

                    # Value loss
                    v_pred = critic(obs_tensor).squeeze()
                    loss_v = F.mse_loss(v_pred, returns)

                    # Update actor
                    actor_optim.zero_grad()
                    loss_pi.backward()
                    actor_optim.step()

                    # Update critic
                    critic_optim.zero_grad()
                    loss_v.backward()
                    critic_optim.step()

                # Clear buffer
                obs_buf = []
                act_buf = []
                logp_buf = []
                rew_buf = []
                done_buf = []
                val_buf = []

        episode_rewards.append(ep_reward)

        # Log progress every 10 episodes
        if len(episode_rewards) % 10 == 0:
            print(f"Timesteps: {timesteps}, Episode: {len(episode_rewards)}, AvgReward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    return actor, episode_rewards

# === Evaluation function ===
import time

def evaluate_policy(env, actor_path, render=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else 1

    # Create Actor and load trained weights
    actor_eval = Actor(obs_dim, act_dim).to(device)
    actor_eval.load_state_dict(torch.load(actor_path))
    actor_eval.eval()

    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, std = actor_eval(obs_tensor)
            dist = torch.distributions.Normal(mu, std)
            action = dist.mean  # use deterministic action for evaluation
            action_np = action.cpu().numpy().squeeze()

        obs, reward, terminated, truncated, info = env.step(np.array(action_np).reshape(-1))
        total_reward += reward

        if render:
            env.render()
            time.sleep(0.025)  # slow down rendering

        done = terminated or truncated

    print(f"Evaluation Episode Return: {total_reward:.2f}")

if __name__ == "__main__":
    max_timesteps = 200_000
    batch_size = 2048
    ppo_epochs = 10
    clip_ratio = 0.2
    # === Run training ===
    actor, episode_rewards = train_ppo(max_timesteps, batch_size, ppo_epochs, clip_ratio)

    # Save actor
    torch.save(actor.state_dict(), "ppo_actor.pth")
    print("Training finished and model saved.")

    # Plot reward curve
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO on UnbalancedDisk")
    plt.show()

    # === Run evaluation ===
    env_eval = UnbalancedDisk(dt=0.025)
    env_eval = gym.wrappers.TimeLimit(env_eval, max_episode_steps=350)

    evaluate_policy(env_eval, actor_path="ppo_actor.pth", render=True)


