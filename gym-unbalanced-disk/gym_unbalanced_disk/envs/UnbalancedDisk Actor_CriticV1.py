import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import defaultdict
from scipy.integrate import solve_ivp
import time
from matplotlib import pyplot as plt


# Define the Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, env, hidden_size=40):
        super(ActorCritic, self).__init__()
        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # Define your layers here:
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)
        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
    
    def actor(self, state, return_logp=False):
        hidden = torch.tanh(self.actor_linear1(state))
        h = self.actor_linear2(hidden)
        
        # Use log_softmax instead of manually computing log probabilities
        logp = torch.nn.functional.log_softmax(h, dim=1)
        
        if return_logp:
            return logp
        else:
            return torch.exp(logp) 

    
    def critic(self, state):
        hidden = torch.tanh(self.critic_linear1(state))
        return self.critic_linear2(hidden)[:, 0]
    
    def forward(self, state):
        return self.critic(state), self.actor(state)


# Define Discretize Observation Wrapper (used in your environment)
class Discretize_obs(gym.Wrapper):
    def __init__(self, env, nvec=10):
        super(Discretize_obs, self).__init__(env)  # sets self.env
        self.nvec = nvec if isinstance(nvec, list) else [nvec] * np.prod(env.observation_space.shape, dtype=int)
        self.nvec = np.array(nvec)  # (Nobs,) array
        self.observation_space = gym.spaces.MultiDiscrete(self.nvec)  # b)
        self.olow, self.ohigh = np.array([-np.pi, -40]), np.array([np.pi, 40])

    def discretize(self, observation):
        return tuple(((observation - self.olow) / (self.ohigh - self.olow) * self.nvec).astype(int))

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.discretize(observation), reward, terminated, truncated, info

    def reset(self):
        obs, info = self.env.reset()
        obs_dis = self.discretize(obs)
        return obs_dis, info


class UnbalancedDisk(gym.Env):
    def __init__(self, nvec=40, umax=3., dt=0.025, render_mode='human'):
        '''
        UnbalancedDisk
        th =            
                    +-pi
                        |
            pi/2   ----- -pi/2
                        |
                        0  = starting location
        '''
        # Initialize environment parameters
        self.omega0 = 11.339846957335382
        self.delta_th = 0
        self.gamma = 1.3328339309394384
        self.Ku = 28.136158407237073
        self.Fc = 6.062729509386865
        self.coulomb_omega = 0.001

        self.umax = umax
        self.dt = dt
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(low=np.array([-np.pi, -40], dtype=np.float32),
                                                 high=np.array([np.pi, 40], dtype=np.float32), shape=(2,))

        self.reward_fun = lambda self: (
            # Big reward for being upright
            1000 * np.cos(self.th - np.pi)

            # Reward for being upright for a long time
            + 50 * np.cos(self.th - np.pi) * (self.dt / 0.025)  # dt is the time step

            # 2. Energy build-up at bottom (encourage fast motion at base)
            + 1 * np.exp(-(self.th**2) / 0.5) * abs(self.omega)

            # 3. Penalize standing still at bottom (inaction)
            - 2 * np.exp(-(self.th**2) / 0.5) * np.exp(-abs(self.omega))

            # 4. Penalize control effort
            - 0.01 * (self.u**2)

            # 5. Penalize high velocity at top
            - 100 * (abs(self.omega)) if abs((self.th - np.pi) % (2 * np.pi) - np.pi) < 0.15 else 0
        )

        self.render_mode = render_mode
        self.viewer = None  # Initialize the viewer here
        self.reset()

    def step(self, action):
        self.u = [-3, -1, -0.5, 0, 0.5, 1, 3][action]
        self.u = np.clip(self.u, -self.umax, self.umax)

        def f(t, y):
            th, omega = y
            dthdt = omega
            friction = self.gamma * omega + self.Fc * np.tanh(omega / self.coulomb_omega)
            domegadt = -self.omega0 ** 2 * np.sin(th + self.delta_th) - friction + self.Ku * self.u
            return np.array([dthdt, domegadt])

        sol = solve_ivp(f, [0, self.dt], [self.th, self.omega])
        th, self.omega = sol.y[:, -1]
        self.delta_th = np.arctan2(np.sin(th - self.th), np.cos(th - self.th))
        self.th = th
        self.costh = -np.cos(th)

        reward = self.reward_fun(self)
        terminated = False
        return self.get_obs(), reward, terminated, False, [self.th, self.omega, self.delta_th]

    def reset(self):
        self.th = np.random.normal(loc=0, scale=0.001)
        self.omega = np.random.normal(loc=0, scale=0.001)
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

        fname = "clockwise.png"
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


def train_actor_critic(env, actor_crit, n_episodes=1000, gamma=0.995, tau=1e-3,
                       initial_epsilon=1.0, final_epsilon=0.05, epsilon_decay=0.995):
    optimizer = optim.Adam(actor_crit.parameters(), lr=1e-3)
    epsilon = initial_epsilon

    for episode in range(n_episodes):
        print(f"Starting Episode {episode + 1}/{n_episodes} (ε={epsilon:.3f})")

        # Pass epsilon to rollout
        states, actions, rewards, next_states, dones = rollout(actor_crit, env, N_rollout=50, epsilon=epsilon)

        # Training steps...
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        next_values = actor_crit.critic(next_states)
        target_values = rewards + gamma * next_values * (1 - dones.float())

        values = actor_crit.critic(states)
        critic_loss = nn.MSELoss()(values, target_values)

        log_probs = actor_crit.actor(states, return_logp=True)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1))
        advantage = target_values - values.detach()
        actor_loss = -(selected_log_probs * advantage.detach()).mean()

        total_loss = critic_loss + actor_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Decay epsilon (clipped to final_epsilon)
        epsilon = max(final_epsilon, epsilon * epsilon_decay)

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Loss: {total_loss.item()}")


def rollout(actor_crit, env, N_rollout=250, epsilon=0.999):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    obs, info = env.reset()

    for _ in range(N_rollout):
        probs = actor_crit.actor(torch.tensor(obs, dtype=torch.float32)[None, :])[0].detach().numpy()

        # Epsilon-greedy action
        if np.random.rand() < epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            action = np.argmax(probs)

        states.append(obs)
        actions.append(action)

        obs_next, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        next_states.append(obs_next)
        dones.append(terminated or truncated)
        epsilon = max(0.05, epsilon * epsilon)

        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs = obs_next

    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)



# Run simulation (visualize the policy)
def show(actor_crit, env):
    pi = lambda x: actor_crit.actor(torch.tensor(x[None, :], dtype=torch.float32))[0].numpy()
    with torch.no_grad():
        try:
            obs, info = env.reset()
            env.render()
            time.sleep(1)
            while True:
                action = np.argmax(pi(obs))
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                if terminated or truncated:
                    time.sleep(0.5)
                    break
        finally:
            env.close()


if __name__ == '__main__':
    env_name = 'UnbalancedDisk'
    env = UnbalancedDisk()
    actor_crit = ActorCritic(env)

    # Train the Actor-Critic model
    train_actor_critic(env, actor_crit)

    # Show the trained policy
    show(actor_crit, env)
