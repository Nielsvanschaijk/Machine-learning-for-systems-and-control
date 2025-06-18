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
    def __init__(self, env, hidden_size=256):
        super(ActorCritic, self).__init__()
        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # Define your layers here:
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)
        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
    
    def actor(self, state, return_logp=False):
        hidden = torch.relu(self.actor_linear1(state))
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
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=np.array([-np.pi, -40], dtype=np.float32),
                                                 high=np.array([np.pi, 40], dtype=np.float32), shape=(2,))

        self.reward_fun = lambda self: (
            # Position reward
            1 * abs(self.costh) 

            # Swing motion at the bottom
            #100 * abs(self.omega) #if abs(self.th) <= 0.1415 else 0

            # Control penalty
            #-0.01 * self.u 

            # Position at the top half reward
            # 10 * abs(np.cos(np.pi - abs(self.th)))
            # Overshoot penalty near upright
            #(-5.0 if abs(self.delta_th) > 0.5 and abs(np.arctan2(np.sin(self.th - np.pi), np.cos(self.th - np.pi))) < 0.2 else 0)
        )



        self.render_mode = render_mode
        self.viewer = None  # Initialize the viewer here
        self.reset()

    def step(self, action):
        self.u = [-3, -1, 0, 1, 3][action]
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


def train_actor_critic(env, actor_crit, n_episodes=100, gamma=0.95, tau=1e-3,
                       initial_epsilon=0.99, final_epsilon=0.05, epsilon_decay=0.99,
                       n_action_repeat=8):
    optimizer = optim.Adam(actor_crit.parameters(), lr=3e-4)
    epsilon = initial_epsilon

    reward_history = []
    actor_losses = []
    critic_losses = []

    for episode in range(n_episodes):
        # Run rollout
        states, actions, rewards, next_states, dones = rollout(
            actor_crit, env, N_rollout=1, epsilon=epsilon, n_action_repeat=n_action_repeat)

        # Episode total reward
        reward_history.append(np.sum(rewards))

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        # Target and value estimates
        next_values = actor_crit.critic(next_states)
        target_values = rewards + gamma * next_values * (~dones).float()

        values = actor_crit.critic(states)
        critic_loss = nn.MSELoss()(values, target_values.detach())

        log_probs = actor_crit.actor(states, return_logp=True)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1))
        advantage = target_values.detach() - values
        actor_loss = -(selected_log_probs * advantage.detach()).mean()

        total_loss = actor_loss + critic_loss

        # Update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Decay epsilon
        epsilon = max(final_epsilon, epsilon * epsilon_decay)

        # Store losses
        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())

    return reward_history, actor_losses, critic_losses



def rollout(actor_crit, env, N_rollout=250, epsilon=0.99, n_action_repeat=8):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    obs, info = env.reset()

    action = None  # current action being repeated
    repeat_counter = 0  # how many times the action has been repeated

    for step in range(N_rollout):
        # Choose new action if repeat count is over or if there's no action yet
        if repeat_counter == 0 or action is None:
            probs = actor_crit.actor(torch.tensor(obs, dtype=torch.float32)[None, :])[0].detach().numpy()
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax(probs)
            repeat_counter = n_action_repeat  # reset repeat counter

        # Perform step with the current repeated action
        obs_next, reward, terminated, truncated, info = env.step(action)

        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        next_states.append(obs_next)
        dones.append(terminated or truncated)

        # Decrement repeat counter
        repeat_counter -= 1

        # Handle reset if episode ends
        if terminated or truncated:
            obs, info = env.reset()
            action = None
            repeat_counter = 0
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
    reward_history, actor_losses, critic_losses = train_actor_critic(env, actor_crit)
    # Plot total rewards
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(reward_history)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    # Plot actor loss
    plt.subplot(1, 3, 2)
    plt.plot(actor_losses)
    plt.title("Actor Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)

    # Plot critic loss
    plt.subplot(1, 3, 3)
    plt.plot(critic_losses)
    plt.title("Critic Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Show the trained policy
    show(actor_crit, env)
