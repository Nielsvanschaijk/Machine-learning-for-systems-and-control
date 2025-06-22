import os
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import defaultdict
from scipy.integrate import solve_ivp
import time
from matplotlib import pyplot as plt
from os import path
from AC_Continue_Learning import UnbalancedDisk

max_episode_steps = 300

env = UnbalancedDisk()
env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

class ActorCritic(nn.Module):
    def __init__(self, env, hidden_size=64):
        super(ActorCritic, self).__init__()
        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.n

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

    def actor(self, state, return_logp=False):
        hidden = torch.tanh(self.actor_linear1(state))
        logits = self.actor_linear2(hidden)
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # stabilize

        logp = logits - torch.log(torch.sum(torch.exp(logits), dim=1, keepdim=True))  # log-softmax
        if return_logp:
            return logp
        else:
            return torch.exp(logp)

    def critic(self, state):
        hidden = torch.tanh(self.critic_linear1(state))
        return self.critic_linear2(hidden).squeeze(-1)

    def forward(self, state):
        return self.critic(state), self.actor(state)

# Initialize the ActorCritic model
model = ActorCritic(env)

# Test the model on a single observation
obs, _ = env.reset()
obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # shape (1, obs_dim)

# Get action probabilities and value estimate
action_probs = model.actor(obs_tensor)[0]  # shape (n_actions,)
value_estimate = model.critic(obs_tensor)[0]  # scalar

# Helper function to use policy externally
pi = lambda x: model.actor(torch.tensor(x[None, :], dtype=torch.float32))[0].detach().numpy()

import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import time

def run_simulation(env, model, max_steps=500, render_delay=1/24):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_probs = model.actor(state_tensor)[0]
        action = torch.argmax(action_probs).item()  # choose the most likely action

        state, reward, done, _, _ = env.step(action)
        total_reward += reward

        env.render()
        time.sleep(render_delay)  # slow down rendering

        if done:
            break

    env.close()
    print(f"Simulation ended. Total reward: {total_reward:.2f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ActorCritic(env).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

def select_action(state):
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    probs = model.actor(state)[0]
    action = torch.multinomial(probs, num_samples=1).item()
    return action, probs[action]

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32, device=device)

skip_simulation = False

# Training Loop
def training_loop(num_episodes = 10000, gamma = 0.95):

    try:
        reward_log = []
        reward_threshold = 25000

        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False

            log_probs = []
            values = []
            rewards = []

            index = 0
            while not done:
                action, log_prob = select_action(state)
                value = model.critic(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))[0]

                next_state, reward, done, _, _ = env.step(action)

                log_probs.append(torch.log(log_prob + 1e-8))  # log prob of taken action
                values.append(value)
                rewards.append(reward)

                state = next_state
                if index > 500:
                    done = True
                index += 1

            # Compute returns and advantages
            returns = compute_returns(rewards, gamma)
            values = torch.stack(values)
            log_probs = torch.stack(log_probs)

            advantage = returns - values

            # Losses
            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = F.mse_loss(values, returns)
            loss = actor_loss + critic_loss

            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_reward = sum(rewards)
            reward_log.append(episode_reward)

            avg_reward = np.mean(reward_log[-10:])
            print(f"Episode {episode}, reward: {episode_reward:.2f}, avg: {avg_reward:.2f}")
            
            if episode_reward >= reward_threshold:
                print(f"Stopping early at episode {episode} with reward {episode_reward:.2f}")
                break
        # Plot reward history
        import matplotlib.pyplot as plt
        plt.plot(reward_log)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Progress")
        plt.grid(True)
        plt.show()
        run_simulation(env, model)

    except KeyboardInterrupt:
        # Plot reward history
        import matplotlib.pyplot as plt
        plt.plot(reward_log)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Progress")
        plt.grid(True)
        plt.show()
        run_simulation(env, model)
        skip_simulation = True

training_loop()

if not skip_simulation:
    run_simulation(env, model)