import numpy as np
import gymnasium as gym
import time
from AC_Continue_Learning import UnbalancedDisk, Discretize_obs  # Replace with actual filename

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_actor(env, Actor):
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    while True:#not (terminated or truncated):
        time.sleep(1/24)
        env.render()
        probs = softmax(Actor[obs])
        action = np.random.choice(np.arange(env.action_space.n), p=probs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print("Episode ended. Total reward:", total_reward)
    env.close()

if __name__ == '__main__':
    # Load saved actor policy
    try:
        Actor = np.load('actor_policy_highest.npy')
        print("Loaded actor policy.")
    except FileNotFoundError:
        print("Error: 'actor_policy.npy' not found.")
        exit()

    # Create and wrap environment
    env = UnbalancedDisk()
    env = gym.wrappers.TimeLimit(env, max_episode_steps=300)
    env = Discretize_obs(env, nvec=40)  # Assuming square nvec

    # Run simulation using loaded actor
    run_actor(env, Actor)
