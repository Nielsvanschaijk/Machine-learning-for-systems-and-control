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

    def discretize(self,observation): #b)
        # print("observation", type(observation))
        # print("olow", type(self.olow), np.array(self.olow))
        # print("minus", observation - self.olow)
        # print("minus2", np.array(self.ohigh) - np.array(self.olow))
        # print((observation - self.olow)/(np.array(self.ohigh) - np.array(self.olow)))
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
        # self.action_space = gym.spaces.Box(low=-3, high = 3)
        self.observation_space = gym.spaces.Box(low=np.array([-np.pi, -40], dtype=np.float32),
                                                 high=np.array([np.pi, 40], dtype=np.float32), shape=(2,))
        l = np.array([-float('inf')]).astype(np.float32)
        # self.P = P
        self.th = 0
        self.omega = 0
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
        self.x = np.array([self.th, self.omega])
        self.r_matrix = np.array([[5, 0], [0, 0.1]])
        self.P = self.reward_fun
            # Big reward for being upright
            

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
        self.x = np.array([self.th, self.omega])

        # reward = self.reward_fun(self)
        reward = self.P(self)
        # terminated = False
        terminated = np.abs(self.th) > 3.05 and np.abs(self.omega) < 0.1
      
        return self.get_obs(), reward, terminated, False, [self.th, self.omega, self.delta_th]

    def reset(self, seed=None, options=None):
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


def eval_actor(Actor,env, obs_start, deterministic=True):
    actions = np.arange(env.action_space.n,dtype=int)
    reward_acc = 0
    obs, info = env.reset()
    while True:
        action = np.argmax(Actor[obs]) if deterministic else np.random.choice(actions,p=softmax(Actor[obs_start]))
        obs, reward, terminated, truncated, info = env.step(action)
        reward_acc += reward
        if terminated or truncated:
            env.reset()
            return reward_acc
    
def softmax(h):
    hp = h-np.max(h)
    return np.exp(hp)/np.sum(np.exp(hp))
# env = gym.make('CartPole-v1')
# t, n_episodes=1000, gamma=0.995, tau=1e-3,
                    #    initial_epsilon=1.0, final_epsilon=0.05, epsilon_decay=0.995):

from tqdm.auto import tqdm
def train_actor_critic(env, nvec=10):
    Actor = np.zeros((nvec, nvec, env.action_space.n)) #a=) #array of size (N states, N actions)
    Critic = np.zeros((nvec, nvec)) #a=) array of size (N states,)
    actions = np.arange(env.action_space.n,dtype=int)

    obs_start, info = env.reset()
    step_size_actor = 0.1
    step_size_critic = 0.3
    gamma = 0.95
    rewards = []
    index = []
    it = 0
    for i in tqdm(range(40_000)):
        #take action
        probs = softmax(Actor[obs_start]) #b=)
        action = np.random.choice(actions,p=probs) #b=)
        
        obs_next, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            returns = reward #b)
        else:
            returns = reward+gamma*Critic[obs_next] #b)
        
        advantage = (returns-Critic[obs_start]) #b=)
        
        tmp = np.zeros((env.action_space.n,)) #c)
        tmp[action] = 1 #c)
        grad_actor = (tmp-probs)*advantage #c)  
        grad_critic = -advantage #c) 
        
        Actor[obs_start] += step_size_actor*grad_actor #c)
        Critic[obs_start] -= step_size_critic*grad_critic #c)
        
        if terminated or truncated:
            # print("terminated")
            it += 1
            if it%10==0:
                rewards.append(np.mean([eval_actor(Actor,env, obs_start) for i in range(200)]))
                index.append(i)
            obs_start, info = env.reset()
        else:
            obs_start = obs_next
    plt.plot(index,rewards,'.')
    plt.xlabel('update count')
    plt.ylabel('mean episode reward')
    plt.show()
    np.save('actor_policy.npy', Actor)



# Run simulation (visualize the policy)
def show_self(env):
    Actor = np.load('actor_policy.npy')
    obs, info = env.reset()
    env.render()
    while True:
        action = np.argmax(Actor[obs]) #e=)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            env.close()
            break

if __name__ == '__main__':
    env_name = 'UnbalancedDisk'
    env = UnbalancedDisk()
    env = gym.wrappers.TimeLimit(env, max_episode_steps=400) 
    env = Discretize_obs(env, nvec=10)
    # actor_crit = ActorCritic(env)
    # actor_crit = ActorCritic(env)

    # Train the Actor-Critic model
    # train_actor_critic(env, actor_crit)
    train_actor_critic(env)

    # Show the trained policy
    show_self(env)
