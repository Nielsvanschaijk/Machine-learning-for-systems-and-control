
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from os import path
import pickle
import torch.nn as nn
import torch
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
import time

# class gekopieerd van opdracht 6
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
        idx = ((observation - self.olow)/(self.ohigh - self.olow)*self.nvec).astype(int)
        idx = np.clip(idx, 0, self.nvec-1)  # <-- Add this line
        return tuple(idx)
          
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action) #b)
        return self.discretize(observation), reward, terminated, truncated, info #b)

    def reset(self):
        obs, info = self.env.reset()
        obs_dis = self.discretize(obs)  #b=)
        return obs_dis, info


class UnbalancedDisk(gym.Env):
    def __init__(self,nvec=40, umax=3., dt = 0.025, render_mode='human'):
        ############# start do not edit  ################
        self.omega0 = 11.339846957335382
        self.delta_th = 0
        self.gamma = 1.3328339309394384
        self.Ku = 28.136158407237073
        self.Fc = 6.062729509386865
        self.coulomb_omega = 0.001

        # self.g = 9.80155078791343
        # self.J = 0.000244210523960356
        # self.Km = 10.5081817407479
        # self.I = 0.0410772235841364
        # self.M = 0.0761844495320390
        # self.tau = 0.397973147009910
        ############# end do not edit ###################

        self.umax = umax
        self.dt = dt #time step
 

        # change anything here (compilable with the exercise instructions)
        self.action_space = spaces.Box(low=-umax,high=umax,shape=tuple()) #continuous
        
        self.action_space = spaces.Discrete(6)#7) #discrete
        # print(self.action_space)
        # low = [-float('inf'),-40] 
        # high = [float('inf'),40]
        # aangepast
        low = [-np.pi,-40] 
        high = [np.pi,40]
        self.observation_space = spaces.Box(low=np.array(low,dtype=np.float32),high=np.array(high,dtype=np.float32),shape=(2,))
        # print(self.observation_space)
        nvec = nvec # was 100
        '''
        UnbalancedDisk
        th =            
                    +-pi
                        |
            pi/2   ----- -pi/2
                        |
                        0  = starting location
        '''
        # self.reward_fun = lambda self: (
        #     # # Big reward for being upright
        #     100 * abs(np.sin(self.th))

        #     # # Reward for being upright for a long time
        #     + 500 * np.cos(self.th - np.pi) * (self.dt / 0.025)  # dt is the time step
            
        #     # # Reward for swing amplitude: high when |th| is large (upside)
        #     # + 10 * abs(np.sin(self.th / 2))  # peaks at th=±π
            
        #     # Reward fast motion near bottom to encourage energy build-up
        #     + 25 * abs(self.th) * abs(self.omega)
            
        #     # Penalize control effort
        #     - 0.001 * self.u**2

        #     # Penalize no swing angle at the bottom
        #     - 0.1 * abs(self.delta_th) if abs(self.delta_th) < 1 else 0

        #     # Pelanize large swing angle at the top
        #     # - 50 * abs(self.omega) if abs(self.delta_th) >= np.pi-0.1415 else 0

        # )
        self.reward_fun = lambda self: (
            # Reward being near upright (θ ≈ π)
            + 10 * np.cos(self.th - np.pi)

            # Encourage motion: more ω = more potential to swing
            + 0.5 * abs(self.omega)

            # Encourage being far from bottom (θ = 0)
            + 2.0 * abs(np.sin(self.th))

            # Penalize excessive torque
            - 0.01 * self.u**2

            # Bonus for being very upright and stable
            + (20 if abs(self.th - np.pi) < 0.1 and abs(self.omega) < 0.2 else 0)
        )


        # self.reward_fun = lambda self: np.exp(-self.th)

        #                                100*(1-np.abs(self.costh)) if abs(self.costh) < np.pi
        #self.reward_fun = lambda self: 10000 if self.costh > 0.9 and np.abs(self.delta_th) > 0.1 else \
        #                                100 - 5 * np.abs(self.delta_th) if self.costh > 0.9  else \
        #                                75 * np.abs(self.delta_th) + 100 * self.costh 
        self.render_mode = render_mode
        self.viewer = None
        self.u = 0 #for visual
        self.reset()

    def step(self, action):
        self.u = [-3, -1, -0.5, 0.5, 1, 3][action]
        self.u = np.clip(self.u, -self.umax, self.umax)

        def f(t, y):
            th, omega = y
            dthdt = omega
            friction = self.gamma * omega + self.Fc * np.tanh(omega / self.coulomb_omega)
            domegadt = -self.omega0**2 * np.sin(th + self.delta_th) - friction + self.Ku * self.u
            return np.array([dthdt, domegadt])

        sol = solve_ivp(f, [0, self.dt], [self.th, self.omega])
        th, self.omega = sol.y[:, -1]
        self.delta_th = np.arctan2(np.sin(th - self.th), np.cos(th - self.th))
        self.th = th
        self.costh = -np.cos(th)

        reward = self.reward_fun(self)
        terminated = False
        #terminated = abs(np.arctan2(np.sin(self.th - np.pi), np.cos(self.th - np.pi))) < 0.05 and abs(self.omega) < 0.1
        if terminated:
            reward += 1000.0

        return self.get_obs(), reward, terminated, False, [self.th, self.omega, self.delta_th]

         
    def reset(self,seed=None, options=None):
        self.th = np.random.normal(loc=0,scale=0.001)
        self.omega = np.random.normal(loc=0,scale=0.001)
        self.u = 0
        self.delta_th = 0
        
        return self.get_obs(), {}

    def get_obs(self):
        self.th_noise = self.th + np.random.normal(loc=0,scale=0.001) #do not edit
        self.omega_noise = self.omega + np.random.normal(loc=0,scale=0.001) #do not edit
        return np.array([self.th_noise, self.omega_noise])

    def render(self):
        import pygame
        from pygame import gfxdraw
        
        screen_width = 500
        screen_height = 500

        th = self.th
        omega = self.omega #x = self.state

        if self.viewer is None:
            pygame.init()
            pygame.display.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))
        
        gfxdraw.filled_circle( #central blue disk
            self.surf,
            screen_width//2,
            screen_height//2,
            int(screen_width/2*0.65*1.3),
            (32,60,92),
        )
        gfxdraw.filled_circle( #small midle disk
            self.surf,
            screen_width//2,
            screen_height//2,
            int(screen_width/2*0.06*1.3),
            (132,132,126),
        )
        
        from math import cos, sin
        r = screen_width//2*0.40*1.3
        gfxdraw.filled_circle( #disk
            self.surf,
            int(screen_width//2-sin(th)*r), #is direction correct?
            int(screen_height//2-cos(th)*r),
            int(screen_width/2*0.22*1.3),
            (155,140,108),
        )
        gfxdraw.filled_circle( #small nut
            self.surf,
            int(screen_width//2-sin(th)*r), #is direction correct?
            int(screen_height//2-cos(th)*r),
            int(screen_width/2*0.22/8*1.3),
            (71,63,48),
        )
        
        fname = path.join(path.dirname(__file__), "clockwise.png")
        self.arrow = pygame.image.load(fname)
        if self.u:
            if isinstance(self.u, (np.ndarray,list)):
                if self.u.ndim==1:
                    u = self.u[0]
                elif self.u.ndim==0:
                    u = self.u
                else:
                    raise ValueError(f'u={u} is not the correct shape')
            else:
                u = self.u
            arrow_size = abs(float(u)/self.umax*screen_height)*0.25
            Z = (arrow_size, arrow_size)
            arrow_rot = pygame.transform.scale(self.arrow,Z)
            if self.u<0:
                arrow_rot = pygame.transform.flip(arrow_rot, True, False)
                
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.viewer.blit(self.surf, (0, 0))
        if self.u:
            self.viewer.blit(arrow_rot, (screen_width//2-arrow_size//2, screen_height//2-arrow_size//2))
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
import numpy as np

def rollout(actor_crit, env, N_rollout=20000):
    Start_state = []
    Actions = []
    Rewards = []
    End_state = []
    Terminal = []

    pi = lambda x: actor_crit.actor(torch.tensor(x[None, :], dtype=torch.float32))[0].detach().numpy()

    obs, info = env.reset()
    for i in range(N_rollout):
        probs = pi(obs)
        action = np.random.choice(env.action_space.n, p=probs)
        obs_next, reward, terminated, truncated, info = env.step(action)

        Start_state.append(obs)
        Actions.append(action)
        Rewards.append(reward)
        End_state.append(obs_next)
        Terminal.append(terminated or truncated)

        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs = obs_next

    return (np.array(Start_state), np.array(Actions), np.array(Rewards),
            np.array(End_state), np.array(Terminal))
                
    #error checking:
    assert len(Start_state)==len(Actions)==len(Rewards)==len(End_state)==len(Terminal), f'error in lengths: {len(Start_state)}=={len(Actions)}=={len(Rewards)}=={len(End_state)}=={len(Terminal)}'
    return np.array(Start_state), np.array(Actions), np.array(Rewards), np.array(End_state), np.array(Terminal).astype(int)

def eval_actor(actor_crit, env):
    
    with torch.no_grad():
        rewards_acc = 0 
        obs, info = env.reset() 
        while True: 
            action = np.argmax(pi(obs)) #b=)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards_acc += reward 
            if terminated or truncated: 
                return rewards_acc 
            
def show(actor_crit,env):
    pi = lambda x: actor_crit.actor(torch.tensor(x[None,:],dtype=torch.float32))[0].numpy()
    with torch.no_grad():
        try:
            obs, info = env.reset() 
            env.render() 
            time.sleep(1) 
            while True: 
                action = np.argmax(pi(obs)) #b=)
                obs, reward, terminated, truncated, info = env.step(action) 
                time.sleep(1/60) 
                env.render()
                if terminated or truncated: 
                    time.sleep(0.5) 
                    break  
        finally: #this will always run even when an error occurs
            env.close()

class ActorCritic(nn.Module):
    def __init__(self, env, hidden_size=32):
        super(ActorCritic, self).__init__()
        obs, _ = env.reset()
        num_inputs = len(obs)
        num_actions = env.action_space.n

        #define your layers here:
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)  #a)
        self.critic_linear2 = nn.Linear(hidden_size, 1) #a)
        self.actor_linear1 = nn.Linear(num_inputs, hidden_size) #a)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions) #a)
    
    def actor(self, state, return_logp=False):
        #state has shape (Nbatch, Nobs)
        hidden = torch.tanh(self.actor_linear1(state)) #a)
        h = self.actor_linear2(hidden) #a=)
        h = h - torch.max(h,dim=1,keepdim=True)[0] #for additional numerical stability
        logp = h - torch.log(torch.sum(torch.exp(h),dim=1,keepdim=True)) #log of the softmax
        if return_logp:
            return logp
        else:
            return torch.exp(logp) #by default it will return the probability
    
    def critic(self, state):
        #state has shape (Nbatch, Nobs)
        hidden = torch.tanh(self.critic_linear1(state)) #a)
        return self.critic_linear2(hidden)[:,0] #a) #no activation function
    
    def forward(self, state):
        #state has shape (Nbatch, Nobs)
        return self.critic(state), self.actor(state)
    
def roll_mean(ar,start=2000,N=10):
    s = 1-1/N
    k = start
    out = np.zeros(ar.shape)
    for i,a in enumerate(ar):
        k = s*k + (1-s)*a
        out[i] = k
    return out

import torch
import torch.nn as nn
import torch.optim as optim

def train():
    nvec = 10
    max_episode_steps = 200
    env = UnbalancedDisk(nvec=nvec, dt=0.025)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Discretize_obs(env, nvec=nvec)
    actor_crit = ActorCritic(env)
    optimizer = optim.Adam(actor_crit.parameters(), lr=1e-3)
    gamma = 0.99
    n_episodes = 200
    epsilon_start = 0.7
    epsilon_end = 0.05
    epsilon_decay = 0.95  # Decay rate per episode
    epsilon = epsilon_start

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        while not done:

            state = torch.tensor(np.array(obs)[None, :], dtype=torch.float32)  # <-- Add this line
            value = actor_crit.critic(state)
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                probs_np = actor_crit.actor(state)[0].detach().numpy()
                action = np.argmax(probs_np)

            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(np.array(obs_next)[None, :], dtype=torch.float32)

            # Critic values
            value = actor_crit.critic(state)
            next_value = actor_crit.critic(next_state)
            target = reward + (0 if done else gamma * next_value.item())
            advantage = target - value

            # Actor loss (policy gradient)
            probs = actor_crit.actor(state)[0]  # Always get as tensor for log_prob
            log_prob = torch.log(probs[action] + 1e-8)
            actor_loss = -log_prob * advantage.detach()

            # Critic loss (value regression)
            critic_loss = advantage.pow(2)

            # Total loss
            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = obs_next
            episode_reward += reward
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}, reward: {episode_reward/10:.2f}")
            print(f"Current epsilon: {epsilon:.4f}")

    # Optionally save the trained model
    torch.save(actor_crit.state_dict(), "actor_critic.pth")

def run_simulation(actor_crit, env):

    import numpy as np
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
                time.sleep(1/60)
                if terminated or truncated:
                    time.sleep(0.5)
                    break
        finally:
            env.close()
    import numpy as np
    # print("Y", len(Y), Y)
    Y = np.array(Y)
    undiscretizedY = []
    for item in Y[:,0]:
        undiscretizedItem = approx_observation = -np.pi + (item + 0.5) * 2*np.pi / 100
        undiscretizedY.append(undiscretizedItem)
    # plt.plot(Y[:,0])
    undiscretizedY = np.array(undiscretizedY)
    plt.plot(undiscretizedY)
    plt.title(f'max(Y[:,0])={max(undiscretizedY)}')
    plt.show()

if __name__ == '__main__':
    import argparse
    env_name = 'UnbalancedDisk'
    env = UnbalancedDisk()
    actor_crit = ActorCritic(env)
    train()
    run_simulation(actor_crit, env)