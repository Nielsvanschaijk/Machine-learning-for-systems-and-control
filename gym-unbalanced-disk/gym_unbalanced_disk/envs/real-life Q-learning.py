
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from os import path
import pickle
from matplotlib import pyplot as plt
import gym_unbalanced_disk
import UnbalancedDiskExp
import gymnasium
import numpy as np
from matplotlib import pyplot as plt
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
        self.action_space = spaces.Box(low=-3.0,high=3.0,shape=tuple()) #continuous
        
        self.action_space = spaces.Discrete(7)#7) #discrete
    def discretize(self,observation): #b)
        return tuple(((observation - self.olow)/(self.ohigh - self.olow)*self.nvec).astype(int)) #b)
        
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action) #b)
        return self.discretize(observation), reward, terminated, truncated, info #b)

    def reset(self):
        obs, info = self.env.reset()
        obs_dis = self.discretize(obs)  #b=)
        return obs_dis, info



def argmax(a):
    #argmax([0,1,2,3]) -> 3
    #argmax([0,1,2,2]) -> 2 or 3 with equal probability of both (np.argmax would only return 2)
    #argmax([0,0,0,0]) -> 0, 1, 2 or 3 with equal probability of each (np.argmax would only return 0)
    a = np.array(a)
    return np.random.choice(np.arange(a.shape[0],dtype=int)[a==np.max(a)])



def Qlearn(env, nsteps=5000, callbackfeq=100, alpha=0.05,eps=0.9995, gamma=0.9): # was alpha = 0.2 eps 0.2 gamma = 0.99
    from collections import defaultdict
    Qmat = defaultdict(float) #any new argument set to zero
    env_time = env
    # env_time = env.unwrapped
    while not isinstance(env_time,gym.wrappers.TimeLimit):
        env_time = env_time.env
    ep_lengths = []
    ep_lengths_steps = []
    rewards = []
    omegas = []
    actions = []
    thetas = []
    delta_ths = []
    obs, info = env.reset()
    print('goal reached time:')
    for z in range(nsteps):

        if np.random.uniform()<eps:
            action = env.action_space.sample()
        else:
            action = argmax([Qmat[obs,i] for i in range(env.action_space.n)])
        actions.append(action)
        obs_new, reward, terminated, truncated, info = env.step(action)
        if terminated: #terminal state and not by timeout
            #saving results:
            print(env_time._elapsed_steps, end=' ')
            ep_lengths.append(env_time._elapsed_steps)
            ep_lengths_steps.append(z)
            
            #updating Qmat:
            A = reward - Qmat[obs,action] # adventage or TD
            Qmat[obs,action] += alpha*A
            obs, info = env.reset()
        else: #not terminal
            A = reward + gamma*max(Qmat[obs_new, action_next] for action_next in range(env.action_space.n)) - Qmat[obs,action]
            Qmat[obs,action] += alpha*A
            obs = obs_new
            
            if truncated: #terminal by truncation with timeout
                #saving results:
                ep_lengths.append(env_time._elapsed_steps)
                ep_lengths_steps.append(z)
                print('out', end=' ')
                
                #reset:
                obs, info = env.reset()
        eps = max(0.05, eps * 0.999) 
    print()
    
    return Qmat, np.array(ep_lengths_steps), np.array(ep_lengths), []

def roll_mean(ar,start=2000,N=50):
    s = 1-1/N
    k = start
    out = np.zeros(ar.shape)
    for i,a in enumerate(ar):
        k = s*k + (1-s)*a
        out[i] = k
    return out

def train():
    Qmats = {}
    try:
        for nvec in [10]:
            max_episode_steps = 300
            env = UnbalancedDiskExp.UnbalancedDisk_exp(umax = 3.0,dt = 0.025)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps) 
            env = Discretize_obs(env, nvec=nvec)

            print('nvec =', nvec)
            Qmat, ep_lengths_steps, ep_lengths, info = Qlearn(env, nsteps=350_000, callbackfeq=5000)
            plt.plot(ep_lengths_steps, roll_mean(ep_lengths, start=max_episode_steps), label=str(nvec))
            Qmats[nvec] = Qmat

        plt.legend()
        plt.show()
    finally:
        with open("real_qmats.pkl", "wb") as f:
            pickle.dump(Qmats, f)

def run_simulation():
    with open("real_qmats.pkl", "rb") as f:
        Qmats = pickle.load(f)
    import time
    env = UnbalancedDiskExp.UnbalancedDisk_exp(umax = 3.0,dt = 0.025)
    env = Discretize_obs(env, nvec=10) 
    Qmat = Qmats[10]

    obs, info = env.reset()
    omegas = []
    delta_ths = []
    ths = []
    omega_calcs = []
    Y = [obs]
    env.render()
    try:
        for i in range(300):
            time.sleep(1/24)
            u = argmax([Qmat[obs,i] for i in range(env.action_space.n)])
            obs, reward, done, truncated, info = env.step(u)
            omegas.append(info["omega"])
            delta_ths.append(info["delta_th"])
            ths.append(info["th"])
            omega_calcs.append(info["omega_calc"])
            Y.append(obs)
            env.render()
    finally:
        env.close()
    
    import numpy as np
    Y = np.array(Y)
    undiscretizedY = []
    for item in Y[:,0]:
        undiscretizedItem = approx_observation = -np.pi + (item + 0.5) * 2*np.pi / 100
        undiscretizedY.append(undiscretizedItem)
    undiscretizedY = np.array(undiscretizedY)
    plt.plot(undiscretizedY)
    plt.title(f'max(Y[:,0])={max(undiscretizedY)}')
    plt.show()
    with open('real-life_delta_thetas.pkl', 'wb') as f:
        pickle.dump(delta_ths, f)
    with open('real-life_thetas.pkl', 'wb') as f:
        pickle.dump(ths, f)
    with open('real-life_omegas.pkl', 'wb') as f:
        pickle.dump(omegas, f)
    with open('real-life_omega_calcs.pkl', 'wb') as f:
        pickle.dump(omega_calcs, f)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model and save Q-table')
    parser.add_argument('--simulate', action='store_true', help='Run simulation using saved Q-table')
    args = parser.parse_args()
    # train()
    run_simulation()