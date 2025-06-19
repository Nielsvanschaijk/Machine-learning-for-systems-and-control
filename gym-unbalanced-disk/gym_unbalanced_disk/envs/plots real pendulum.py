
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
        self.action_space = spaces.Box(low=-3.0,high=3.0,shape=tuple()) #continuous
        
        self.action_space = spaces.Discrete(7)#7) #discrete
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
        # print("reward", reward, "   info", info)
        # rewards.append(reward)
        # thetas.append(info[0])
        # omegas.append(info[1])
        # delta_ths.append(info[2])
        if terminated: #terminal state and not by timeout
            #saving results:
            print(env_time._elapsed_steps, end=' ')
            ep_lengths.append(env_time._elapsed_steps)
            ep_lengths_steps.append(z)
            # print("terminated") # verwijderen
            
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


def plots_still():
    with open("real_qmats.pkl", "rb") as f:
        Qmats = pickle.load(f)
    import time
    env = UnbalancedDiskExp.UnbalancedDisk_exp(umax = 3.0,dt = 0.025)
    env = Discretize_obs(env, nvec=10) 
    Qmat = Qmats[10]

    obs, info = env.reset()
    omegas = []
    delta_ths = []
    omega_calcs = []
    ths = []
    Y = [obs]
    env.render()
    try:
        for i in range(100):
            time.sleep(1/24)
            u = 3
            obs, reward, done, truncated, info = env.step(u)
            omegas.append(info["omega"])
            delta_ths.append(info["delta_th"])
            omega_calcs.append(info["omega_calcs"])
            Y.append(obs)
            env.render()
    finally:
        env.close()
    
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    ax1.tick_params(labelbottom=True)
    timesteps = np.arange(100)
    ax1.plot(timesteps, omegas, label='Velocity 1')
    ax1.set_ylabel('velocity (rad/s)')
    ax1.set_xlabel('Time step')
    ax1.set_title("$\omega$ per time step for $u=0$ and $\\theta = 0$")
    ax1.grid(True)
    ax1.set_xlim(0, 99)

    ax2.plot(timesteps, omega_calcs, label='Velocity 2', color='orange')
    ax2.set_ylabel('velocity (rad/s)')
    ax2.set_xlabel('Time Step')
    ax2.set_title('$\omega_{calc}$ per time step for $u=0$ and $\\theta = 0$')
    ax2.grid(True)
    ax2.set_xlim(0,99)
    plt.subplots_adjust(hspace=0.4)
    

    plt.savefig('real omega vs omega_calc.png', dpi=300, bbox_inches='tight')
    plt.show()

    with open('real-life_delta_thetas.pkl', 'wb') as f:
        pickle.dump(delta_ths, f)
    with open('real-life_thetas.pkl', 'wb') as f:
        pickle.dump(ths, f)
    with open('real-life_omegas.pkl', 'wb') as f:
        pickle.dump(omegas, f)

def plots_moving():
    with open("real_qmats.pkl", "rb") as f:
        Qmats = pickle.load(f)
    import time
    env = UnbalancedDiskExp.UnbalancedDisk_exp(umax = 3.0,dt = 0.025)
    env = Discretize_obs(env, nvec=10)
    Qmat = Qmats[10]

    obs, info = env.reset()
    Y = [obs]
    env.render()
    thetas = []
    omega_calcs = []
    omegas = []
    delta_ths = []
    omega_calcs = []
    try:
        for i in range(100):
            time.sleep(1/24)
            u = argmax([Qmat[obs,i] for i in range(env.action_space.n)])
            obs, reward, done, truncated, info = env.step(u)
            thetas.append(info["th"])
            omega_calcs.append(info["omega_calc"])
            omegas.append(info["omega"])
            delta_ths.append(info["delta_th"])
            Y.append(obs)
            env.render()
    finally:
        env.close()
    
    import numpy as np

    # angle vs velocity
    fig, ax = plt.subplots()
    ax.plot(thetas, omega_calcs, 'bx')
    plt.xlabel("angle (rad)")
    plt.ylabel("angular velocity (rad/s)")
    plt.title("angular velocity vs angle")
    ax.axhline(y=0, color='r')
    ax.axvline(x=np.pi, color='r')
    ax.text(3.14, ax.get_ylim()[0] - 1.2, '3.14', ha='center', va='top', color='red')
    ax.text(3.14, ax.get_ylim()[0], '|', ha='center', va='top', color='red')
    plt.savefig('sim angle vs velocity.png', dpi=300, bbox_inches='tight')
    plt.show()

    # velocity over time
    timesteps = np.arange(100)
    fig, ax = plt.subplots()
    ax.plot(timesteps, omega_calcs, 'bx')
    plt.xlabel("Time step")
    plt.ylabel("angular velocity (rad/s)")
    plt.title("angular velocity over time")
    ax.axhline(y=0, color='r')
    plt.savefig('sim velocity over time.png', dpi=300, bbox_inches='tight')
    plt.show()

    # angle over time
    timesteps = np.arange(100)
    fig, ax = plt.subplots()
    ax.plot(timesteps, thetas, 'bx')
    plt.xlabel("Time step")
    plt.ylabel("angle (rad)")
    plt.title("angle over time")
    ax.axhline(y=np.pi, color='r')
    ax.text(ax.get_xlim()[0] - 5, np.pi, '3.14', va='center', ha='right', color='red')
    ax.text(ax.get_xlim()[0], np.pi, '-', va='center', ha='right', color='red')
    plt.savefig('sim angle over time.png', dpi=300, bbox_inches='tight')
    plt.show()
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model and save Q-table')
    parser.add_argument('--simulate', action='store_true', help='Run simulation using saved Q-table')
    args = parser.parse_args()
    plots_still()
    plots_moving()