
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


def plots_still():
    with open("real_75mil_qmats.pkl", "rb") as f:
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
            omega_calcs.append(info["omega_calc"])
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

    # with open('real-life_delta_thetas.pkl', 'wb') as f:
    #     pickle.dump(delta_ths, f)
    # with open('real-life_thetas.pkl', 'wb') as f:
    #     pickle.dump(ths, f)
    with open('real-life_omegas_plots.pkl', 'wb') as f:
        pickle.dump(omegas, f)
    with open('real-life_omega_calcs_plots.pkl', 'wb') as f:
        pickle.dump(omega_calcs, f)



def plots_moving():
    # with open("real_75mil_qmats.pkl", "rb") as f:
    #     Qmats = pickle.load(f)
    # import time
    # env = UnbalancedDiskExp.UnbalancedDisk_exp(umax = 3.0,dt = 0.025)
    # env = Discretize_obs(env, nvec=10)
    # Qmat = Qmats[10]

    # obs, info = env.reset()
    # Y = [obs]
    # env.render()
    # with open("real-life_thetas.pkl", 'rb') as f:
    #     thetas_01 = pickle.load(f) 
    with open("real-life_thetas_2.pkl", 'rb') as f:
        thetas_006 = pickle.load(f) 
    with open("real-life_thetas_3.pkl", 'rb') as f:
        thetas_0065 = pickle.load(f) 

    with open("real-life_omega_calcs.pkl", 'rb') as f:
        omega_calcs_01 = pickle.load(f) 
    with open("real-life_omega_calcs_2.pkl", 'rb') as f:
        omega_calcs_006 = pickle.load(f) 
    with open("real-life_omega_calcs_3.pkl", 'rb') as f:
        omega_calcs_0065 = pickle.load(f) 

    with open("real-life_omegas.pkl", 'rb') as f:
        omegas_01 = pickle.load(f) 
    with open("real-life_omegas_2.pkl", 'rb') as f:
        omegas_006 = pickle.load(f) 
    with open("real-life_omegas_3.pkl", 'rb') as f:
        omegas_0065 = pickle.load(f) 

    with open("real-life_delta_thetas.pkl", 'rb') as f:
        delta_thetas_01 = pickle.load(f) 
    with open("real-life_delta_thetas_2.pkl", 'rb') as f:
        delta_thetas_006 = pickle.load(f) 
    with open("real-life_delta_thetas_3.pkl", 'rb') as f:
        delta_thetas_0065 = pickle.load(f) 
    # try:
    #     for i in range(100):
    #         time.sleep(1/24)
    #         u = argmax([Qmat[obs,i] for i in range(env.action_space.n)])
    #         obs, reward, done, truncated, info = env.step(u)
    #         thetas.append(info["th"])
    #         omega_calcs.append(info["omega_calc"])
    #         omegas.append(info["omega"])
    #         delta_ths.append(info["delta_th"])
    #         Y.append(obs)
    #         env.render()
    # finally:
    #     env.close()
    
    import numpy as np

    # angle vs velocity
    fig, ax = plt.subplots()
    timesteps = np.arange(500)
    points = np.array([thetas_006, omega_calcs_006]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    from matplotlib.collections import LineCollection
# Create LineCollection
    lc = LineCollection(segments, cmap='viridis_r', norm=plt.Normalize(timesteps.min(), timesteps.max()))
    lc.set_array(timesteps)
    lc.set_linewidth(2)
    # ax.plot(thetas_006, omega_calcs_01)
    ax.add_collection(lc)
    sc = ax.scatter(thetas_006, omega_calcs_006, c=timesteps, cmap='viridis_r', marker='x')
    plt.xlabel("angle (rad)")
    plt.ylabel("angular velocity (rad/s)")
    plt.colorbar(sc, label="Timestep")
    plt.title("angular velocity vs angle")
    ax.axhline(y=0, color='r')
    ax.axvline(x=np.pi, color='r')
    ax.text(3.14, ax.get_ylim()[0] - 1.2, '3.14', ha='center', va='top', color='red')
    ax.text(3.14, ax.get_ylim()[0], '|', ha='center', va='top', color='red')
    plt.savefig('real angle vs velocity.png', dpi=300, bbox_inches='tight')
    plt.show()

    # velocity over time
    
    fig, ax = plt.subplots()
    # ax.plot(timesteps, omega_calcs_006, 'bx')
    sc = ax.scatter(timesteps, omega_calcs_006, marker='x', c=timesteps, cmap = 'viridis_r')
    points = np.array([timesteps, omega_calcs_006]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis_r', norm=plt.Normalize(timesteps.min(), timesteps.max()))
    lc.set_array(timesteps)
    lc.set_linewidth(1)
    ax.add_collection(lc)
    plt.colorbar(sc, label="Timestep")
    plt.xlabel("Time step")
    plt.ylabel("angular velocity (rad/s)")
    plt.title("angular velocity over time")
    ax.axhline(y=0, color='r')
    plt.savefig('real velocity over time.png', dpi=300, bbox_inches='tight')
    plt.show()

    # angle over time
    fig, ax = plt.subplots()
    ymin = min(thetas_006)
    ymax = max(thetas_006)
    # Compute lower and upper bounds for odd multiples of π
    lower_top = int(np.ceil((ymin - np.pi) / (2 * np.pi)))
    upper_top = int(np.floor((ymax - np.pi) / (2 * np.pi)))

    # Generate odd multiples of π within the range
    top_positions = (2 * np.arange(lower_top, upper_top + 1) + 1) * np.pi

    bottom_multiples = np.arange(np.floor(ymin / (2 * np.pi)), np.ceil(ymax / (2*np.pi)) + 1)
    bottom_positions = bottom_multiples[1:-1] * 2*np.pi

    sc = ax.scatter(timesteps, thetas_006, marker='x', c=timesteps, cmap = 'viridis_r')
    points = np.array([timesteps, thetas_006]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis_r', norm=plt.Normalize(timesteps.min(), timesteps.max()))
    lc.set_array(timesteps)
    lc.set_linewidth(1)
    ax.add_collection(lc)

    for pos in top_positions:
        ax.axhline(y=pos, color='red', linestyle='--', linewidth=0.8)
        ax.text(0, pos, 'top', color='red', fontsize=9, va='bottom', ha='left')

    for pos in bottom_positions:
        ax.axhline(y=pos, color='blue', linestyle='--', linewidth=0.8)
        ax.text(0, pos, 'bottom', color='blue', fontsize=9, va='bottom', ha='left')

    plt.colorbar(sc, label="Timestep")
    
    plt.xlabel("Time step")
    plt.ylabel("angle (rad)")
    plt.title("angle over time")
    plt.savefig('real angle over time.png', dpi=300, bbox_inches='tight')
    plt.show()
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model and save Q-table')
    parser.add_argument('--simulate', action='store_true', help='Run simulation using saved Q-table')
    args = parser.parse_args()
    # plots_still()
    plots_moving()