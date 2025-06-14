
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from os import path
import pickle
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

    def discretize(self,observation): #b)
        return tuple(((observation - self.olow)/(self.ohigh - self.olow)*self.nvec).astype(int)) #b)
        
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
        
        self.action_space = spaces.Discrete(7)#7) #discrete
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
        self.reward_fun = lambda self: (
            # Big reward for being upright
            1000 * np.cos(self.th - np.pi)

            # Reward for being upright for a long time
            + 100 * np.cos(self.th - np.pi) * (self.dt / 0.025)  # dt is the time step
            
            # Reward for swing amplitude: high when |th| is large (upside)
            + 100 * abs(np.sin(self.th / 2))  # peaks at th=±π
            
            # Reward fast motion near bottom to encourage energy build-up
            + 0.5 * (1 - np.cos(self.th)) * abs(self.omega)
            
            # Penalize control effort
            - 0.001 * self.u**2

            # Penalize no swing angle at the bottom
            - 0.1 * abs(self.delta_th) if abs(self.delta_th) < np.pi/2 else 0

            # Pelanize large swing angle at the top
            - 50 * abs(self.omega) if abs(self.delta_th) >= np.pi-0.1415 else 0

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
        self.u = [-3, -1, -0.5, 0, 0.5, 1, 3][action]
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
        rewards.append(reward)
        thetas.append(info[0])
        omegas.append(info[1])
        delta_ths.append(info[2])
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
    
    return Qmat, np.array(ep_lengths_steps), np.array(ep_lengths), [rewards, omegas, actions, thetas, delta_ths]

def roll_mean(ar,start=2000,N=50):
    s = 1-1/N
    k = start
    out = np.zeros(ar.shape)
    for i,a in enumerate(ar):
        k = s*k + (1-s)*a
        out[i] = k
    return out


def run_simulation():
    with open("qmats.pkl", "rb") as f:
        Qmats = pickle.load(f)
    import time
    env = UnbalancedDisk(dt=0.025)
    env = Discretize_obs(env, nvec=10) # was 100
    Qmat = Qmats[10]

    obs, info = env.reset()
    Y = [obs]
    env.render()
    angles = []
    velocities = []
    try:
        for i in range(100):
            time.sleep(1/24)
            u = argmax([Qmat[obs,i] for i in range(env.action_space.n)])
            obs, reward, done, truncated, info = env.step(u)
            angles.append(info[0])
            velocities.append(info[1])
            Y.append(obs)
            env.render()
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
    # plt.plot(undiscretizedY)
    # plt.title(f'max(Y[:,0])={max(undiscretizedY)}')
    # plt.show()

    # plt.plot(angles, velocities, 'bx')
    # plt.xlabel("angle (rad)")
    # plt.ylabel("angular velocity (rad/s)")
    # plt.title("angular velocity vs angle")
    # plt.show()
    fig, ax = plt.subplots()
    ax.plot(angles, velocities, 'bx')
    plt.xlabel("angle (rad)")
    plt.ylabel("angular velocity (rad/s)")
    plt.title("angular velocity vs angle")
    ax.axhline(y=0, color='r')
    ax.axvline(x=np.pi, color='r')
    # xticks = list(ax.get_xticks())  # get current ticks
    # xticks.append(3.14)            # add your custom tick
    # ax.set_xticks(sorted(xticks))
    ax.text(3.14, ax.get_ylim()[0] - 1.5, '3.14', ha='center', va='top', color='red')
    ax.text(3.14, ax.get_ylim()[0], '|', ha='center', va='top', color='red')
    plt.show()
    fig.savefig("angle vs velocity.png")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model and save Q-table')
    parser.add_argument('--simulate', action='store_true', help='Run simulation using saved Q-table')
    args = parser.parse_args()
    # train()
    run_simulation()