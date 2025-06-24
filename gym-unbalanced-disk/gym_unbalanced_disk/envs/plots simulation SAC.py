
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from os import path
import pickle
from matplotlib import pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.env_util import make_vec_env


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
        
        # self.action_space = spaces.Discrete(7)#7) #discrete

        low = [-float("inf"),-40] 
        high = [float("inf"),40]
        self.observation_space = spaces.Box(low=np.array(low,dtype=np.float32),high=np.array(high,dtype=np.float32),shape=(2,))
        nvec = nvec 
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
            np.exp(-((self.th % (2 * np.pi) - np.pi) ** 2) / (2 * (np.pi / 7) ** 2)) + 0.2 * (1 - np.cos(self.th)) - 0.001 * float(self.u)**2 - 0.001 * abs(self.omega)
            
        )
        self.render_mode = render_mode
        self.viewer = None
        self.u = 0 #for visual
        self.reset()

    def step(self, action):
        self.u = action
        # self.u = [-3, -1, -0.5, 0, 0.5, 1, 3][action]
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
        # terminated = False
        # if terminated:
        #     reward += 1000.0

        return self.get_obs(), reward, False, False, {"ths": self.th, "omegas": self.omega, "delta_ths": self.delta_th}

         
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

class UnbalancedDisk_sincos(UnbalancedDisk):
    """docstring for UnbalancedDisk_sincos"""
    def __init__(self, umax=3., dt = 0.025):
        super(UnbalancedDisk_sincos, self).__init__(umax=umax, dt=dt)
        low = [-1,-1,-40.] 
        high = [1,1,40.]
        self.observation_space = spaces.Box(low=np.array(low,dtype=np.float32),high=np.array(high,dtype=np.float32),shape=(3,))

    def get_obs(self):
        self.th_noise = self.th + np.random.normal(loc=0,scale=0.001) #do not edit
        self.omega_noise = self.omega + np.random.normal(loc=0,scale=0.001) #do not edit
        return np.array([np.sin(self.th_noise), np.cos(self.th_noise), self.omega_noise]) #change anything here



def argmax(a):
    #argmax([0,1,2,3]) -> 3
    #argmax([0,1,2,2]) -> 2 or 3 with equal probability of both (np.argmax would only return 2)
    #argmax([0,0,0,0]) -> 0, 1, 2 or 3 with equal probability of each (np.argmax would only return 0)
    a = np.array(a)
    return np.random.choice(np.arange(a.shape[0],dtype=int)[a==np.max(a)])

def make_env(experiment=False, render_mode=None):
    env = UnbalancedDisk_sincos(umax=3.0, dt=0.025)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=300) 
    return env


def plots_moving():
    vec_env = make_vec_env(lambda: make_env(), n_envs=8)
    vec_env = VecMonitor(vec_env)

    model = SAC.load("sac model test", env=vec_env)
    demo_env = make_env(experiment=False, render_mode="human")
    obs, _ = demo_env.reset()
    angles = [0]
    velocities = [0]
    import time
    for _ in range(499):  # or however long you want to run
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = demo_env.step(action)
        demo_env.render()
        angles.append(info['ths'])
        velocities.append(info['omegas'])
        time.sleep(1/24)  # Control rendering speed
    demo_env.close()
    
    import numpy as np
    timesteps = np.arange(500)
    # angle vs velocity
    fig, ax = plt.subplots()
    points = np.array([angles, velocities]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, cmap='viridis_r', norm=plt.Normalize(timesteps.min(), timesteps.max()))
    lc.set_array(timesteps)
    lc.set_linewidth(2)

    xmin = min(angles)
    xmax = max(angles)
    lower_top = int(np.ceil((xmin - np.pi) / (2 * np.pi)))
    upper_top = int(np.floor((xmax - np.pi) / (2 * np.pi)))
    top_positions = (2 * np.arange(lower_top, upper_top + 1) + 1) * np.pi

    bottom_multiples = np.arange(np.floor(xmin / (2 * np.pi)), np.ceil(xmax / (2*np.pi)) + 1)
    bottom_positions = bottom_multiples[1:-1] * 2*np.pi

    ax.add_collection(lc)
    sc = ax.scatter(angles, velocities, c=timesteps, cmap='viridis_r', marker='x')
    for pos in top_positions:
        ax.axvline(x=pos, color='red', linestyle='--', linewidth=0.8)
        ax.text(pos + 0.1, ax.get_ylim()[0], 'top', color='red', fontsize=9, va='bottom', ha='left')

    for pos in bottom_positions:
        ax.axvline(x=pos, color='blue', linestyle='--', linewidth=0.8)
        ax.text(pos + 0.1, ax.get_ylim()[0], 'bottom', color='blue', fontsize=9, va='bottom', ha='left')

    plt.xlabel("angle (rad)")
    plt.ylabel("angular velocity (rad/s)")
    plt.colorbar(sc, label="Timestep")
    plt.title("angular velocity $\omega$ vs angle $\\theta$ for SAC")
    ax.axhline(y=0, color='r')
    plt.savefig('SAC sim angle vs velocity.png', dpi=300, bbox_inches='tight')
    plt.show()

    # velocity over time
    fig, ax = plt.subplots()
    plt.plot(timesteps, velocities)
    points = np.array([timesteps, velocities]).T.reshape(-1, 1, 2)
    plt.xlabel("Time step")
    plt.ylabel("angular velocity (rad/s)")
    plt.title("angular velocity $\omega$ over time for SAC")
    ax.axhline(y=0, color='r', linestyle='--', linewidth = 0.8)

    plt.savefig('SAC sim velocity over time.png', dpi=300, bbox_inches='tight')
    plt.show()

    # angle over time
    fig, ax = plt.subplots()
    ymin = min(angles)
    ymax = max(angles)
    # Compute lower and upper bounds for odd multiples of π
    lower_top = int(np.ceil((ymin - np.pi) / (2 * np.pi)))
    upper_top = int(np.floor((ymax - np.pi) / (2 * np.pi)))

    # Generate odd multiples of π within the range
    top_positions = (2 * np.arange(lower_top, upper_top + 1) + 1) * np.pi

    bottom_multiples = np.arange(np.floor(ymin / (2 * np.pi)), np.ceil(ymax / (2*np.pi)) + 1)
    bottom_positions = bottom_multiples[1:-1] * 2*np.pi

    plt.plot(timesteps, angles)

    for pos in top_positions:
        ax.axhline(y=pos, color='red', linestyle='--', linewidth=0.8)
        ax.text(0, pos, 'top', color='red', fontsize=9, va='bottom', ha='left')

    for pos in bottom_positions:
        ax.axhline(y=pos, color='blue', linestyle='--', linewidth=0.8)
        ax.text(0, pos, 'bottom', color='blue', fontsize=9, va='bottom', ha='left')

        
    plt.xlabel("Time step")
    plt.ylabel("angle (rad)")
    plt.title("angle $\\theta$ over time for SAC")
    plt.savefig('SAC sim angle over time.png', dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model and save Q-table')
    parser.add_argument('--simulate', action='store_true', help='Run simulation using saved Q-table')
    args = parser.parse_args()
    plots_moving()
