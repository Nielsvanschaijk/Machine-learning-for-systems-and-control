
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



def argmax(a):
    #argmax([0,1,2,3]) -> 3
    #argmax([0,1,2,2]) -> 2 or 3 with equal probability of both (np.argmax would only return 2)
    #argmax([0,0,0,0]) -> 0, 1, 2 or 3 with equal probability of each (np.argmax would only return 0)
    a = np.array(a)
    return np.random.choice(np.arange(a.shape[0],dtype=int)[a==np.max(a)])



def plots_moving():
    with open("sac_real-life_thetas.pkl", 'rb') as f:
        thetas_1 = pickle.load(f) 
    with open("sac_real-life_thetas_2.pkl", 'rb') as f:
        thetas_2 = pickle.load(f) 

    with open("sac_real-life_omega_calcs.pkl", 'rb') as f:
        omega_calcs_1 = pickle.load(f) 
    with open("sac_real-life_omega_calcs_2.pkl", 'rb') as f:
        omega_calcs_2 = pickle.load(f) 


    import numpy as np

    thetas = [thetas_2]
    omega_calcs = [omega_calcs_2]
    dts = ["1"]
    # angle vs velocity
    timesteps = np.arange(100)
    for i in range(len(dts)):
        fig, ax = plt.subplots()
        points = np.array([thetas[i], omega_calcs[i]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='viridis_r', norm=plt.Normalize(timesteps.min(), timesteps.max()))
        lc.set_array(timesteps)
        lc.set_linewidth(2)

        xmin = min(thetas[i])
        xmax = max(thetas[i])
        lower_top = int(np.ceil((xmin - np.pi) / (2 * np.pi)))
        upper_top = int(np.floor((xmax - np.pi) / (2 * np.pi)))
        top_positions = (2 * np.arange(lower_top, upper_top + 1) + 1) * np.pi

        bottom_multiples = np.arange(np.floor(xmin / (2 * np.pi)), np.ceil(xmax / (2*np.pi)) + 1)
        bottom_positions = bottom_multiples[1:-1] * 2*np.pi

        ax.add_collection(lc)
        sc = ax.scatter(thetas[i], omega_calcs[i], c=timesteps, cmap='viridis_r', marker='x')
        for pos in top_positions:
            ax.axvline(x=pos, color='red', linestyle='--', linewidth=0.8)
            ax.text(pos + 0.1, ax.get_ylim()[0], 'top', color='red', fontsize=9, va='bottom', ha='left')

        for pos in bottom_positions:
            ax.axvline(x=pos, color='blue', linestyle='--', linewidth=0.8)
            ax.text(pos + 0.1, ax.get_ylim()[0], 'bottom', color='blue', fontsize=9, va='bottom', ha='left')

        plt.xlabel("angle (rad)")
        plt.ylabel("angular velocity (rad/s)")
        plt.colorbar(sc, label="Timestep")
        plt.title("angular velocity $\omega_{calc}$ vs angle $\\theta$ for SAC")
        ax.axhline(y=0, color='k', linestyle = '--', linewidth = 0.8)
        plot_save_name = 'sac real angle vs velocity' + dts[i] + '.png'
        plt.savefig(plot_save_name, dpi=300, bbox_inches='tight')
        plt.show()

        # velocity over time
        fig, ax = plt.subplots()
        plt.plot(timesteps, omega_calcs[i])

        plt.xlabel("Time step")
        plt.ylabel("angular velocity (rad/s)")
        plt.title("angular velocity $\omega_{calc}$ over time for SAC")
        ax.axhline(y=0, color='r', linewidth = 0.8, linestyle = '--')
        plot_save_name = 'sac real velocity over time' + dts[i] + '.png'
        plt.savefig(plot_save_name, dpi=300, bbox_inches='tight')
        plt.show()

        # angle over time
        fig, ax = plt.subplots()
        ymin = min(thetas[i])
        ymax = max(thetas[i])
        # Compute lower and upper bounds for odd multiples of π
        lower_top = int(np.ceil((ymin - np.pi) / (2 * np.pi)))
        upper_top = int(np.floor((ymax - np.pi) / (2 * np.pi)))

        # Generate odd multiples of π within the range
        top_positions = (2 * np.arange(lower_top, upper_top + 1) + 1) * np.pi

        bottom_multiples = np.arange(np.floor(ymin / (2 * np.pi)), np.ceil(ymax / (2*np.pi)) + 1)
        bottom_positions = bottom_multiples[1:-1] * 2*np.pi

        plt.plot(timesteps, thetas[i])

        for pos in top_positions:
            ax.axhline(y=pos, color='red', linestyle='--', linewidth=0.8)
            ax.text(0, pos, 'top', color='red', fontsize=9, va='bottom', ha='left')

        for pos in bottom_positions:
            ax.axhline(y=pos, color='blue', linestyle='--', linewidth=0.8)
            ax.text(0, pos, 'bottom', color='blue', fontsize=9, va='bottom', ha='left')

        # plt.colorbar(sc, label="Timestep")
        
        plt.xlabel("Time step")
        plt.ylabel("angle (rad)")
        plt.title("angle $\\theta$ over time for SAC")
        plot_save_name = 'sac real angle over time' + dts[i] + '.png'
        plt.savefig(plot_save_name, dpi=300, bbox_inches='tight')
        plt.show()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model and save Q-table')
    parser.add_argument('--simulate', action='store_true', help='Run simulation using saved Q-table')
    args = parser.parse_args()
    plots_moving()
