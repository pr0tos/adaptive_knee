import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Callable

class Visualizer:

    @staticmethod
    def plot_results(t: np.ndarray, q1_func: Callable, d_q1_func: Callable, 
                     q2: np.ndarray, d_q2: np.ndarray):
        # Get values in degrees
        q1 = q1_func(t)  # q1_func already returns degrees
        d_q1 = d_q1_func(t)  # derivative in degrees/s
        q2_deg = np.rad2deg(q2)  # convert q2 to degrees
        d_q2_deg = np.rad2deg(d_q2)  # convert d_q2 to degrees/s

        # Create a figure for on-screen display with all three subdata
        plt.figure(figsize=(15, 5))
        
        # 1. Angles vs time
        plt.subplot(1, 3, 1)
        plt.plot(t, q1, label=r'$\theta_1$ (°) - hip')
        plt.plot(t, q2_deg, label=r'$\theta_2$ (°) - knee')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (°)')
        plt.legend()
        plt.grid()
        plt.title('Angles')

        # 2. Angular velocities vs time
        plt.subplot(1, 3, 2)
        plt.plot(t, d_q1, label=r'$\omega_1$ (°/s) - hip')
        plt.plot(t, d_q2_deg, label=r'$\omega_2$ (°/s) - knee')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular velocity (°/s)')
        plt.legend()
        plt.grid()
        plt.title('Angular Velocities')

        # 3. Parametric plot
        plt.subplot(1, 3, 3)
        plt.plot(q1, q2_deg, 'b-')
        plt.xlabel(r'$\theta_1$ (°)')
        plt.ylabel(r'$\theta_2$ (°)')
        plt.title('Parametric curve')
        plt.grid()

        plt.tight_layout()
        plt.show()

        # 1. Angles vs time
        plt.figure(figsize=(10, 6))
        plt.plot(t, q1, label=r'$\theta_1$ (°) - hip')
        plt.plot(t, q2_deg, label=r'$\theta_2$ (°) - knee')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (°)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig('data/angles.png')
        plt.close()

        # 2. Angular velocities vs time
        plt.figure(figsize=(10, 6))
        plt.plot(t, d_q1, label=r'$\omega_1$ (°/s) - hip')
        plt.plot(t, d_q2_deg, label=r'$\omega_2$ (°/s) - knee')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular velocity (°/s)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig('data/angular_velocities.png')
        plt.close()

        # 3. Parametric plot
        plt.figure(figsize=(10, 6))
        plt.plot(q1, q2_deg, 'b-')
        plt.xlabel(r'$\theta_1$ (°)')
        plt.ylabel(r'$\theta_2$ (°)')
        plt.title('Parametric curve')
        plt.grid()
        plt.tight_layout()
        plt.savefig('data/parametric_curve.png')
        plt.close()