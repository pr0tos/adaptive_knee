import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

class Visualizer:
    @staticmethod
    def plot_results(t: np.ndarray, q1_func: Callable, d_q1_func: Callable, 
                     q2: np.ndarray, d_q2: np.ndarray):
        # Вычисляем q1 и d_q1 для графиков
        q1 = np.deg2rad(q1_func(t))
        d_q1 = d_q1_func(t)

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, q1, label=r'$\theta_1$ (рад) - заданный сигнал')
        plt.plot(t, q2, label=r'$\theta_2$ (рад)')
        plt.xlabel('Время (с)')
        plt.ylabel('Угол (рад)')
        plt.legend()
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(t, d_q1, label=r'$\omega_1$ (рад/с) - заданный сигнал')
        plt.plot(t, d_q2, label=r'$\omega_2$ (рад/с)')
        plt.xlabel('Время (с)')
        plt.ylabel('Угловая скорость (рад/с)')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()