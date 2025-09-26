import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import mujoco
import mujoco.viewer

class Visualizer:
    @staticmethod
    def plot_results(t: np.ndarray, 
                    q1_func: Callable, 
                    d_q1_func: Callable, 
                    q2: np.ndarray, 
                    d_q2: np.ndarray):
        """Старый метод для построения графиков"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Углы
        axes[0, 0].plot(t, np.rad2deg(q1_func(t)), label='q1 (бедро)')
        axes[0, 0].plot(t[:len(q2)], np.rad2deg(q2), label='q2 (колено)')
        axes[0, 0].set_xlabel('Время, с')
        axes[0, 0].set_ylabel('Угол, град')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_title('Угловые положения')
        
        # Угловые скорости
        axes[0, 1].plot(t, np.rad2deg(d_q1_func(t)), label='dq1/dt')
        axes[0, 1].plot(t[:len(d_q2)], np.rad2deg(d_q2), label='dq2/dt')
        axes[0, 1].set_xlabel('Время, с')
        axes[0, 1].set_ylabel('Угловая скорость, град/с')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_title('Угловые скорости')
        
        # Фазовый портрет колена
        axes[1, 0].plot(np.rad2deg(q2), np.rad2deg(d_q2))
        axes[1, 0].set_xlabel('Угол колена, град')
        axes[1, 0].set_ylabel('Угловая скорость колена, град/с')
        axes[1, 0].grid(True)
        axes[1, 0].set_title('Фазовый портрет коленного сустава')
        
        # Траектория кончика стопы
        l1, l2 = 0.4, 0.3  # длины звеньев
        x_foot = l1 * np.cos(q1_func(t[:len(q2)])) + l2 * np.cos(q1_func(t[:len(q2)]) + q2)
        y_foot = l1 * np.sin(q1_func(t[:len(q2)])) + l2 * np.sin(q1_func(t[:len(q2)]) + q2)
        axes[1, 1].plot(x_foot, y_foot)
        axes[1, 1].set_xlabel('X, м')
        axes[1, 1].set_ylabel('Y, м')
        axes[1, 1].grid(True)
        axes[1, 1].set_title('Траектория стопы')
        axes[1, 1].set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_mujoco(t: np.ndarray, 
                        q1_func: Callable, 
                        q2_data: np.ndarray,
                        model_path: str = 'model.xml'):
        """Визуализация в MuJoCo"""
        # Загрузка модели
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        # Создание визуализатора
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Частота обновления
            fps = 30
            frame_interval = int(1.0 / (fps * model.opt.timestep))
            
            for i in range(0, len(t), frame_interval):
                if i >= len(q2_data):
                    break
                    
                # Получаем значения углов
                q1_val = q1_func(t[i])  # Уже в радианах
                q2_val = q2_data[i]     # Уже в радианах
                
                # Установка позиций суставов
                data.qpos[0] = q1_val  # бедро
                data.qpos[1] = q2_val  # колено
                
                # Обновление симуляции
                mujoco.mj_forward(model, data)
                
                # Обновление визуализации
                viewer.sync()
                
                # Небольшая задержка для плавности
                import time
                time.sleep(model.opt.timestep * frame_interval)