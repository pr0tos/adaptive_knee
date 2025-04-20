import numpy as np
from dataloader import DataLoader
from system import MotionSystem
from solver import Solver 
from system_par import SystemParameters
from visualizer import Visualizer
import yaml

def load_config(config_path: str) -> SystemParameters:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Конвертация углов из градусов в радианы
    config['q2_min'] = np.deg2rad(config['q2_min'])
    config['q2_max'] = np.deg2rad(config['q2_max'])
    config['q2_0'] = np.deg2rad(config['q2_0'])

    return SystemParameters(**config)

def main():
    # Загрузка конфигурации
    config = load_config('config.yaml')
    
    # Загрузка данных
    time_data, q1_data = DataLoader.load_data('data/motion.csv')
    
    # Создание системы
    system = MotionSystem(config, time_data, q1_data)
    
    # Начальные условия
    y0 = [config.q2_0, config.d_q2_0]
    
    # Временной интервал
    t_max = max(time_data)
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, 1000)
    
    # Решение системы
    t, q2, d_q2 = Solver.solve(system, t_span, y0, t_eval)
    
    # Визуализация
    Visualizer.plot_results(t, system.q1_func, system.d_q1_func, q2, d_q2)

if __name__ == "__main__":
    main()