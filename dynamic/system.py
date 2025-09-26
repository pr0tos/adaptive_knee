from scipy.interpolate import CubicSpline
from system_par import SystemParameters
import numpy as np
from typing import List 


class MotionSystem:
    def __init__(self, params: SystemParameters, time_data: np.ndarray, q1_data: np.ndarray):
        self.params = params
        self.time_data = time_data
        self.q1_data = q1_data
        
        # Создание интерполяционных функций
        self.q1_func = CubicSpline(time_data, q1_data)
        self.d_q1_func = self.q1_func.derivative()
        self.dd_q1_func = self.q1_func.derivative(2)
        
    def system_equations(self, t: float, y: np.ndarray) -> List[float]:
        q2, d_q2 = y  # Распаковываем переменные

        # Получаем q1 и его производные (уже в радианах, так как данные в градусах преобразованы)
        q1 = np.deg2rad(self.q1_func(t))
        d_q1 = np.deg2rad(self.d_q1_func(t))
        dd_q1 = np.deg2rad(self.dd_q1_func(t)) 
        
        # Параметры системы
        params = self.params
        
        # Расчет управляющего воздействия (пример, может потребовать корректировки)
        Q2 = -params.d * d_q2  # Простое демпфирование
        
        # Уравнения движения для двойного маятника
        # Матрица инерции
        c = np.cos(q1 - q2)
        s = np.sin(q1 - q2)
        
        M11 = params.I1 + params.I2 + params.m1*params.r1**2 + params.m2*(params.l1**2 + params.r2**2 + 2*params.l1*params.r2*c)
        M12 = params.I2 + params.m2*params.r2*(params.r2 + params.l1*c)
        M21 = M12
        M22 = params.I2 + params.m2*params.r2**2
        
        # Вектор кориолисовых и центробежных сил
        C1 = -params.m2*params.l1*params.r2*s*(d_q2**2) + params.m2*params.l1*params.r2*s*(d_q1**2)
        C2 = params.m2*params.l1*params.r2*s*(d_q1**2)
        
        # Вектор гравитации
        G1 = -(params.m1*params.r1 + params.m2*params.l1)*params.g*np.sin(q1) - params.m2*params.r2*params.g*np.sin(q1 + q2)
        G2 = -params.m2*params.r2*params.g*np.sin(q1 + q2)
        
        # Решение системы уравнений
        # M * [ddq1, ddq2]^T = [Q1, Q2]^T - C - G
        
        det_M = M11*M22 - M12*M21
        if abs(det_M) > 1e-10:
            dd_q2 = (M11*(Q2 - C2 - G2) - M21*(0 - C1 - G1)) / det_M
        else:
            dd_q2 = 0
        
        return [d_q2, dd_q2]