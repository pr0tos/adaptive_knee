from scipy.interpolate import CubicSpline
from system_par import SystemParameters
import numpy as np
from typing import List 


class MotionSystem:
    def __init__(self, params: SystemParameters, time_data: np.ndarray, q1_data: np.ndarray):
        self.params = params
        self.time_data = time_data
        self.q1_data = q1_data
        self._friction_velocity_eps = 1e-3
        
        # Создание интерполяционных функций
        self.q1_func = CubicSpline(time_data, q1_data)
        self.d_q1_func = self.q1_func.derivative()
        self.dd_q1_func = self.q1_func.derivative(2)

    def _compute_joint_friction(self, lever: float, d_q2: float) -> float:
        """Сухое трение, масштабируемое геометрическим плечом механизма."""
        friction_direction = np.tanh(d_q2 / self._friction_velocity_eps)
        return self.params.d * lever * friction_direction
        
    def system_equations(self, t: float, y: np.ndarray) -> List[float]:
        q2, d_q2 = y  # Распаковываем переменные

        # Получаем q1 и его производные, преобразуем в радианы
        q1 = np.deg2rad(self.q1_func(t))
        d_q1 = np.deg2rad(self.d_q1_func(t))
        dd_q1 = np.deg2rad(self.dd_q1_func(t)) 

        radicand = (2 * self.params.r**2 * np.sin(d_q2)**2 - self.params.d**2 +
                    self.params.d * self.params.r * np.cos(d_q2) - self.params.r**2 + self.params.L**2)
        radicand = max(radicand, 0.0)
        lever = self.params.r * np.cos(d_q2) + np.sqrt(radicand)

        friction_torque = self._compute_joint_friction(lever, d_q2)
        Q2 = -2 * lever * d_q2 - friction_torque
        
        A22 = self.params.I2 + self.params.m2 * self.params.r2**2
        B2 = (Q2 - self.params.m2 * self.params.g * self.params.r2 * np.sin(q2) + 
              self.params.m2 * self.params.l1 * self.params.r2 * 
              (d_q1**2 * np.sin(q1 - q2) - dd_q1 * np.cos(q1 - q2)))
        alpha2 = B2 / A22
        
        return [d_q2, alpha2]
