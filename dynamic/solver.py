import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, List
from system import MotionSystem

class Solver:
    @staticmethod
    def solve(system: MotionSystem, t_span: Tuple[float, float], y0: List[float], 
              t_eval: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sol = solve_ivp(
            fun=system.system_equations,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45'
        )
        return sol.t, sol.y[0], sol.y[1]