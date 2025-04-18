import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('dynamic/motion.csv', header=None, sep=';', decimal=',')
time_data = data.iloc[:, 0].values
q1_data = data.iloc[:, 1].values

# Удаление дубликатов
time_data, unique_indices = np.unique(time_data, return_index=True)
q1_data = q1_data[unique_indices]

# Сглаживание (без уменьшения длины)
q1_data = uniform_filter1d(q1_data, size=20, mode='nearest')

# Интерполяция
q1_func = CubicSpline(time_data, q1_data)
d_q1_func = q1_func.derivative()
# Параметры системы
I1 = 0.25  # Момент инерции бедра (кг·м²)
I2 = 0.1   # Момент инерции протеза (кг·м²)
m1 = 10.0  # Масса бедра (кг)
m2 = 2.0   # Масса протеза (кг)
r1 = 0.3   # Расстояние от оси вращения до центра масс бедра (м)
r2 = 0.2   # Расстояние от оси вращения до центра масс протеза (м)
l1 = 0.4   # Длина бедра (м)
g = 9.81   # Ускорение свободного падения (м/с²)
Q1 = 0.0   # Обобщенная сила для бедра
Q2 = 0.0   # Обобщенная сила для протеза

# Параметры поршня
r = 0.05   # Радиус поршня (м)
d = 0.02   # Расстояние (м)
L = 0.1    # Длина (м)



# Модифицированная функция системы (теперь y содержит только q2, d_q2)
def system(t, y):
    q2, d_q2 = y  # Распаковываем переменные
    
    # Получаем q1 и его производную из известных функций
    q1 = q1_func(t)
    d_q1 = d_q1_func(t)
    
    c = r * np.cos(d_q2) + np.sqrt(2 * r**2 * np.sin(d_q2)**2 - d**2 + d * r * np.cos(d_q2) - r**2 + L**2) #dh
    Q2 = - c**2 * d_q2
    
    # Уравнения для угловых ускорений
    # Теперь матрица A - это просто скаляр (поскольку q1 известно)
    A22 = I2 + m2*r2**2
    B2 = Q2 - m2*g*r2*np.sin(q2) + m2*l1*r2*np.sin(q1 - q2)*d_q1**2 
    
    # Решаем уравнение для углового ускорения alpha2
    alpha2 = B2 / A22
    
    return [d_q2, alpha2]

# Начальные условия (только для q2 и d_q2 теперь)
q2_0 = 0  # Начальный угол второго тела (0 градусов)
d_q2_0 = 0.0  # Начальная угловая скорость второго тела
y0 = [q2_0, d_q2_0]

# Временной интервал
t_max = max(time_data)
t_span = (0, t_max)  # От 0 до 10 секунд
t_eval = np.linspace(0, t_max, 1000)  # Точки для вывода решения

# Решение системы
sol = solve_ivp(system, t_span, y0, t_eval=t_eval, method='RK45')

# Результаты
q2 = sol.y[0]  # Угол второго тела
d_q2 = sol.y[1]  # Угловая скорость второго тела
t = sol.t  # Время

# Вычисляем q1 и d_q1 для графиков
q1 = q1_func(t)
d_q1 = d_q1_func(t)

# Построение графиков
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