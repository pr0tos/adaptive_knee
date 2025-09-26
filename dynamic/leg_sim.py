import time
import mujoco
import mujoco.viewer
import numpy as np
import utils
import matplotlib.pyplot as plt
from dataloader import DataLoader
import pandas as pd

m = mujoco.MjModel.from_xml_path('dynamic/leg_model.xml')
d = mujoco.MjData(m)
df = pd.read_csv('dynamic/data/1.csv', sep = ';', encoding='utf-16le')
print(df.head())
time_data, q1_data = np.linspace(0., 10., len(np.array(df['d:AngleThigh']))), np.array(df['d:AngleThigh'])
# plt.plot(q1_data)
# plt.show()
q1_check = []


def control(model, data):
    global current_time  # Используем global, чтобы получить доступ к current_time из основного цикла
    # Находим индекс времени, ближайший к текущему времени симуляции
    index = np.argmin(np.abs(time_data - current_time))

    
    # Используем значение q1 из данных
    q1 = q1_data[index]
    q1_check.append(q1)
    # Ограничиваем значение q1 в пределах допустимого диапазона
    # q1 = np.clip(q1, -0.27, 0.44)

    # Для q2 пока оставим синусоидальный сигнал, но вы можете заменить его данными, если они у вас есть
    q2 = np.deg2rad(45) * np.sin(2 * np.pi * 0.5 * current_time)

    d.ctrl = np.array([q1, q2])

# def lgger(model, data):

logdata_q = []
logdata_dq = []
logdata_ee_pos = []

with mujoco.viewer.launch_passive(m, d) as viewer:
    utils.look_at_zx(viewer)

    mujoco.set_mjcb_control(control)
    start = time.time()
    while viewer.is_running() and time.time() - start < 10: 
        current_time = time.time() - start

        mujoco.mj_step(m, d)

        logdata_q.append(d.qpos.copy())
        logdata_dq.append(d.qvel.copy())
        logdata_ee_pos.append(d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'ee')].copy())



        viewer.sync()

logdata_ee_pos = np.asarray(logdata_ee_pos)
plt.plot(q1_check)
plt.show()
# plt.subplot()
# plt.plot(logdata_ee_pos[:,0], logdata_ee_pos[:,2])
# plt.show()