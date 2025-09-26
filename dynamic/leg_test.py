import time as time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

try:
    import utils
except Exception:
    utils = None

# -------------------- Parameters --------------------
MODEL_PATH = "dynamic/leg_model.xml"
CSV_PATH   = "dynamic/data/1.csv"
CSV_COL    = "d:AngleThigh"
SIM_T      = 10.0       # total simulation time [s]
REALTIME   = True       # synchronize simulation with wall-clock time

# -------------------- Model --------------------
m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)

# Joint/actuator indices
q1_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "q1")
q1_qposadr = m.jnt_qposadr[q1_jid]  # index of q1 in d.qpos

# -------------------- Data (CSV) --------------------
df = pd.read_csv(CSV_PATH, sep=';', encoding='utf-16le')
if CSV_COL not in df.columns:
    raise KeyError(f"CSV column '{CSV_COL}' not found. Available: {list(df.columns)}")

q1_data_deg = np.asarray(df[CSV_COL], dtype=float)
q1_data     = np.deg2rad(q1_data_deg)               # reference trajectory in radians
time_data   = np.linspace(0.0, SIM_T, len(q1_data)) # time axis for CSV

# -------------------- Logs --------------------
q1_check = []        # what was fed into the controller (rad)
log_q, log_dq = [], []
log_ee_pos    = []
log_t         = []

# -------------------- Control callback --------------------
def control(model, data):
    t = data.time
    # nearest index on the uniform reference time grid
    idx = int(np.clip(round(t / SIM_T * (len(time_data) - 1)), 0, len(time_data) - 1))
    q1  = q1_data[idx]
    data.ctrl[0] = q1  # position actuator for the hip joint
    q1_check.append(q1)

# -------------------- Simulation loop --------------------
with mujoco.viewer.launch_passive(m, d) as viewer:
    if utils and hasattr(utils, "look_at_zx"):
        try:
            utils.look_at_zx(viewer)
        except Exception:
            pass

    mujoco.set_mjcb_control(control)

    try:
        ee_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee")
        has_ee = True
    except Exception:
        ee_id = None
        has_ee = False

    t0_wall = time.time()
    while viewer.is_running() and d.time < SIM_T:
        mujoco.mj_step(m, d)

        # Logging
        log_q.append(d.qpos.copy())
        log_dq.append(d.qvel.copy())
        if has_ee:
            log_ee_pos.append(d.site_xpos[ee_id].copy())
        log_t.append(d.time)

        viewer.sync()

        if REALTIME:
            # real-time synchronization
            t_target = t0_wall + d.time
            dt_sleep = t_target - time.time()
            if dt_sleep > 0:
                time.sleep(dt_sleep)

# -------------------- Postprocessing --------------------
log_q      = np.asarray(log_q) if log_q else np.zeros((0, m.nq))
log_t      = np.asarray(log_t)

if len(log_t) == 0:
    raise RuntimeError("No simulation data (log_t is empty). Check if the loop started or if the viewer closed immediately.")

# actual hip joint angle from simulation (rad)
q1_fact = log_q[:, q1_qposadr]

# reference interpolated to the simulation time grid (rad)
q1_ref_at_logt = np.interp(log_t, time_data, q1_data)

# error (rad)
err  = q1_fact - q1_ref_at_logt
rmse = float(np.sqrt(np.mean(err**2)))
mae  = float(np.mean(np.abs(err)))
merr = float(np.max(np.abs(err)))

print(f"[q1 vs CSV] RMSE = {rmse:.6f} rad, MAE = {mae:.6f} rad, Max|err| = {merr:.6f} rad")

# ---- Plots in radians ----
plt.figure()
plt.plot(log_t, q1_ref_at_logt, label="q1 reference (rad)")
plt.plot(log_t, q1_fact,        label="q1 actual (rad)", linestyle="--")
plt.xlabel("t, s")
plt.ylabel("rad")
plt.title("Comparison q1: reference (CSV) vs actual (simulation)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(log_t, err)
plt.xlabel("t, s")
plt.ylabel("rad")
plt.title("Error q1 = actual - reference (rad)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
