# Adaptive Knee Dynamics Simulation

## Project Setup

### 1.  Create and Activate Conda Environment

```bash
# Create conda environment from YAML file
conda env create -f environment.yml

# Activate environment
conda activate knee
```

### 2.  Configuration
All simulation parameters are set in 'config.yml':
```yaml
dynamics:
  initial_conditions:
    q1: 0.0       # Initial angle of joint 1 [rad]
    q2: 0.5       # Initial angle of joint 2 [rad]
    q1_dot: 0.0   # Initial angular velocity of joint 1 [rad/s]
    q2_dot: 0.0   # Initial angular velocity of joint 2 [rad/s]
  parameters:
    m1: 1.0       # Mass of link 1 [kg]
    m2: 1.5       # Mass of link 2 [kg]
    l1: 0.5       # Length of link 1 [m]
    l2: 0.5       # Length of link 2 [m]
    g: 9.81       # Gravitational acceleration [m/s²]
```
### 3.  Running the Simulation
Inside \bold{dynamic}
```bash
python main.py
```
