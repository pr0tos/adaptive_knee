import torch
from agents.sac import SAC
from myosuite.utils import gym
import numpy as np

# Initialize environment
env = gym.make('myoLegWalk-v0', normalize_act=False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Initialize SAC agent
agent = SAC(
    state_dim=state_dim,
    action_dim=action_dim,
    gamma=0.99,
    alpha=0.2,
    tau=0.005,
    batch_size=256,
    pi_lr=3e-4,
    q_lr=3e-4
)

# Load saved model
agent.load_model("rl_walking/models/sac_policy.pth")
print("Loaded policy from sac_policy.pth")

# Run visualization for multiple episodes without rendering
max_steps = 100  # Steps per episode
episode_n = 10   # Number of episodes

for episode in range(episode_n):
    state, _ = env.reset()
    total_reward = 0
    
    print(f"\nStarting Episode {episode}")
    
    for t in range(max_steps):
        action = agent.get_action(state)
        # Scale action from [-1, 1] to [0, 1]
        scaled_action = (action + 1) / 2
        next_state, reward, done, _, info = env.step(scaled_action)
        
        total_reward += reward
        state = next_state
        
        # Log key metrics for analysis
        height = info['obs_dict']['height']
        com_vel = info['obs_dict']['com_vel'][0]
        print(f"Step {t}: Reward = {reward:.2f}, Height = {height:.4f}, COM Vel = {com_vel:.4f}, Fell = {1 if height < 0.3 else 0}")
        env.unwrapped.mj_render()
        if done:
            print(f"Episode {episode} terminated after {t+1} steps with total reward: {total_reward:.2f}")
            break

# Close environment
env.close()
print("Visualization completed")