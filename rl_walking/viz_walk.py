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

# Run visualization
state, _ = env.reset()
total_reward = 0
max_steps = 100  # Run for a fixed number of steps
episode_n = 10

for _ in range(episode_n):
    for t in range(max_steps):
        action = agent.get_action(state)
        # Scale action from [-1, 1] to [0, 1]
        scaled_action = (action + 1) / 2
        next_state, reward, done, _, _ = env.step(scaled_action)
        
        total_reward += reward
        state = next_state
        
        # Render environment
        env.unwrapped.mj_render()
        
        # if done:
        #     print(f"Episode terminated after {t+1} steps with total reward: {total_reward:.2f}")
        #     state, _ = env.reset()
        #     total_reward = 0
        #     break

    env.close()
print("Visualization completed")