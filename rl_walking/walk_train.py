import wandb
from sac import SAC
from myosuite.utils import gym
import numpy as np
import matplotlib.pyplot as plt
import os

def train(agent, env, episode_n=10000, max_steps=1000):
    """
    Обучает агента в заданной среде.

    Args:
        agent: SAC.
        env: Gym 
        episode_n (int): Counts of episodes.
        max_steps (int): Max counts of steps in episode.

    Returns:
        list: Max reward.
    """
    total_rewards = []
    running_reward = []
    best_avg_reward = -float('inf')

    # Ensure the models directory exists
    os.makedirs("rl_walking/models", exist_ok=True)

    for episode in range(episode_n):
        total_reward = 0
        state, _ = env.reset()
        
        for t in range(max_steps):
            action = agent.get_action(state)
            # Scale action from [-1, 1] to [0, 1]
            scaled_action = (action + 1) / 2
            next_state, reward, done, _, _ = env.step(scaled_action)
        
            agent.fit(state, action, reward, done, next_state)
        
            total_reward += reward
            state = next_state
            
            # Optional rendering for debugging (uncomment to enable)
            # env.unwrapped.mj_render()
            
            if done:
                break
        
        total_rewards.append(total_reward)
        running_reward.append(total_reward)
        if len(running_reward) > 10:
            running_reward.pop(0)
        
        # Compute average reward
        avg_reward = np.mean(running_reward)
        
        # Log metrics to WandB
        wandb.log({
            "episode": episode,
            "total_reward": total_reward,
            "avg_reward": avg_reward
        })
        
        # Save model if average reward improves
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save_model("rl_walking/models/sac_policy.pth")
            print(f"Episode {episode}: Saved model with avg_reward = {avg_reward:.2f}")
        
        print(f'Episode {episode}: Total Reward = {total_reward}')

    return total_rewards

def main():
    # Initialize WandB
    wandb.init(project="myoLegWalk-training", config={
        "episode_n": 10000,
        "max_steps": 1000,
        "gamma": 0.99,
        "alpha": 0.2,
        "tau": 0.005,
        "batch_size": 256,
        "pi_lr": 3e-4,
        "q_lr": 3e-4
    })

    # Initialize environment
    env = gym.make('myoLegWalk-v0')
    print(f'Environment name: {env.unwrapped.spec.id}')
    state_dim = env.observation_space.shape[0]
    print(f'Observation space: {env.observation_space.shape[0]}')
    action_dim = env.action_space.shape[0]
    print(f'Action space: {env.action_space.shape[0]}')

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


    # Set up visualization (make skin transparent)
    geom_1_indices = np.where(env.unwrapped.sim.model.geom_group == 1)
    env.unwrapped.sim.model.geom_rgba[geom_1_indices, 3] = 0

    # Train the agent
    total_rewards = train(agent, env)

    # Close environment
    env.close()

    # Plot rewards
    plt.plot(total_rewards)
    plt.title('Total Rewards')
    plt.grid()
    plt.show()

    # Finish WandB run
    wandb.finish()

if __name__ == "__main__":
    main()