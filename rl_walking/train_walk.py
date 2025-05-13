import wandb
from agents.sac import SAC
from myosuite.utils import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime

def train(agent, env, episode_n=20000, max_steps=1000, load_model_path=None, use_wandb=False):
    """
    Обучает агента в заданной среде.

    Args:
        agent: SAC.
        env: Gym 
        episode_n (int): Counts of episodes.
        max_steps (int): Max counts of steps in episode.
        load_model_path (str): Path to a pre-trained model to load (optional).
        use_wandb (bool): Whether to log to WandB.

    Returns:
        list: Max reward.
    """
    total_rewards = []
    running_reward = []
    best_avg_reward = -float('inf')
    global_step = 0  # Глобальный счётчик шагов

    # Ensure the models directory exists
    os.makedirs("rl_walking/models", exist_ok=True)

    # Load pre-trained model if provided
    if load_model_path and os.path.exists(load_model_path):
        print(f"Loading pre-trained model from {load_model_path}")
        agent.load_model(load_model_path)
    else:
        print("No pre-trained model found, training from scratch.")

    for episode in range(episode_n):
        total_reward = 0
        state, _ = env.reset()
        
        for t in range(max_steps):
            action = agent.get_action(state)
            # Scale action from [-1, 1] to [0, 1]
            scaled_action = (action + 1) / 2
            next_state, reward, done, _, _ = env.step(scaled_action)

            agent.fit(state, action, reward, done, next_state, use_wandb=use_wandb)
        
            total_reward += reward
            state = next_state
            
            global_step += 1
            
            if done and episode > 1000:  # Allow exploration for first 50 episodes
                break
        
        total_rewards.append(total_reward)
        running_reward.append(total_reward)
        if len(running_reward) > 10:
            running_reward.pop(0)
        
        # Compute average reward
        avg_reward = np.mean(running_reward)
        
        # Log metrics to WandB if enabled
        if use_wandb:
            wandb.log({
                "episode": episode,
                "total_reward": total_reward,
                "avg_reward": avg_reward,
                "steps_per_episode": t + 1,
                "alpha": agent.alpha.item()  # Log alpha
            })
        
        # Save model if average reward improves
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save_model("rl_walking/models/sac_policy.pth")
            print(f"Episode {episode}: Saved model with avg_reward = {avg_reward:.2f}")
        
        # Printing every 100 episodes
        if episode % 100 == 0 or episode == 0:
            print(f'Episode {episode}: Total Reward = {total_reward:.2f}, Avg Reward = {avg_reward:.2f}, Steps = {t + 1}')

    return total_rewards

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train SAC agent on myoLegWalk-v0")
    parser.add_argument('--use_wandb', action='store_true', help='Enable WandB logging')
    args = parser.parse_args()

    # Get current date and time for run name (format: YYYY-MM-DD_HH-MM)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Initialize WandB if enabled
    if args.use_wandb:
        wandb.init(
            project="myoLegWalk-training",
            name=f"train_{current_time}",  # Name with date and time
            config={
                "episode_n": 20000,
                "max_steps": 1000,
                "gamma": 0.99,
                "alpha": 0.7,
                "tau": 0.005,
                "batch_size": 256,
                "pi_lr": 5e-5,
                "q_lr": 5e-5
            }
        )
    else:
        print("WandB logging disabled.")

    # Initialize environment
    env = gym.make('myoLegWalk-v0', normalize_act=False)
    print(f'Environment name: {env.unwrapped.spec.id}')
    state_dim = env.observation_space.shape[0]
    print(f'Observation space: {state_dim}')
    action_dim = env.action_space.shape[0]
    print(f'Action space: {action_dim}')

    # Initialize SAC agent
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        alpha=0.7,
        tau=0.005,
        batch_size=256,
        pi_lr=5e-5,
        q_lr=5e-5
    )

    # Train the agent with option to load a pre-trained model
    total_rewards = train(agent, env, use_wandb=args.use_wandb)

    # Close environment
    env.close()

    # Plot rewards
    plt.plot(total_rewards)
    plt.title('Total Rewards')
    plt.grid()
    plt.show()

    # Finish WandB run if enabled
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()