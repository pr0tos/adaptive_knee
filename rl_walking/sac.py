import torch
import torch.nn as nn
from copy import deepcopy
import random
from torch.distributions import Normal
import numpy as np
from collections import deque
import wandb

class SAC(nn.Module):
    def __init__(self, state_dim, action_dim, gamma=0.99, alpha=0.2, tau=1e-2,
                 batch_size=256, pi_lr=3e-4, q_lr=3e-4):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy Network
        self.pi_model = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2 * action_dim)
        ).to(self.device)

        # Q Networks
        self.q1_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)

        self.q2_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)

        # Target Networks
        self.q1_target_model = deepcopy(self.q1_model).to(self.device)
        self.q2_target_model = deepcopy(self.q2_model).to(self.device)

        # Other parameters
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size
        self.memory = deque(maxlen=100000)  # Fixed-size replay buffer

        # Optimizers
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1_model.parameters(), lr=q_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2_model.parameters(), lr=q_lr)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.predict_actions(state)
        return (action.detach().cpu().numpy().flatten() + 1) / 2

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        
        # Transform data
        states, actions, rewards, dones, next_states = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)

        # Train Q-networks
        with torch.no_grad():
            next_actions, next_log_probs = self.predict_actions(next_states)
            next_states_actions = torch.cat((next_states, next_actions), dim=1)
            next_q1 = self.q1_target_model(next_states_actions)
            next_q2 = self.q2_target_model(next_states_actions)
            next_min_q = torch.min(next_q1, next_q2)
            targets = rewards + self.gamma * (1 - dones) * (next_min_q - self.alpha * next_log_probs)

        states_actions = torch.cat((states, actions), dim=1)
        q1_values = self.q1_model(states_actions)
        q2_values = self.q2_model(states_actions)

        q1_loss = ((q1_values - targets.detach()) ** 2).mean()
        q2_loss = ((q2_values - targets.detach()) ** 2).mean()

        self.update_model(q1_loss, self.q1_optimizer, self.q1_model, self.q1_target_model)
        self.update_model(q2_loss, self.q2_optimizer, self.q2_model, self.q2_target_model)

        # Train policy
        pred_actions, log_probs = self.predict_actions(states)
        states_pred_actions = torch.cat((states, pred_actions), dim=1)
        q1_pred = self.q1_model(states_pred_actions)
        q2_pred = self.q2_model(states_pred_actions)
        min_q_pred = torch.min(q1_pred, q2_pred)
        pi_loss = -torch.mean(min_q_pred - self.alpha * log_probs)

        self.update_model(pi_loss, self.pi_optimizer)

        # Log losses to WandB
        wandb.log({
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "pi_loss": pi_loss.item()
        })

    def update_model(self, loss, optimizer, model=None, target_model=None):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Soft update for target networks
        if model is not None and target_model is not None:
            for param, target_param in zip(model.parameters(), target_model.parameters()):
                target_param.data.copy_(
                    (1 - self.tau) * target_param.data + self.tau * param.data
                )

    def predict_actions(self, states):
        states = states.to(self.device)
        outputs = self.pi_model(states)
        means, log_stds = torch.chunk(outputs, 2, dim=-1)

        log_stds = torch.clamp(log_stds, -20, 2)
        stds = torch.exp(log_stds)

        dists = Normal(means, stds)
        actions = dists.rsample()
        log_probs = dists.log_prob(actions)

        clipped_actions = torch.tanh(actions)
        log_probs -= torch.log(1 - clipped_actions.pow(2) + 1e-6).sum(dim=1, keepdim=True)

        return clipped_actions, log_probs

    def save_model(self, path):
        torch.save({
            'pi_model_state_dict': self.pi_model.state_dict(),
            'q1_model_state_dict': self.q1_model.state_dict(),
            'q2_model_state_dict': self.q2_model.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.pi_model.load_state_dict(checkpoint['pi_model_state_dict'])
        self.q1_model.load_state_dict(checkpoint['q1_model_state_dict'])
        self.q2_model.load_state_dict(checkpoint['q2_model_state_dict'])
        self.pi_model.eval()
        self.q1_model.eval()
        self.q2_model.eval()