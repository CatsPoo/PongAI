import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from .ReplayBuffer import ReplayBuffer
from.NN import NN


class DQN:
    def __init__(self, state_dim, action_dim, buffer_size=10000, batch_size=64, gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9999, target_update=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_net = NN(state_dim, action_dim).to(self.device)
        self.target_net = NN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.step_count = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            print('Not enought action played to train the agent')
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Convert any array to tensor format
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)

        # Target: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
