import gym
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.from_numpy(state).float().unsqueeze(0)
        q_values = self.model(state)
        return np.argmax(q_values.detach().numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.from_numpy(state).float().unsqueeze(0)
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)
            target = reward
            if not done:
                target = (reward + self.gamma * torch.max(self.model(next_state).detach()))
            q_values = self.model(state)
            q_values[0][action] = target
            loss = F.mse_loss(q_values, self.model(state))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
env_name = 'MountainCar-v0'
episodes = 10000
env = gym.make(env_name)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
batch_size = 32


reward_list=[]
timestep_list=[]
for e in range(episodes):
    state,_ = env.reset()
    # print(state)
    done = False
    time = 0
    rewards=0
    
    while not done:
        action = agent.act(state)
        # print(env.step(action))
        next_state, reward, done,terminated, _ = env.step(action)
        reward = state[0] + 0.5
        
       
# Adjust reward for task completion
        if state[0] >= 0.5:
          reward += 1
        rewards+=reward
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        time += 1

        if done or terminated:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, episodes, time, agent.epsilon))
            break
    

        agent.replay(batch_size)
    reward_list.append(rewards)
    timestep_list.append(time)
import matplotlib.pyplot as plt 
plt.figure()
plt.plot(reward_list)
plt.savefig("rewards")
plt.figure()
plt.plot(timestep_list)
plt.savefig("Time_steps")
agent.save(filename="model_mountain_car.pth")