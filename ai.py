# AI for self driving car
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# architecture of the neural network
class Network(nn.Module):
    def __init__(self, input_size,nb_action):
        # input_size = 5(3 sensor,2 orientations), nb_action=3(l,s,r)
        super(Network, self).__init__()
        self.input_size= input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(in_features=input_size,out_features=30)
        self.fc2 = nn.Linear(in_features=30,out_features=nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Experience Replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    def sample(self,batch_size):
        # format- state,action,reward
        #list = [[1,2,3],[4,5,6]] then zip(*list) = [[1,4],[2,5],[3,6]]
        samples = zip(*random.sample(self.memory,batch_size))
        return map(lambda x: Variable(torch.cat(x,0)),samples)

# Deep Q Learning
class Dqn():
    def __init__(self,input_size,nb_action,gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size,nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
         # Softmax ([1,2,3]) = [0.04,0.11,0.85] => softmax([1,2,3]*3) = [0,0,02,0.98]
        action = probs.multinomial(num_samples=1)
        return action.data[0,0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad() # reinitializr optimizer at each loop
        td_loss.backward(retain_graph = True)
        self.optimizer.step() #updates the weight
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.0)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
