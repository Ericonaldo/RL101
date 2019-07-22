#!/Users/mhliu/Program/anaconda3/envs/rl/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
import gym
import numpy as np
import collections
import random
import matplotlib
import matplotlib.pyplot as plt
from math import *

ENV = "Pendulum-v0"

MEMORY_SIZE = 10000
EPISODES = 250
MAX_STEP = 200
GAMMA = 0.9
BATCH_SIZE = 64
LR_A = 0.001    # learning rate for actor
LR_C = 0.002     # learning rate for critic

TAU = 0.01 # soft replacement
sigma = 3
mu = 0

env = gym.make(ENV)
env.seed(1)
action_bound = [env.action_space.low, env.action_space.high] 
criterion = nn.MSELoss()

class Actor(nn.Module):
    def __init__(self, env, hiddens):
        super(Actor, self).__init__()
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.hiddens = hiddens[0]
        self.fc1 = nn.Linear(self.state_dim, self.hiddens)
        self.fc2 = nn.Linear(self.hiddens, self.action_dim)

    def forward(self, inputs):
        out = inputs
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.tanh(out)
        out = out * 2
        # out = torch.clamp(out, float(action_bound[0]), float(action_bound[1])) 
        return out

class Critic(nn.Module):
    def __init__(self, env, hiddens):
        super(Critic, self).__init__()
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.hiddens = hiddens[0]
        self.fc1 = nn.Linear(self.state_dim, self.hiddens)
        self.fc2 = nn.Linear(self.action_dim, self.hiddens)
        self.fc3 = nn.Linear(self.hiddens, 1)

    def forward(self, input_s, input_a):
        out = self.fc1(input_s)
        out1 = torch.relu(out)
        out = self.fc2(input_a)
        out2 = torch.relu(out)
        out = out1+out2
        out = self.fc3(out)
        #print(out)
        return out


class DDPG(object):
    def __init__(self, env, hiddens):
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.hiddens = hiddens[0]
           
        # policy network
        self.eval_a = Actor(env, hiddens)
        self.target_a = Actor(env, hiddens)
        self.optim_a = Adam(self.eval_a.parameters(), lr = LR_A) 
        # q_network
        self.eval_q = Critic(env, hiddens)
        self.target_q = Critic(env, hiddens)
        self.optim_q = Adam(self.eval_q.parameters(), lr = LR_C)
        hard_update(self.target_a, self.eval_a) 
        hard_update(self.target_q, self.eval_q) 
        
        self.is_training = True
        
    def train(self, state, next_state, action, reward, done):
        # critic update
        next_q = self.target_q(
                to_tensor(next_state, volatile=True),
                self.target_a(to_tensor(next_state, volatile=True))
                )
        next_q.volatile=False
        #print("action: ", action)
        #print("q(s,a): ", self.eval_q(to_tensor(state), to_tensor(action)))
        q_t = to_tensor(reward) + GAMMA * to_tensor(done) * next_q.squeeze()
        self.eval_q.zero_grad()
        q = self.eval_q(to_tensor(state), to_tensor(action)).squeeze()
        self.closs =  criterion(q, q_t)
        self.closs.backward()
        #print("closs: ", self.closs.data)
        # print(self.eval_q.parameters)
        self.optim_q.step()

        # actor update
        self.eval_a.zero_grad()
        self.aloss = -self.eval_q(
                to_tensor(state), 
                self.eval_a(to_tensor(state))
                )
        self.aloss = self.aloss.mean()
        #print("state: ", state)
        #print("action: ", self.eval_a(to_tensor(state)))
        #print("q(s,a): ", -self.eval_q(to_tensor(state), self.eval_a(to_tensor(state))))
        self.aloss.backward()
        #print("aloss: ", self.aloss.data)
        self.optim_a.step()
        
        # Target update
        soft_update(self.target_a, self.eval_a, TAU)
        soft_update(self.target_q, self.eval_q, TAU)
            
    def choose_action(self, current_state):
        current_state = current_state[np.newaxis, :]
        current_state = to_tensor(current_state)
        action = to_numpy(self.eval_a(current_state)).squeeze(0)

        return action
    
def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=torch.FloatTensor):
     
     return Variable(
             torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad).type(dtype)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau
         )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data) 

def to_numpy(var):
    return var.cpu().data.numpy() 
memory=[]
Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

ep = []
re = []
if __name__ == "__main__":
    ddpg = DDPG(env, [30])

    running_reward=0
    for episode in range(EPISODES):
        state = env.reset()
        reward_all = 0
        for step in range(MAX_STEP):
            # env.render()
                        
            action = ddpg.choose_action(state)              
            # action = np.clip(np.random.normal(action, sigma), action_bound[0], action_bound[1])    # add randomness to action selection for exploration
            action = np.clip(action + np.random.normal(mu, sigma), action_bound[0],action_bound[1])
            next_state, reward, done , _ = env.step(action)
            # print(reward)
            reward_all += reward
            
            memory.append(Transition(state, action, reward/10, next_state, float(done)))
            
            if len(memory) > 10000: # BATCH_SIZE * 4:
                sigma *= .99995
                batch_trasition = random.sample(memory, BATCH_SIZE)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = map(np.array, zip(*batch_trasition))
                ddpg.train(state = batch_state, next_state = batch_next_state, action = batch_action, reward = batch_reward, done=batch_done)
            if done:
                break
            state = next_state
            
        #running_reward = running_reward*0.99 + 0.01*reward_all
        #print("episode = {} reward = {}".format(episode, running_reward))
        print("episode = {} reward = {}".format(episode, reward_all))
        ep.append(episode)
        re.append(reward_all)
    env.close()
    plt.plot(ep, re)
    plt.show()
