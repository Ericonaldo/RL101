import tensorflow as tf
import gym
import numpy as np
import collections
import random
import tensorflow.contrib.layers as layers
import matplotlib
from math import *

ENV = "Pendulum-v0"

MEMORY_SIZE = 10000
EPISODES = 1000
MAX_STEP = 200
BATCH_SIZE = 32
UPDATE_PERIOD = 200

env = gym.make(ENV)
action_bound = [env.action_space.low, env.action_space.high]

def normal(x, mu, sigma):
    a = tf.exp(-1*(x-mu)**2/(2*sigma*sigma))
    b = 1/(sigma*tf.sqrt(2*pi))
    return a*b

class Reinforce():
    def __init__(self, env, hidden_size, sess=None, gamma = 0.8):
        self.gamma = gamma
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.hidden_size = hidden_size
        scope_var = "r_network"
        clt_name_var = ["r_net_prmt", tf.GraphKeys.GLOBAL_VARIABLES] # 定义了collections
        self.policynet(scope_var,clt_name_var)
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter("./Reinforce_con/summaries", sess.graph)


    def policynet(self, scope, collections_name):
        weights_init = tf.truncated_normal_initializer(0, 0.3)
        bias_init = tf.constant_initializer(0.1)
        
        self.inputs = tf.placeholder(dtype = tf.float32, shape=[None, self.state_dim],  name = "inputs")
        self.target = tf.placeholder(dtype = tf.float32, shape = [None], name = "rewards")
        self.action = tf.placeholder(dtype = tf.float32, shape = [None, self.action_dim], name = "action")

        with tf.variable_scope(scope):
            with tf.variable_scope("layer1"):
                weights1 = tf.get_variable(name = "weights", dtype=tf.float32, shape=[self.state_dim, self.hidden_size], initializer=weights_init, collections=collections_name)
                bias1 = tf.get_variable(name = "bias", dtype=tf.float32, shape=[self.hidden_size], initializer=bias_init, collections=collections_name)
                wx_b = tf.matmul(self.inputs, weights1) + bias1
                h1 = tf.nn.relu(wx_b)            
           
            with tf.variable_scope("layer2"):
                weights2 = tf.get_variable(name = "weights", dtype=tf.float32, shape=[self.hidden_size, self.action_dim], initializer=weights_init, collections=collections_name)
                bias2 = tf.get_variable(name = "bias", dtype=tf.float32, shape=[self.action_dim], initializer=bias_init, collections=collections_name)
                mu = tf.nn.tanh(tf.matmul(h1, weights2) + bias2)
                
            with tf.variable_scope("layer3"):
                weights3 = tf.get_variable(name = "weights", dtype=tf.float32, shape=[self.hidden_size, self.action_dim], initializer=weights_init, collections=collections_name)
                bias3 = tf.get_variable(name = "bias", dtype=tf.float32, shape=[self.action_dim], initializer=bias_init, collections=collections_name)
                sigma = tf.nn.softplus(tf.matmul(h1, weights3) + bias3)
    
        self.mu, self.sigma = tf.squeeze(mu*2), tf.squeeze(sigma+0.1)
        tf.summary.histogram('/mu', self.mu) 
        tf.summary.histogram('/sigma', self.sigma) 
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)
        self.act = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0], action_bound[1])
        
        with tf.variable_scope("loss"):
            self.action = tf.clip_by_value(self.action, action_bound[0], action_bound[1])
            self.log_prob = self.normal_dist.log_prob(self.action)
            self.entropy = self.normal_dist.entropy()
            tf.summary.histogram('/entropy', self.entropy) 
            self.loss = -tf.reduce_mean(self.log_prob * self.target-self.entropy*0.01)
            tf.summary.histogram('/loss', self.loss) 

        with tf.variable_scope("train"):
            self.train_op = tf.train.RMSPropOptimizer(0.01).minimize(self.loss)
    
    def train(self, state, target, action):
        "train process"
        #print("state", state)
        #print("action", action)
        #print("before",self.sess.run([self.mu, self.sigma, self.action_prob], feed_dict={self.inputs: state, self.action: action}))
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.inputs: state, self.target: target, self.action: action})
        #print("loss",loss)
        #print("after",self.sess.run([self.mu, self.sigma, self.action_prob], feed_dict={self.inputs: state, self.action: action}))
        #print("normal",normal(self.action, self.mu, self.sigma))
    
    def choose_action(self, current_state):
        current_state = current_state[np.newaxis, :]
        return self.sess.run(self.act, feed_dict={self.inputs: current_state})

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

if __name__ == "__main__":
    with tf.Session() as sess:
        RF = Reinforce(env, 64, sess)
        update_iter = 0
        running_reward=0
        for episode in range(EPISODES):
            state = env.reset()
            memory = []
            reward_all = 0
            #print("state",state)
            for step in range(MAX_STEP):
                env.render()
                action = RF.choose_action(state)
                next_state, reward, done , _ = env.step(action)
                reward /=   10
                reward_all += reward
                #keep track of transition        
                memory.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))
                state = next_state
                #print("step = {}".format(step))
                if done:
                    break
                
            running_reward = running_reward*0.99 + 0.01*reward_all
            print("episode = {} reward = {}".format(episode, running_reward))
                
	        # Go through the episode and make policy updates
            advantage = []
            states = []
            actions = []
            for t, transition in enumerate(memory):
                # The return after this timestep
                total_return = sum(RF.gamma**i * t.reward for i, t in enumerate(memory[t:]))
                states.append(transition.state)
                actions.append(transition.action) 
                advantage.append(total_return)
                #print(advantage)
                # Update our policy estimator
                #print(transition)      
            advantage -= np.mean(advantage)
            advantage /= np.std(advantage)
            RF.train(states, advantage, actions)
            merged = tf.summary.merge_all()
            rs = sess.run(merged,feed_dict={RF.inputs: states, RF.target: advantage, RF.action: actions})
            RF.writer.add_summary(rs, episode)
    env.close()
