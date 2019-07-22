#!/Users/mhliu/Program/anaconda3/envs/rl/bin/python
import tensorflow as tf
import gym
import numpy as np
import collections
import random
import tensorflow.contrib.layers as layers
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import *

ENV = "Pendulum-v0"

MEMORY_SIZE = 10000
EPISODES = 250
MAX_STEP = 200
GAMMA = 0.9
BATCH_SIZE = 32
LR_A = 0.001    # learning rate for actor
LR_C = 0.002     # learning rate for critic

TAU = 0.01 # soft replacement

env = gym.make(ENV)
env.seed(1)
action_bound = [env.action_space.low, env.action_space.high] 

class DDPG():
    def __init__(self, env, hiddens, sess=None):
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.hiddens = hiddens[0]
        self.sess = sess
        
        self.eval_s = tf.placeholder(dtype = tf.float32, shape = [None, self.state_dim], name = "eval_s")
        self.target_s = tf.placeholder(dtype = tf.float32, shape = [None, self.state_dim], name = "traget_s")
        self.reward = tf.placeholder(dtype = tf.float32, shape = [None,], name = "reward")
                
        with tf.variable_scope('actor_network'):
            self.eval_a = self.actor_net(self.eval_s, scope = "eval", trainable=True) # eval net
            self.target_a = self.actor_net(self.target_s, scope = "target", trainable=False) # target net
        with tf.variable_scope('critic_network'):
            self.eval_q = self.critic_net(self.eval_s, self.eval_a, scope = "eval", trainable=True)
            self.target_q = self.critic_net(self.target_s, self.target_a, scope = "target", trainable=False)
            
        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_network/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_network/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_network/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_network/target')
        
        with tf.variable_scope("soft_replace"):
            # target net replacement
            self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                                 for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]
                                 
        
        with tf.variable_scope("loss"):
            self.td_error = tf.reduce_mean(self.reward + GAMMA * self.target_q - self.eval_q)
            self.closs = tf.square(self.td_error)
            self.aloss = -tf.reduce_mean(self.eval_q)
            tf.summary.histogram("closs", self.closs)
            tf.summary.histogram("aloss", self.aloss)
            
        with tf.variable_scope("train"):
            self.ctrain_op = tf.train.AdamOptimizer(LR_C).minimize(self.closs, var_list=self.ce_params)
            self.atrain_op = tf.train.AdamOptimizer(LR_A).minimize(self.aloss, var_list=self.ae_params)
            
        sess.run(tf.global_variables_initializer())  
        
    def actor_net(self, inputs, scope, trainable, reuse=tf.AUTO_REUSE):
        #weights_init = tf.truncated_normal_initializer(0, 0.3)
        #bias_init = tf.constant_initializer(0.1)
        
        with tf.variable_scope(scope, reuse = reuse):
           out = inputs
           out = tf.contrib.layers.fully_connected(out, num_outputs = self.hiddens, activation_fn = tf.nn.relu, trainable = trainable)
           out = tf.contrib.layers.fully_connected(out, num_outputs = self.action_dim, activation_fn = tf.nn.tanh, trainable = trainable)
           out = out * 2
           out = tf.clip_by_value(out, action_bound[0], action_bound[1])    
        return out        
        
    def critic_net(self, input_s, input_a, scope, trainable, reuse=tf.AUTO_REUSE):
        #weights_init = tf.truncated_normal_initializer(0, 0.3)
        #bias_init = tf.constant_initializer(0.1)
        
        with tf.variable_scope(scope, reuse = reuse):
            w1_s = tf.get_variable('w1_s', [self.state_dim, self.hiddens], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.action_dim, self.hiddens], trainable=trainable)
            b1 = tf.get_variable('b1', [1, self.hiddens], trainable=trainable)
            net = tf.nn.relu(tf.matmul(input_s, w1_s) + tf.matmul(input_a, w1_a) + b1)
            out = tf.contrib.layers.fully_connected(net, num_outputs = 1, activation_fn = None) # Q(s,a)
        return out
        
    
    def train(self, state, next_state, action, reward):
        "train process"
        self.sess.run(self.soft_replace)
        
        self.sess.run(self.atrain_op, {self.eval_s: state}) # 注意这里只传入了eval_s，说明Q(s,a)的a也是经过policy network出来的，这样是为了计算梯度。否则就没有梯度了
        self.sess.run(self.ctrain_op, {self.eval_s: state, self.target_s: next_state, self.eval_a: action, self.reward: reward}) # 注意eval_a结点直接用action填充，eval_a的梯度不需要回传
    
    
    def choose_action(self, current_state):
        current_state = current_state[np.newaxis, :]
        action = self.sess.run(self.eval_a, feed_dict={self.eval_s: current_state})
        return action[0]
    
memory=[]
Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


sigma = 3
mu = 0

ep=[]
re=[]
if __name__ == "__main__":
    with tf.Session() as sess:
        ddpg = DDPG(env, [30], sess)
        #tf.summary.FileWriter("./Reinforce_con/summaries", sess.graph)

        running_reward=0
        for episode in range(EPISODES):
            state = env.reset()
            reward_all = 0
            for step in range(MAX_STEP):
                #  env.render()
                            
                action = ddpg.choose_action(state)              
                # action = np.clip(np.random.normal(action, sigma), action_bound[0], action_bound[1])    # add randomness to action selection for exploration
                action = np.clip(action + np.random.normal(mu, sigma), action_bound[0],action_bound[1])
                next_state, reward, done , _ = env.step(action)
                # print(reward)
                reward_all += reward
                
                memory.append(Transition(state, action, reward/10, next_state, float(done)))
                
                if len(memory) > 10000: # BATCH_SIZE * 4:
                    sigma *= .9995
                    batch_trasition = random.sample(memory, BATCH_SIZE)
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = map(np.array, zip(*batch_trasition))
                    ddpg.train(state = batch_state, next_state = batch_next_state, action = batch_action, reward = batch_reward)

                state = next_state
                
            #merged = tf.summary.merge_all()
            #running_reward = running_reward*0.99 + 0.01*reward_all
            #print("episode = {} reward = {}".format(episode, running_reward))
            print("episode = {} reward = {}".format(episode, reward_all))
            ep.append(episode)
            re.append(reward_all)
    env.close()
    plt.plot(ep, re)
    plt.show()
