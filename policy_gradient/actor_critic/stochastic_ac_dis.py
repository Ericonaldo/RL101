import tensorflow as tf
import gym
import numpy as np
import collections
import random
import tensorflow.contrib.layers as layers
import matplotlib
from math import *

ENV = "CartPole-v0"

MEMORY_SIZE = 10000
EPISODES = 1000
MAX_STEP = 500
GAMMA = 0.9
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make(ENV)

class actor():
    def __init__(self, env, hiddens, sess=None):
        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.hiddens = hiddens
        scope_var = "actor_network"
        clt_name_var = ["actor_net_prmt", tf.GraphKeys.GLOBAL_VARIABLES] # 定义了collections
        self.network(scope_var,clt_name_var)
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter("./ac_dis/summaries", sess.graph)


    def network(self, scope, collections_name, reuse=tf.AUTO_REUSE):
        #weights_init = tf.truncated_normal_initializer(0, 0.3)
        #bias_init = tf.constant_initializer(0.1)
        
        self.inputs = tf.placeholder(dtype = tf.float32, shape=[None, self.state_dim],  name = "inputs")
        self.td_error = tf.placeholder(dtype = tf.float32, shape = [None,], name = "td_error")
        self.action = tf.placeholder(dtype = tf.int32, shape = [None,], name = "action")

        with tf.variable_scope(scope, reuse = reuse):
            out = self.inputs
            for hidden in self.hiddens:
                out = tf.contrib.layers.fully_connected(out, num_outputs = hidden, activation_fn = tf.nn.relu)
            out = tf.contrib.layers.fully_connected(out, num_outputs = self.action_dim, activation_fn = None)
        
        self.act_prob = tf.nn.softmax(out)
        tf.summary.histogram("actorlayer" + '/act_prob', self.act_prob)
        
        with tf.variable_scope("loss"):
            self.log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.act_prob, labels = self.action)
            self.loss = tf.reduce_mean(self.log_prob * self.td_error)

        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer(LR_A).minimize(self.loss, global_step = tf.Variable(0, trainable=False))
    
    def train(self, state, td_error, action):
        "train process"
        state = state[np.newaxis, :]
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.inputs: state, self.td_error: [td_error], self.action: [action]})
        return loss
    
    def choose_action(self, current_state):
        current_state = current_state[np.newaxis, :]
        prob_weights = self.sess.run(self.act_prob, feed_dict={self.inputs: current_state})    # 所有 action 的概率
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # 根据概率来选 action
        return action

class critic():
    def __init__(self, env, hiddens, sess=None):
        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.hiddens = hiddens
        scope_var = "critic_network"
        clt_name_var = ["critic_net_prmt", tf.GraphKeys.GLOBAL_VARIABLES] # 定义了collections
        self.network(scope_var,clt_name_var)
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter("./Reinforce_con/summaries", sess.graph)

    def network(self, scope, collections_name, reuse=tf.AUTO_REUSE):
        #weights_init = tf.truncated_normal_initializer(0, 0.3)
        #bias_init = tf.constant_initializer(0.1)
        
        self.inputs = tf.placeholder(dtype = tf.float32, shape=[None, self.state_dim], name = "inputs")
        self.value_pre = tf.placeholder(dtype = tf.float32, shape = None, name = "value_pre")
        self.reward = tf.placeholder(dtype = tf.float32, shape = None, name="rewards")

        with tf.variable_scope(scope, reuse = reuse):
            out = self.inputs
            for hidden in self.hiddens:
                out = tf.contrib.layers.fully_connected(out, num_outputs = hidden, activation_fn = tf.nn.relu)
            out = tf.contrib.layers.fully_connected(out, num_outputs = 1, activation_fn = None)

        self.value = out
 
        with tf.variable_scope("loss"):
         self.td_error = tf.reduce_mean(self.reward + GAMMA * self.value_pre - self.value)
         self.loss = tf.square(self.td_error)

        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer(LR_C).minimize(self.loss)
    
    def train(self, state, action, next_state):
        "train process"
        state, next_state = state[np.newaxis, :], next_state[np.newaxis, :]
        value_pre = self.sess.run(self.value, feed_dict={self.inputs: next_state})
        td_error, _ = self.sess.run([self.td_error,self.train_op], feed_dict={self.inputs: state, self.value_pre: value_pre, self.reward: reward})
        return td_error

    

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

if __name__ == "__main__":
    with tf.Session() as sess:
        a = actor(env, [30], sess)
        c = critic(env, [30], sess)
        running_reward=0
        for episode in range(EPISODES):
            state = env.reset()
            reward_all = 0
            step_view = 0
            for step in range(MAX_STEP):
                env.render()
                action = a.choose_action(state)
                next_state, reward, done , _ = env.step(action)
                if step < 199 and done:
                    reward -= 10
                reward_all += reward
                td_error = c.train(state, action, next_state)
                a.train(state, td_error, action)
                if done:
                    step_view = step
                    break
                state = next_state
                if done:
                    break

                
            merged = tf.summary.merge_all()
            running_reward = running_reward*0.99 + 0.01*reward_all
            print("episode = {} step_view={} reward = {}".format(episode, step_view, running_reward))
    env.close()
