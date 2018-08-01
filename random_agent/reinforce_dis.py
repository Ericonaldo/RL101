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
EPISODES = 10000
MAX_STEP = 500
BATCH_SIZE = 32

env = gym.make(ENV)

class Reinforce():
    def __init__(self, env, hiddens, sess=None, gamma = 0.8):
        self.gamma = gamma
        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.hiddens = hiddens
        scope_var = "r_network"
        clt_name_var = ["r_net_prmt", tf.GraphKeys.GLOBAL_VARIABLES] # 定义了collections
        self.policynet(scope_var,clt_name_var)
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter("./Reinforce_con/summaries", sess.graph)


    def policynet(self, scope, collections_name, reuse=tf.AUTO_REUSE):
        weights_init = tf.truncated_normal_initializer(0, 0.3)
        bias_init = tf.constant_initializer(0.1)
        
        self.inputs = tf.placeholder(dtype = tf.float32, shape=[None, self.state_dim],  name = "inputs")
        self.target = tf.placeholder(dtype = tf.float32, shape = [None,], name = "rewards")
        self.action = tf.placeholder(dtype = tf.int32, shape = [None,], name = "action")

        with tf.variable_scope(scope, reuse = reuse):
            out = self.inputs
            for hidden in self.hiddens:
                out = tf.contrib.layers.fully_connected(out, num_outputs = hidden, activation_fn = tf.nn.relu)
            out = tf.contrib.layers.fully_connected(out, num_outputs = self.action_dim, activation_fn = None)

        self.act_prob = tf.nn.softmax(out)
                
        with tf.variable_scope("loss"):
            self.log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.act_prob, labels = self.action)
            self.loss = tf.reduce_mean(self.log_prob * self.target)

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
        prob_weights = self.sess.run(self.act_prob, feed_dict={self.inputs: current_state})    # 所有 action 的概率
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # 根据概率来选 action
        return action

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

if __name__ == "__main__":
    with tf.Session() as sess:
        RF = Reinforce(env, [64], sess)
        update_iter = 0
        step_his = []
        running_reward=0
        for episode in range(EPISODES):
            state = env.reset()
            memory = []
            reward_all = 0
            step_view = 0
            #print("state",state)
            for step in range(MAX_STEP):
                env.render()
                action = RF.choose_action(state)
                next_state, reward, done , _ = env.step(action)
                #reward /=   10
                reward_all += reward
                #keep track of transition        
                memory.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))
                if step != 199 and done:
                    step_view = step
                    break
                state = next_state
            
            running_reward = running_reward*0.99 + 0.01*reward_all
            print("episode = {} step_view={} reward = {}".format(episode, step_view, running_reward))
                
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
    env.close()
