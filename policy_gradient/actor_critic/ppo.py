import tensorflow as tf
import gym
import numpy as np
import collections
import random
import tensorflow.contrib.layers as layers
import matplotlib
from math import *

ENV = "Pendulum-v0"

EPISODES = 1000
MAX_STEP = 200
GAMMA = 0.9
BATCH_SIZE = 32
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0002     # learning rate for critic
S_DIM, A_DIM = 3, 1


env = gym.make(ENV)
env.seed(1)

class PPO(object):
    def __init__(self, sess):
        self.sess = sess
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, )
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)
    def update(self, s, a, r):
        pass
    def choose_action(self, s):
        pass
    def get_v(self, s):
        pass


if __name__ == "__main__":
    with tf.Session() as sess:
        ppo = PPO(sess)
        for ep in range(EPISODES):
            s = env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(MAX_STEP):
                env.render()
                a = ppo.choose(s)
                s_, r, done, _ = env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                s = s_

                if (t+1) % BATCH_SIZE == 0 or t == MAX_STEP-1:
                    v_s_ = ppo.get_v(s_)
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = batch(buffer_s, buffer_a, buffer_r)
                    # 清空buffer
                    buffer_s, buffer_a, buffer_r = [], [], []
                    ppo.update(bs, ba, br) # 更新ppo


        tf.summary.FileWriter("./Reinforce_con/summaries", sess.graph)