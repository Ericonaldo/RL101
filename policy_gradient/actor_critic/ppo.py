import tensorflow as tf
import gym
import numpy as np
import collections
import random
import tensorflow.contrib.layers as layers
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("Agg")
from math import *
import math

ENV = "Pendulum-v0"

EPISODES = 1000
MAX_STEP = 200
GAMMA = 0.9
BATCH_SIZE = 32
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0002     # learning rate for critic
S_DIM, A_DIM = 3, 1
EPSILON = 0.2
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10

class PPO(object):
    def __init__(self, sess):
        self.sess = sess
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(LR_C).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('old_pi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv

            self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-EPSILON, 1.+EPSILON) *self.tfadv
                ))
        
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(LR_A).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
    
    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable = trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable = trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable = trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r:r})

        # update actor
        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs:s, self.tfdc_r:r}) for _ in range(C_UPDATE_STEPS)]

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim <2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs:s})[0, 0]


if __name__ == "__main__":
    env = gym.make(ENV).unwrapped 
    env.seed(1)
    all_ep_r = []
    with tf.Session() as sess:
        ppo = PPO(sess)
        for ep in range(EPISODES):
            ep_r = 0
            s = env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(MAX_STEP):
                env.render()
                a = ppo.choose_action(s)
                s_, r, done, _ = env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r+8)/8)
                s = s_
                ep_r += r

                # update ppo
                if (t+1) % BATCH_SIZE == 0 or t == MAX_STEP-1:
                    v_s_ = ppo.get_v(s_)
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    # 清空buffer
                    buffer_s, buffer_a, buffer_r = [], [], []
                    ppo.update(bs, ba, br) # 更新ppo
                if done:
                    break


            if ep==0:
                all_ep_r.append(ep_r)
            else:
                all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)

            print(
                'Ep: ', ep,
                '|Ep_r: ', ep_r,
            )
    env.close()
    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.savefig("ppo_reward.png")
    plt.show()
