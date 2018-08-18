import tensorflow as tf
import gym
import numpy as np
import collections
import random
import tensorflow.contrib.layers as layers

ENV = "CartPole-v0"

MEMORY_SIZE = 10000
EPISODES = 500
MAX_STEP = 500
BATCH_SIZE = 32
UPDATE_PERIOD = 200

class DeepQNetwork():
    def __init__(self, env, sess=None, gamma = 0.8, epsilon=0.8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.network()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter("DQN_dis/summaries", sess.graph)


    def net_frame(self, scope, collections_name, num_actions, inputs):
        "basic net frame"
        weights_init = tf.truncated_normal_initializer(0, 0.3)
        bias_init = tf.constant_initializer(0.1)

        with tf.variable_scope(scope):
            with tf.variable_scope("layer1"):
                weights1 = tf.get_variable(name = "weights", dtype=tf.float32, shape=[self.state_dim, 64], initializer=weights_init, collections=collections_name)
                bias1 = tf.get_variable(name = "bias", dtype=tf.float32, shape=[64], initializer=bias_init, collections=collections_name)
                wx_b = tf.matmul(inputs, weights1) + bias1
                h1 = tf.nn.relu(wx_b)            
            with tf.variable_scope("layer2"):
                weights2 = tf.get_variable(name = "weights", dtype=tf.float32, shape=[64, 64], initializer=weights_init, collections=collections_name)
                bias2 = tf.get_variable(name = "bias", dtype=tf.float32, shape=[64], initializer=bias_init, collections=collections_name)
                wx_b = tf.matmul(h1, weights2) + bias2
                h2 = tf.nn.relu(wx_b)
            
            with tf.variable_scope("layer3"):
                weights3 = tf.get_variable(name = "weights", dtype=tf.float32, shape=[64, num_actions], initializer=weights_init, collections=collections_name)
                bias3 = tf.get_variable(name = "bias", dtype=tf.float32, shape=[num_actions], initializer=bias_init, collections=collections_name)
                q_out = tf.matmul(h2, weights3) + bias3
            
            return q_out            
    
    def network(self):
        "networks"
        # q_network
        self.inputs_q = tf.placeholder(dtype = tf.float32, shape = [None, self.state_dim], name = "inputs_q")
        scope_var = "q_network"
        clt_name_var = ["q_net_prmt", tf.GraphKeys.GLOBAL_VARIABLES] # 定义了collections
        self.q_value = self.net_frame(scope_var, clt_name_var, self.action_dim, self.inputs_q)
    
        # target_network
        self.inputs_target = tf.placeholder(dtype = tf.float32, shape = [None, self.state_dim], name = "inputs_target")
        scope_var = "target_network"
        clt_name_var = ["target_net_prmt", tf.GraphKeys.GLOBAL_VARIABLES] # 定义了collections
        self.q_target = self.net_frame(scope_var, clt_name_var, self.action_dim, self.inputs_target)

        with tf.variable_scope("loss"):
            self.target = tf.placeholder(dtype = tf.float32, shape = [None, self.action_dim], name="target")
            self.loss = tf.reduce_mean(tf.square(self.q_value - self.target))

        with tf.variable_scope("train"):
            self.train_op = tf.train.RMSPropOptimizer(0.01).minimize(self.loss)
    
    def train(self, state, reward, action, state_next, done):
        "train process"
        q, q_target = self.sess.run([self.q_value, self.q_target], feed_dict={self.inputs_q: state, self.inputs_target: state_next})
        target = reward + self.gamma * np.max(q_target, axis=1)

        self.reform_target = q.copy()
        batch_index = np.arange(BATCH_SIZE, dtype = np.int32)
        self.reform_target[batch_index, action] = target

        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.inputs_q:state, self.target:self.reform_target})

    def update_prmt(self):
        "update target network parameters"
        q_prmts = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "q_network")
        target_prmts = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "target_network")
        self.sess.run([tf.assign(t, q) for t,q in zip(target_prmts, q_prmts)]) #将Q网络参数赋值给target
        print("updating target-network parameters...")
    
    def choose_action(self, current_state):
        current_state = current_state[np.newaxis, :]
        # array dim : (xx, ) --> (1, xx)
        q = self.sess.run(self.q_value, feed_dict={self.inputs_q: current_state})

        # e-greedy
        if np.random.random() < self.epsilon:
            action_chosen = np.random.randint(0, self.action_dim)
        else:
            action_chosen = np.argmax(q)
        action_chosen = np.argmax(q)
        return action_chosen
    
    def decay_epsilon(self):
        if self.epsilon > 0.03:
            self.epsilon -= 0.02

#memory for memory replay
memory = []
Transition = collections.namedtuple("Transition", ["state", "action", "reward","next_state", "done"])

if __name__ == "__main__":
    env = gym.make(ENV)
    with tf.Session() as sess:
        DQN = DeepQNetwork(env, sess)
        update_iter = 0
        step_his = []
        for episode in range(EPISODES):
            state = env.reset()
            reward_all = 0
            for step in range(MAX_STEP):
                env.render()
                action = DQN.choose_action(state)
                next_state, reward, done , _ = env.step(action)
                reward_all += reward

                if len(memory) > MEMORY_SIZE:
                    memory.pop(0)
                memory.append(Transition(state, action, reward, next_state, float(done)))

                if len(memory) > BATCH_SIZE * 4:
                    batch_trasition = random.sample(memory, BATCH_SIZE)
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = map(np.array, zip(*batch_trasition))
                    DQN.train(state = batch_state, reward = batch_reward, action = batch_action, state_next = batch_next_state, done = batch_done)
                    update_iter += 1
    
                if update_iter and update_iter % UPDATE_PERIOD == 0:
                    DQN.update_prmt()

                if update_iter and update_iter % 200 == 0:
                    DQN.decay_epsilon()
                               
                if done:
                    print("[episode = {}] step = {}".format(episode, step))
                    break
 
                state = next_state

