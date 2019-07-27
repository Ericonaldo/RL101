import random

from collections import namedtuple


Transition = namedtuple("Transition", "state, action, next_state, reward, done")


class Buffer:
    def __init__(self, capacity):
        self._data = []
        self._capacity = capacity
        self._flag = 0

    def __len__(self):
        return len(self._data)

    def push(self, *args):
        """args: state, action, next_state, reward, done"""

        if len(self._data) < self._capacity:
            self._data.append(None)

        self._data[self._flag] = Transition(*args)
        self._flag = (self._flag + 1) % self._capacity

    def sample(self, batch_size):
        if len(self._data) < batch_size:
            return None

        samples = random.sample(self._data, batch_size)

        return Transition(*zip(*samples))

class ReplayBuffer(object):
    def __init__(self, config):
        """ReplayBuffer work for off-policy, it will store experiences for agent,
        and its inner data-structure will store some `(s_t, action_t, reward_t, terminal)` sequences
        """

        self.memory_size, obs_height, obs_width = config.memory_size, config.obs_height, config.obs_width

        self._obs_mem = np.empty(shape=(self.memory_size, obs_height, obs_width), dtype=np.float32)
        self._action_mem = np.empty(shape=(self.memory_size, 1), dtype=np.uint8)
        self._reward_mem = np.empty(shape=(self.memory_size, 1), dtype=np.float32)
        self._terminal_mem = np.empty(shape=(self.memory_size, 1), dtype=np.bool)

        self.pos_flag = 0  # indicate the position which newest sequence will insert
        self.dims = (config.obs_height, config.obs_width)   # matains a shape of coming observation
        self.counter = 0  # matains a counter for experiences storage

        self.history_length = config.history_length
        self.batch_size = config.batch_size

    def add(self, obs_t, action_t, reward_t, terminal):
        """Will store newest experience and drop oldest experience
        """

        assert obs_t.shape == self.dims

        self._obs_mem[self.pos_flag] = obs_t
        self._action_mem[self.pos_flag] = action_t
        self._reward_mem[self.pos_flag] = reward_t
        self._terminal_mem[self.pos_flag] = terminal

        # update indication
        self.pos_flag = (self.pos_flag + 1) % self.memory_size
        self.counter += 1 if self.counter < self.memory_size else 0

    def sample(self):
        """Will sample some experiences from its storage
        """

        assert self.counter > self.history_length

        idx = np.random.choice(self.counter, self.batch_size)
        obs_batch = self._obs_mem[idx, :]
        action_batch = self._action_mem[idx, :]
        obs_next_bath = self._obs_mem[(idx + 1) % self.counter, :]
        terminal_batch = self._terminal_mem[idx, :]

        return obs_batch, action_batch, reward_batch, obs_next_bath, terminal_batch

