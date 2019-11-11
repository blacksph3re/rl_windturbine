import random
import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from collections import deque

# OU Noise with constant sigma
class OUNoise:
    def __init__(self, dim, theta, sigma, mu=None):
        self.mu = np.array(mu) if mu is not None else np.zeros(dim)
        self.dim = dim
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu

    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.dim)
        self.state = x + dx
        return self.state

    def get_noise(self, _t=0):
        return self.evolve_state()

# Ornstein-Ulhenbeck Noise
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
# Sigma is the value for how much noise
# Theta is the value for how close to stay around mu
class OUNoiseDec:
    def __init__(self, dim, theta=0.15, sigma_start=0.3, sigma_start_step=0, sigma_end=None, sigma_end_step=1, mu=None):
        sigma_end = sigma_end if sigma_end is not None else sigma_start
        sigma_end_step = max(sigma_start_step+1, sigma_end_step)

        self.sigma_start_step = sigma_start_step
        self.sigma_end_step = sigma_end_step
        self.ounoise = OUNoise(dim, theta, sigma_start, mu)
        m = (sigma_start - sigma_end)/(sigma_start_step - sigma_end_step)
        c = sigma_start - m * sigma_start_step
        self.sigma_polynom = (m, c)

        self.reset()
        
    def reset(self):
        self.ounoise.reset()

    def update_sigma(self, t):
        m, c = self.sigma_polynom
        x = np.clip(t, self.sigma_start_step, self.sigma_end_step)
        self.ounoise.sigma = np.clip((m * x + c), 0, 1)
    
    def get_noise(self, t=0):
        self.update_sigma(t)
        return self.ounoise.get_noise()

class GaussianNoise:
    def __init__(self, dim, mean=0, variance=1):
        self.dim = dim
        self.mean = mean
        self.variance = variance

    def get_noise(self, _t=0):
        return np.random.randn(dim) * variance + mean


class UniformNoise:
    def __init__(self, dim, noise_factor):
        self.dim = dim
        self.noise_factor = noise_factor

    def get_noise(self, _t=0):
        return np.random.uniform(-np.ones(self.dim), np.ones(self.dim), self.dim) * noise_factor

class UniformNoiseDec:
    def __init__(self, dim, noise_start_factor=0.01, noise_start=0, noise_end_factor=None, noise_end=None):
        noise_end = max(noise_start+1, noise_end or 0)
        noise_end_factor = noise_end_factor if noise_end_factor is not None else noise_start_factor
        self.noise_start = noise_start
        self.noise_end = noise_end

        m = (noise_start_factor - noise_end_factor)/(noise_start - noise_end)
        c = noise_start_factor - m * noise_start
        self.noise_polynom = (m,c)
        self.dim = dim

    def calc_level(self, t):
        m, c = self.noise_polynom
        x = np.clip(t, self.noise_start, self.noise_end)
        return np.clip((m * x + c), 0, 1)

    def get_noise(self, t=0):
        noise_level = self.calc_level(t)
        return np.random.uniform(-np.ones(self.dim), np.ones(self.dim), self.dim) * noise_level

  
# Helper class for not having noise at all
class NoNoise:
    def __init__(self, dim, default_action=None):
        self.default_action = default_action if default_action is not None else np.zeros(self.dim)
        self.dim = dim

    def get_noise(self, _t=0):
        return np.zeros(dim)

class CapacityBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def append(self, data):
        self.push(data)
    def push(self, data):
        self.buffer.append(data)
        while(len(self.buffer) > self.max_size):
            del self.buffer[-1]

    def get_buffer(self):
        return self.buffer

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

'''
A tree structure, where every of the nodes is the sum of all the leaves below
This can be used to do efficient sampling (without np.random.choice fucking me up)
The weights buffer is a buffer with the tree weights stored like
1  2 3  4 5 6 7  8 9 10 11 12 13 14 15...
'''
class SumTree:
    def __init__(self, max_size):
        self.max_size = max_size
        self.tree_layers = int(np.log2(max_size)) + 1
        self.tree_size = np.sum([2**x for x in range(0, self.tree_layers)])
        self.weights = np.zeros(self.tree_size + self.max_size)

    # Get the weight trajectory for the leaf
    # All the weights from the leaf to the root node are in here
    # The trajectory will always have self.tree_layers + 1 entries
    def leaf_trajectory(self, leaf_idx):
        # Get the actual index in the leaf parts and make it 1-terminated
        idx = leaf_idx + self.tree_size + 1

        # However, store everything 0-terminated
        trajectory = [idx - 1]
        # Now step through the layers until we are at the root node
        while(idx > 1):
            # The next lower layer is just the current idx divided by two and floored down
            idx = int(idx / 2)
            trajectory.append(idx - 1)

        return trajectory

    def get_leaf_weights(self):
        return self.weights[self.tree_size:]

    # Update a leaf and propagate the sums back through the net
    def update_entry(self, leaf_idx, weight):
        assert(weight >= 0)
        weight = float(weight)

        trajectory = self.leaf_trajectory(leaf_idx)
        # Get the previous weight at this position
        weight_diff = weight - self.weights[trajectory[0]]
        # Add the difference to all the nodes in the trajectory
        self.weights[trajectory] += weight_diff

    # Samples an item from the tree
    def sample(self):
        # Assert not empty
        assert(self.weights[0] != 0)

        return self.sample_rec(2, self.weights[0]) - 1 - self.tree_size

    # Make one branch decision
    # Index is a 1-terminated index in the tree, pointing to the left node of the two between which to decide
    # Regularization is the weight of the parent node
    # It returns the 1-terminated index in the weights
    def sample_rec(self, index, regularization):
        # Make a decision between the two weights
        prob_left = self.weights[index-1]/regularization
        go_left = np.random.uniform() < prob_left

        if(not go_left):
            index += 1

        # Check whether we reached the leaf already
        if(index > self.tree_size):
            return index

        # If not, continue recursively
        return self.sample_rec(index * 2, self.weights[index - 1])

# A "Basic" Buffer which stores entries and can sample them by priority
class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.iterator = 0 # Next position to write to
        self.full = False # Whether we are full
        self.buffer = np.array([(None, None, None, None, None) for _ in range(max_size)])
        self.priorities = SumTree(max_size)
        self.max_priority = 1

    def push(self, state, action, reward, next_state, done, priority=None):
        # If not given a priority, make sure it's getting sampled
        if(priority == None):
            priority = self.max_priority
        # If priority zero, increase it by just a little so it could still be sampled
        # Also avoid filling the buffer with zero-priority samples
        if(priority == 0):
            priority = 1e-15

        experience = (state, action, np.array([reward]), next_state, done)

        self.max_priority = max(self.max_priority, priority)

        # Store experience
        self.buffer[self.iterator] = experience
        self.priorities.update_entry(self.iterator, priority)
        self.iterator += 1

        if(self.iterator >= self.max_size):
            self.iterator = 0
            self.full = True

    def update_priorities(self, indices, priorities):
        assert(len(indices) == len(priorities))
        for (i, p) in zip(indices, priorities):
            self.max_priority = max(self.max_priority, p)
            self.priorities.update_entry(i, p)

    def get_buffer(self):
        if(self.full):
            return self.buffer
        else:
            return self.buffer[0:self.iterator]

    def sample(self, batch_size):
        if(len(self.buffer) == 0):
            return [], [], [], [], [], []


        # Sample according to priorities
        indices = [self.priorities.sample() for _ in range(0, batch_size)]

        # Convert indices to lists of individual batches
        batch = self.buffer[indices]

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices)

    def get_priorities(self):
        return self.priorities.get_leaf_weights()[0:len(self)]

    def sample_sequence(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        min_start = len(self) - batch_size
        start = np.random.randint(0, min_start)

        for sample in range(start, start + batch_size):
            state, action, reward, next_state, done = self.buffer[start]
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        if(not self.full):
            return self.iterator
        return self.max_size

    def save(self, file):
        data = {
            "buffer": self.buffer[0:len(self)],
            "priorities": self.priorities.get_leaf_weights()[0:len(self)]
        }

        with open(file, "wb") as f:
            pickle.dump(data, f)

    def load(self, file):
        try:
            self.full = False
            self.iterator = 0

            # Load everything
            with open(file, "rb") as f:
                data = pickle.load(f)
                
            self.buffer = data["buffer"]
            # Update own class
            if(len(self.buffer) < self.max_size):
                self.full = False
                self.iterator = len(self.buffer)

                self.buffer = np.concatenate((self.buffer, [(None, None, None, None, None) for _ in range(len(self.buffer), self.max_size)]))
            elif(len(self.buffer) > self.max_size):
                self.buffer = self.buffer[0:self.max_size]
                self.full = True
                self.iterator = 0
            else:
                self.full = True
                self.iterator = 0
            
            # Update weights
            max_prio = np.max(data["priorities"])
            for i in range(0, len(self)):
                if(i >= len(data["priorities"])):
                    self.priorities.update_entry(i, max_prio)
                else:
                    self.priorities.update_entry(i, data["priorities"][i])

        except Exception as e:
            print("Could not load memory: %s" % str(e))


class QBladeLogger:

    def __init__(self, logdir, log_steps):
        self.logdir = logdir
        self.log_steps = log_steps
        self.writer = SummaryWriter(logdir + '/' + str(datetime.now()))

    def logObservation(self, step, observation, prefix='obs'):
        labels = {
          0: 'rotational speed [rad/s]',
          1: 'power [kW]',
          2: 'HH wind velocity [m/s]',
          3: 'yaw angle [deg]',
          4: 'pitch blade 1 [deg]',
          5: 'pitch blade 2 [deg]',
          6: 'pitch blade 3 [deg]',
          7: 'tower top bending local x [Nm]',
          8: 'tower top bending local y [Nm]',
          9: 'tower top bending local z [Nm]',
          10: 'oop bending blade 1 [Nm]',
          11: 'oop bending blade 2 [Nm]',
          12: 'oop bending blade 3 [Nm]',
          13: 'ip bending blade 1 [Nm]',
          14: 'ip bending blade 2 [Nm]',
          15: 'ip bending blade 3 [Nm]',
          16: 'oop tip deflection blade 1 [m]',
          17: 'oop tip deflection blade 2 [m]',
          18: 'oop tip deflection blade 3 [m]',
          19: 'ip tip deflection blade 1 [m]',
          20: 'ip tip deflection blade 2 [m]',
          21: 'ip tip deflection blade 3 [m]',
          22: 'current time [s]'
        }

        for i in range(0, len(observation)):
            self.add_scalar('%s/%s'% (prefix, labels[i]), observation[i], step)

    def add_scalar(self, name, val, time):
        # Don't do anything if not in the logging step
        if(time%self.log_steps != 0):
            return

        self.writer.add_scalar(name, val, time)

    def add_scalar_nofilter(self, name, val, time):
        self.writer.add_scalar(name, val, time)

    def add_histogram(self, name, val, time):
        if(time%self.log_steps != 0):
            return
        self.writer.add_histogram(name, val, time)

    def add_histogram_nofilter(self, name, val, time):
        self.writer.add_histogram(name, val, time)

    def logAction(self, step, action, prefix='act'):
        self.add_scalar('%s/generator torque [Nm]' % prefix, action[0], step)
        #self.add_scalar('%s/yaw angle [deg]' % prefix, action[1], step)
        self.add_scalar('%s/pitch blade 1 [deg]' % prefix, action[1], step)
        #self.add_scalar('%s/pitch blade 2 [deg]' % prefix, action[3], step)
        #self.add_scalar('%s/pitch blade 3 [deg]' % prefix, action[4], step)

    def close(self):
        self.writer.close()