import random
import numpy as np
import pickle
import torch
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
        return np.random.randn(self.dim) * self.variance + self.mean


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
        self.dim = dim
        self.default_action = default_action if default_action is not None else np.zeros(self.dim)

    def get_noise(self, _t=0):
        return self.default_action

class CapacityBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def append(self, data):
        self.push(data)

    def push(self, data):
        self.buffer.append(data)
        while(len(self.buffer) > self.max_size):
            self.buffer.pop(0)

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
Note that alpha=0 is not supported!
'''
class SumTree:
    def __init__(self, max_size, alpha=1):
        self.max_size = max_size
        self.alpha = alpha
        self.tree_layers = int(np.log2(max_size)) + 1
        self.tree_size = np.sum([2**x for x in range(0, self.tree_layers)])
        self.weights = np.zeros(self.tree_size + self.max_size)
        self.leaf_weights = np.zeros(self.max_size)

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

    # Return the weights directly
    def get_leaf_weights(self):
        return self.leaf_weights

    def get_sum(self):
        return np.sum(self.leaf_weights)

    # Return the weights potentiated by alpha
    def get_leaf_weights_alpha(self):
        return self.weights[self.tree_size:]

    # We already have the sum of all items stored
    def get_sum_alpha(self):
        return self.weights[0]

    # Sets a new alpha
    # Warning, incurs complete recomputation
    def set_alpha(self, alpha):
        assert(alpha != 0)
        self.alpha = alpha
        self.recompute()

    # Recompute all weights to correct for floating point imprecision
    # Also necessary when changing alpha
    def recompute(self):
        self.weights = np.zeros(self.tree_size + self.max_size)
        for i in range(0, self.max_size):
            self.update_entry(i, self.leaf_weights[i])

    # The root node holds the sum of all weights
    def sum_weights(self):
        return self.weights[0]

    # Update a leaf and propagate the sums back through the net
    def update_entry(self, leaf_idx, weight):
        self.leaf_weights[leaf_idx] = float(weight)
        weight = np.clip(float(weight)**self.alpha, 0, 1.5)

        trajectory = self.leaf_trajectory(leaf_idx)
        # Get the previous weight at this position
        weight_diff = weight - self.weights[trajectory[0]]
        # Add the difference to all the nodes in the trajectory
        self.weights[trajectory] += weight_diff

        return weight


# A "Basic" Buffer which stores entries and can sample them by priority
class BasicBuffer:

    def __init__(self, max_size, device, state_dim, action_dim, alpha=1, dtype=torch.float):
        self.device = device
        self.max_size = max_size
        self.iterator = 0 # Next position to write to
        self.full = False # Whether we are full

        # CPU buffer
        self.buffer = np.array([(None, None, None, None, None) for _ in range(max_size)])

        self.dtype = dtype

        # GPU buffers
        self.buffer_s = torch.zeros(max_size, state_dim, dtype=self.dtype, device=device, requires_grad=False)
        self.buffer_a = torch.zeros(max_size, action_dim, dtype=self.dtype, device=device, requires_grad=False)
        self.buffer_snext = torch.zeros(max_size, state_dim, dtype=self.dtype, device=device, requires_grad=False)
        self.buffer_r = torch.zeros(max_size, 1, dtype=self.dtype, device=device, requires_grad=False)
        self.buffer_d = torch.zeros(max_size, 1, dtype=self.dtype, device=device, requires_grad=False)

        self.priorities = SumTree(max_size, alpha)
        self.max_priority = 1

    def copy_buffer_to_device(self):
        for i in range(0, len(self)):
            self.buffer_s[i] = torch.tensor(self.buffer[i][0], dtype=self.dtype, requires_grad=False)
            self.buffer_a[i] = torch.tensor(self.buffer[i][1], dtype=self.dtype, requires_grad=False)
            self.buffer_r[i] = torch.tensor(self.buffer[i][2], dtype=self.dtype, requires_grad=False)
            self.buffer_snext[i] = torch.tensor(self.buffer[i][3], dtype=self.dtype, requires_grad=False)
            self.buffer_d[i] = torch.tensor(self.buffer[i][4], dtype=self.dtype, requires_grad=False)

    def push(self, state, action, reward, next_state, done, priority=None):
        # If not given a priority, make sure it's getting sampled
        if(priority == None):
            priority = self.max_priority
        # If priority zero, increase it by just a little so it could still be sampled
        # Also avoid filling the buffer with zero-priority samples
        if(priority == 0):
            priority = 1e-15

        done = 1.0 if done else 0.0

        self.max_priority = max(self.max_priority, priority)

        # Store experience
        self.buffer[self.iterator] = (state, action, np.array([reward]), next_state, np.array([done]))

        self.buffer_s[self.iterator] = torch.tensor(state, requires_grad=False)
        self.buffer_a[self.iterator] = torch.tensor(action, requires_grad=False)
        self.buffer_snext[self.iterator] = torch.tensor(next_state, requires_grad=False)
        self.buffer_r[self.iterator] = torch.tensor([reward], requires_grad=False)
        self.buffer_d[self.iterator] = torch.tensor([done], requires_grad=False)

        self.priorities.update_entry(self.iterator, priority)
        self.iterator += 1

        if(self.iterator >= self.max_size):
            self.iterator = 0
            self.full = True

    def update_priorities(self, indices, priorities):
        assert(len(indices) == len(priorities))
        for (i, p) in zip(indices, priorities):
            weight = self.priorities.update_entry(i, p)
            self.max_priority = max(self.max_priority, weight)

    # Gets the prioritized to the power of alpha and normalized to sum to 1
    def get_probabilities(self, indices=None):
        weights = self.priorities.get_leaf_weights_alpha()[0:len(self)]
        if(isinstance(indices, list)):
            weights = weight[indices]
        weights = weights/self.priorities.get_sum_alpha()
        return weights

    def get_priorities(self):
        return self.priorities.get_leaf_weights()[0:len(self)]

    def get_buffer(self):
        if(self.full):
            return self.buffer
        else:
            return self.buffer[0:self.iterator]

    def sample(self, batch_size, uniform=False):
        if(len(self.buffer) == 0):
            return [], [], [], [], [], []

        # Sample according to priorities or uniformly
        if(not uniform):
            indices = np.random.choice(len(self), batch_size, p=self.get_probabilities())
        else:
            indices = np.random.choice(len(self), batch_size)

        return self.buffer_s[indices], self.buffer_a[indices], self.buffer_r[indices], self.buffer_snext[indices], self.buffer_d[indices], indices

    def __len__(self):
        if(not self.full):
            return self.iterator
        return self.max_size

    def save(self, file):
        data = {
            "buffer": self.buffer[0:len(self)],
            "priorities": self.priorities.get_leaf_weights()[0:len(self)],
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

            # Copy to GPU
            self.copy_buffer_to_device()


        except Exception as e:
            print("Could not load memory: %s" % str(e))


class QBladeLogger:

    def __init__(self, logdir, log_steps, run_name, obs_labels = {}, act_labels = {}, feed_past=0):
        self.logdir = logdir
        self.log_steps = log_steps
        run_name = run_name if run_name else str(datetime.now())
        self.writer = SummaryWriter('%s/%s' % (logdir, run_name))
        self.obs_labels = obs_labels
        self.act_labels = act_labels
        self.feed_past = feed_past

    def logObservation(self, step, observation, prefix='obs'):
        length = len(observation)
        if(self.feed_past):
            length = int(length/(self.feed_past+1))
        for i in range(0, length):
            if i in self.obs_labels:
                label = self.obs_labels[i]
            else:
                label = 'unknown %d' % i
            self.add_scalar('%s/%s'% (prefix, label), observation[i], step)

    def add_scalar(self, name, val, time):
        # Don't do anything if not in the logging step
        if(time%self.log_steps != 0):
            return

        self.add_scalar_nofilter(name, val, time)

    def add_scalar_nofilter(self, name, val, time):
        try:
            self.writer.add_scalar(name, val, time)
        except Exception as e:
            print(e)

    def add_histogram(self, name, val, time):
        if(time%self.log_steps != 0):
            return
        self.add_histogram_nofilter(name, val, time)

    def add_histogram_nofilter(self, name, val, time):
        try:
            self.writer.add_histogram(name, val, time)
        except Exception as e:
            print(e)

    def logAction(self, step, action, prefix='act'):
        for i in range(0, len(action)):
            if i in self.act_labels:
                label = self.act_labels[i]
            else:
                label = 'unknown %d' % i
            self.add_scalar('%s/%s'% (prefix, label), action[i], step)


    def close(self):
        self.writer.close()