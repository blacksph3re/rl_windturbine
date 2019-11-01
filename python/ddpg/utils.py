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

class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.iterator = 0 # Next position to write to
        self.full = False # Whether we are full
        self.buffer = np.array([(None, None, None, None, None, 0) for _ in range(max_size)])
        self.total_priority = 0
        self.steps_since_update = 0

    def update_totals(self):
        prios = [p for s,a,r,ns,d,p in self.buffer]
        self.total_priority = np.sum(prios)
        self.steps_since_update = 0

    def push(self, state, action, reward, next_state, done, priority=1):
        experience = (state, action, np.array([reward]), next_state, done, priority)

        # Remove previous prio, add new one
        self.total_priority -= self.buffer[self.iterator][5]
        self.total_priority += priority

        # Store experience
        self.buffer[self.iterator] = experience
        self.iterator += 1

        if(self.iterator >= self.max_size):
            self.iterator = 0
            self.full = True

    def update_priorities(self, indices, priorities):
        assert(len(indices) == len(priorities))
        for (i, p) in zip(indices, priorities):
            self.total_priority -= self.buffer[i][5]
            self.buffer[i][5] = p
            self.total_priority += p

    def get_buffer(self):
        return self.buffer

    def sample(self, batch_size):
        if(len(self.buffer) == 0):
            return [], [], [], [], [], []

        # Every once in a while, update total priorities (floating point inaccuracies)
        self.steps_since_update += 1
        if(self.steps_since_update > 20000):
            self.update_totals()

        # Sample according to priorities
        probs = [p/self.total_priority for s,a,r,ns,d,p in self.buffer[0:len(self)]]
        indices = np.random.choice(len(self), batch_size, p=probs)

        # Convert indices to lists of individual batches
        batch = self.buffer[indices]

        state_batch, action_batch, reward_batch, next_state_batch, done_batch, prios_batch = zip(*batch)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices)

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
        with open(file, "wb") as f:
            pickle.dump(self.buffer[0:len(self)], f)

    def load(self, file):
        try:
            with open(file, "rb") as f:
                self.buffer = pickle.load(f)

            if(len(self.buffer) < self.max_size):
                self.full = False
                self.iterator = len(self.buffer)

                self.buffer = np.concatenate((self.buffer, [(None, None, None, None, None, 0) for _ in range(len(self.buffer), self.max_size)]))
            else:
                self.full = True
                self.iterator = 0
            self.update_totals()

        except:
            print("No memory data found")

class QBladeLogger:

    def __init__(self, logdir, log_steps):
        self.logdir = logdir
        self.log_steps = log_steps
        self.writer = SummaryWriter(logdir + '/' + str(datetime.now()))

    def logObservation(self, step, observation, prefix='obs'):
        self.add_scalar('%s/rotational speed [rad/s]' % prefix, observation[0], step)
        self.add_scalar('%s/power [W]' % prefix, observation[1], step)
        self.add_scalar('%s/HH wind velocity [m/s]' % prefix, observation[2], step)
        self.add_scalar('%s/yaw angle [deg]' % prefix, observation[3], step)
        self.add_scalar('%s/pitch blade 1 [deg]' % prefix, observation[4], step)
        self.add_scalar('%s/pitch blade 2 [deg]' % prefix, observation[5], step)
        self.add_scalar('%s/pitch blade 3 [deg]' % prefix, observation[6], step)
        self.add_scalar('%s/tower top bending local x [Nm]' % prefix, observation[7], step)
        self.add_scalar('%s/tower top bending local y [Nm]' % prefix, observation[8], step)
        self.add_scalar('%s/tower top bending local z [Nm]' % prefix, observation[9], step)
        self.add_scalar('%s/oop bending blade 1 [Nm]' % prefix, observation[10], step)
        self.add_scalar('%s/oop bending blade 2 [Nm]' % prefix, observation[11], step)
        self.add_scalar('%s/oop bending blade 3 [Nm]' % prefix, observation[12], step)
        self.add_scalar('%s/ip bending blade 1 [Nm]' % prefix, observation[13], step)
        self.add_scalar('%s/ip bending blade 2 [Nm]' % prefix, observation[14], step)
        self.add_scalar('%s/ip bending blade 3 [Nm]' % prefix, observation[15], step)
        self.add_scalar('%s/oop tip deflection blade 1 [m]' % prefix, observation[16], step)
        self.add_scalar('%s/oop tip deflection blade 2 [m]' % prefix, observation[17], step)
        self.add_scalar('%s/oop tip deflection blade 3 [m]' % prefix, observation[18], step)
        self.add_scalar('%s/ip tip deflection blade 1 [m]' % prefix, observation[19], step)
        self.add_scalar('%s/ip tip deflection blade 2 [m]' % prefix, observation[20], step)
        self.add_scalar('%s/ip tip deflection blade 3 [m]' % prefix, observation[21], step)
        #self.add_scalar('%s/current time' % prefix, observation[22], step)

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