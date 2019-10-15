import random
import numpy as np
import pickle
from collections import deque

# OU Noise with constant sigma
class OUNoise(object):
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
class OUNoiseDec(object):
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

class GaussianNoise(object):
    def __init__(self, dim, mean=0, variance=1):
        self.dim = dim
        self.mean = mean
        self.variance = variance

    def get_noise(self, _t=0):
        return np.random.randn(dim) * variance + mean


class UniformNoise(object):
    def __init__(self, dim, noise_factor):
        self.dim = dim
        self.noise_factor = noise_factor

    def get_noise(self, _t=0):
        return np.random.uniform(-np.ones(self.dim), np.ones(self.dim), self.dim) * noise_factor


class UniformNoiseDec(object):
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
class NoNoise(object):
    def __init__(self, dim, default_action=None):
        self.default_action = default_action if default_action is not None else np.zeros(self.dim)
        self.dim = dim

    def get_noise(self, _t=0):
        return np.zeros(dim)

class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def sample_sequence(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        min_start = len(self.buffer) - batch_size
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
        return len(self.buffer)

    def save(self, directory, prefix):
        with open("%s/%s_memory.dat" % (directory, prefix), "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, directory, prefix):
        try:
            with open("%s/%s_memory.dat" % (directory, prefix), "rb") as f:
                self.buffer = pickle.load(f)
        except:
            print("No memory data found")