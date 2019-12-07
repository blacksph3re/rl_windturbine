import torch
import numpy as np

from .utils import BasicBuffer

class Normalizer:
  def __init__(self, obs_dim, act_dim, act_low, act_high, device, dtype=torch.float):
    self.device = device
    self.dtype = dtype
    self.obs_dim = obs_dim
    self.act_dim = act_dim
    self.act_high = np.array(act_high)
    self.act_low = np.array(act_low)

    # Calculate action normalization
    m = (self.act_low - self.act_high) / (-1 - 1)
    c = self.act_low - m * (-1)
    assert(not np.any(m == 0))
    self.action_normalizer_params = (m, c)
    self.action_normalizer_params_gpu = (torch.tensor(m, dtype=self.dtype).to(self.device),
                                         torch.tensor(c, dtype=self.dtype).to(self.device))

    # We just set the rest to identity projection
    self.state_normalizer_params = (np.ones(obs_dim),
                                    np.zeros(obs_dim))
    self.state_normalizer_params_gpu = (torch.tensor(np.ones(obs_dim), dtype=self.dtype).to(self.device), 
                                        torch.tensor(np.zeros(obs_dim), dtype=self.dtype).to(self.device))

    self.reward_normalizer_params = (np.ones(1),
                                     np.zeros(1))
    self.reward_normalizer_params_gpu = (torch.tensor(np.ones(1), dtype=self.dtype).to(self.device), 
                                         torch.tensor(np.zeros(1), dtype=self.dtype).to(self.device))

  def to_dict(self):
    return {
      "state_m": self.state_normalizer_params[0].tolist(),
      "state_c": self.state_normalizer_params[1].tolist(),
      "action_m": self.action_normalizer_params[0].tolist(),
      "action_c": self.action_normalizer_params[1].tolist(),
      "reward_m": self.reward_normalizer_params[0].tolist(),
      "reward_c": self.reward_normalizer_params[1].tolist(),
    }

  def from_dict(self, data):
    self.state_normalizer_params = (np.array(data["state_m"]), np.array(data["state_c"]))
    self.action_normalizer_params = (np.array(data["action_m"]), np.array(data["action_c"]))
    self.reward_normalizer_params = (np.array(data["reward_m"]), np.array(data["reward_c"]))

    self.state_normalizer_params_gpu =  (torch.tensor(data["state_m"], dtype=self.dtype).to(self.device),
                                         torch.tensor(data["state_c"], dtype=self.dtype).to(self.device))
    self.action_normalizer_params_gpu = (torch.tensor(data["action_m"], dtype=self.dtype).to(self.device),
                                         torch.tensor(data["action_c"], dtype=self.dtype).to(self.device))
    self.reward_normalizer_params_gpu = (torch.tensor(data["reward_m"], dtype=self.dtype).to(self.device),
                                         torch.tensor(data["reward_c"], dtype=self.dtype).to(self.device))

  def normalize_action(self, action, gpu=False):

    if(gpu):
      m, c = self.action_normalizer_params_gpu
      action = torch.clamp((action - c) / m, -3, 3)
    else:
      m, c = self.action_normalizer_params
      action = np.clip((action - c) / m, -3, 3) 
    
    return action

  def denormalize_action(self, action, gpu=False):
    m, c = self.action_normalizer_params_gpu if gpu else self.action_normalizer_params
    return action * m + c

  def normalize_state(self, state, gpu=False):

    if(gpu):
      m, c = self.state_normalizer_params_gpu
      state = torch.clamp((state - c) / m, -3, 3)
    else:
      m, c = self.state_normalizer_params
      state = np.clip((state - c) / m, -3, 3)
    
    return state

  def denormalize_state(self, state, gpu=False):
    m, c = self.state_normalizer_params_gpu if gpu else self.state_normalizer_params
    return state * m + c

  def normalize_reward(self, reward, gpu=False):

    if(gpu):
      m, c = self.reward_normalizer_params_gpu
      reward = torch.clamp((reward - c) / m, -3, 3) + 3
    else:
      m, c = self.reward_normalizer_params
      reward = np.clip((reward - c) / m, -3, 3) + 3
    
    return reward

  def denormalize_reward(self, reward, gpu=False):
    m, c = self.reward_normalizer_params_gpu if gpu else self.reward_normalizer_params
    return (reward - 3) * m + c

  def calc_normalizations(self, replay_buffer, normalization_extradata=None):
    # Load additional data if wanted
    if(normalization_extradata):
      tmpbuffer = BasicBuffer(100000, torch.device('cpu'), self.obs_dim, self.act_dim)
      tmpbuffer.load(normalization_extradata)
      extrastates = [state for state, _, _, _, _ in tmpbuffer.get_buffer()]
      extrarewards = [reward for _, _, reward, _, _ in tmpbuffer.get_buffer()]

      # Length of our observations and the observations in the data must match
      if(len(extrastates[0]) != self.obs_dim):
        print('Normalization extradata incompatible')
        extrastates = []


    # Calculate state max and min
    states = [state for state, _, _, _, _ in replay_buffer]

    if(normalization_extradata):
      states = states + extrastates
    
    state_max = np.quantile(states, 0.95, axis=0)
    state_min = np.quantile(states, 0.05, axis=0)

    state_max[(state_min - state_max) == 0] += 1e-6


    # Define a linear projection
    m = (state_min - state_max) / (-1 - 1)
    c = state_min - m * (-1)

    assert(not np.any(m == 0))

    # Store once as cpu and once as gpu variant
    self.state_normalizer_params = (m, c)
    self.state_normalizer_params_gpu = (torch.tensor(m, dtype=self.dtype).to(self.device),
                      torch.tensor(c, dtype=self.dtype).to(self.device))

    # Same for rewards
    rewards = [reward for _, _, reward, _, _ in replay_buffer]

    if(normalization_extradata):
      rewards = rewards + extrarewards

    reward_max = np.quantile(rewards, 0.95, axis=0)
    reward_min = np.quantile(rewards, 0.05, axis=0)

    reward_max[(reward_min - reward_max) == 0] += 1e-6

    m = (reward_min - reward_max) / (-1 - 1)
    c = reward_min - m * (-1)

    assert(not np.any(m == 0))

    self.reward_normalizer_params = (m, c)
    self.reward_normalizer_params_gpu = (torch.tensor(m, dtype=self.dtype).to(self.device),
                       torch.tensor(c, dtype=self.dtype).to(self.device))