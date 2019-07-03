import gym

class GymAdapter:
  def __init__(self, hparams):
    self.env = gym.make(hparams.env_name)

  def reset(self):
    self.observation = self.env.reset()
    return self.observation

  def get_obs_dim(self):
    return self.env.observation_space.shape[0]

  def get_act_dim(self):
    return self.env.action_space.shape[0]

  def get_act_limit(self):
    return self.env.action_space.high[0]

  def step(self, action):
    self.observation, self.reward, self.death, _ = self.env.step(action)
    return self.observation, self.reward, self.death