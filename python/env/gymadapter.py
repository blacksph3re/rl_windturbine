import gym

class GymAdapter:
  def __init__(self, env_name):
    self.env = gym.make(env_name)

  def reset(self):
    self.observation = self.env.reset()
    return self.observation

  def get_obs_dim(self):
    return self.env.observation_space.shape[0]

  def get_act_dim(self):
    return self.env.action_space.shape[0]

  def get_act_limit(self):
    return self.env.action_space.high[0]

  def get_act_high(self):
    return self.env.action_space.high

  def get_act_low(self):
    return self.env.action_space.low

  def get_act_max_grad(self):
    return self.env.action_space.high - self.env.action_space.low

  def get_obs_labels(self):
    return {}

  def get_act_labels(self):
    return {}

  def step(self, action):
    self.observation, self.reward, self.death, _ = self.env.step(action)
    return self.observation, self.reward, self.death

  def render(self):
    return self.env.render()

  def close(self):
    return self.env.close()