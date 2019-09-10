# This is the main class, it controls the main loop and interfaces to whatever environment and learning I want to plug together
# An environment has an init routine and a step routine which takes control parameters and returns a reward
# A learner also has an init routine and a step routine, which takes a state and returns control parameters
# The learner takes a hyperparameter map

from ddpg.ddpg import DDPG
from gymadapter import GymAdapter
import tensorflow as tf


def main():

  print('huhu')

  # Set up the environment
  env = GymAdapter(tf.contrib.training.HParams(
    env_name='Walker2d-v3'
  ))

  # Get default hparams
  hparams = DDPG.get_default_hparams()

  # Set environment-specific hparams
  hparams.obs_dim = env.get_obs_dim()
  hparams.act_dim = env.get_act_dim()
  hparams.act_limit = env.get_act_limit()
  print(hparams)

  # Initialize the agent
  agent = DDPG(hparams)

  # Reset the environment
  o = env.reset()
  a = agent.prepare(o)

  total_reward = 0

  for t in range(0, hparams.steps_per_epoch * hparams.epochs):
    o, r, d = env.step(a)
    a, reset = agent.step(o, r, d)

    if(reset):
      o = env.reset()
      a = agent.reset_finalize(o)

  print('Testing results')
  o = env.reset()
  a = agent.reset_finalize(o)
  for t in range(0, hparams.test_steps):
    o, r, d = env.step(a)
    a = agent.get_action(o)
    env.render()


  env.close()


main()