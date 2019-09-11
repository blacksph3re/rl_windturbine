# This is the main class, it controls the main loop and interfaces to whatever environment and learning I want to plug together
# An environment has an init routine and a step routine which takes control parameters and returns a reward
# A learner also has an init routine and a step routine, which takes a state and returns control parameters
# The learner takes a hyperparameter map

from ddpg_pytorch.ddpg import DDPG
from gymadapter import GymAdapter
from gym.wrappers import Monitor
import tensorflow as tf


def main():

  # Set up the environment
  env = GymAdapter(tf.contrib.training.HParams(
    env_name='Pendulum-v0'
  ))
  #recorder = Monitor(env.env, 'videos')

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
  epoch_reward = 0

  for t in range(0, hparams.steps_per_epoch * hparams.epochs):
    o, r, d = env.step(a)

    total_reward += r
    epoch_reward += r

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
  agent.close()


main()