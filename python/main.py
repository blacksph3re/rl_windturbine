# This is the main class, it controls the main loop and interfaces to whatever environment and learning I want to plug together
# An environment has an init routine and a step routine which takes control parameters and returns a reward
# A learner also has an init routine and a step routine, which takes a state and returns control parameters
# The learner takes a hyperparameter map

from ddpg.ddpg import DDPG
from gymadapter import GymAdapter
import tensorflow as tf

def __main__():

  # Set up the environment
  env = GymAdapter(tf.contrib.train.HParams(
    env_name='HalfCheetah-v2'
  ))

  # Get default hparams
  hparams = DDPG.get_default_hparams()

  # Set environment-specific hparams
  hparams.obs_dim = env.get_obs_dim()
  hparams.act_dim = env.get_act_dim()
  hparams.act_limit = env.get_act_limit()

  # Initialize the agent
  agent = DDPG(hparams)

  # Reset the environment
  o = env.reset()
  a = agent.prepare(o)

  for t in range(0, 10000):
    o, r, d = env.step(a)
    a, reset = agent.step(o, r, d)
    print("Step %d of %d" % (t, 10000))

    if(reset):
      o = env.reset()
      a = agent.reset_finalize(o)