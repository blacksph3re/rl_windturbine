# This is the main class, it controls the main loop and interfaces to whatever environment and learning I want to plug together
# An environment has an init routine and a step routine which takes control parameters and returns a reward
# A learner also has an init routine and a step routine, which takes a state and returns control parameters
# The learner takes a hyperparameter map

from ddpg_pytorch.ddpg import DDPG
from gymadapter import GymAdapter
from qbladeadapter import QBladeAdapter
from gym.wrappers import Monitor
import tensorflow as tf
import argparse
import os

def main():

  import argparse
  parser = argparse.ArgumentParser(description='QBlade RL controller training')
  parser.add_argument('--hparams', type=str,
                    help='Comma separated list of "name=value" pairs.')
  parser.add_argument('--load_checkpoint', type=str, help="checkpoint to load")
  args = parser.parse_args()

  # Set up the environment
  #env = GymAdapter(tf.contrib.training.HParams(
  #  env_name='Pendulum-v0'
  #))
  #recorder = Monitor(env.env, 'videos')

  env = QBladeAdapter()

  # Get default hparams
  hparams = DDPG.get_default_hparams()

  # Override arguments in command line
  #hparams.parse(args.hparams)

  # Set environment-specific hparams
  hparams.obs_dim = env.get_obs_dim()
  hparams.act_dim = env.get_act_dim()
  hparams.act_high = env.get_act_high()
  hparams.act_low = env.get_act_low()

  checkpoint_steps = hparams.checkpoint_steps
  checkpoint_dir = hparams.checkpoint_dir
  
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  print(hparams)

  # Initialize the agent
  agent = DDPG(hparams)

  # If we shall load a checkpoint, do so now
  if(args.load_checkpoint):
    prefix = ""
    with open(args.load_checkpoint + '/last', 'r') as f:
      prefix = f.read()
    print("Loading checkpoint from %s, prefix %s" % (args.load_checkpoint, prefix))
    agent.load_checkpoint(args.load_checkpoint, prefix)

  # Reset the environment
  o = env.reset()
  a = agent.prepare(o)

  total_reward = 0
  epoch_reward = 0

  for t in range(0, hparams.steps_per_epoch * hparams.epochs):
    o, r, d = env.step(a)

    env.logAction(agent.writer, t, a)
    env.logObservation(agent.writer, t, o)

    total_reward += r
    epoch_reward += r

    a, reset = agent.step(o, r, d)

    if(reset):
      o = env.reset()
      a = agent.reset_finalize(o)

    # do a checkpoint
    if(t>0 and t%checkpoint_steps == 0):
      print("Writing checkpoint to %s at step %d" % (checkpoint_dir, t))
      agent.save_checkpoint(checkpoint_dir, "step_%d_" % t)
      with open(checkpoint_dir + '/last', 'w') as f:
        f.write("step_%d_" % t)


  env.close()
  agent.close()


main()