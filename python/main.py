# This is the main class, it controls the main loop and interfaces to whatever environment and learning I want to plug together
# An environment has an init routine and a step routine which takes control parameters and returns a reward
# A learner also has an init routine and a step routine, which takes a state and returns control parameters
# The learner takes a hyperparameter map

from ddpg.ddpg import DDPG
from env.gymadapter import GymAdapter
from env.qbladeadapter import QBladeAdapter
from datetime import datetime
import tensorflow as tf
import argparse
import os
import json
import time


def main():

  import argparse
  parser = argparse.ArgumentParser(description='QBlade RL controller training')
  parser.add_argument('--hparams', type=str,
                    help='Comma separated list of "name=value" pairs, overrides defaults and checkpoint hparams')
  parser.add_argument('--load_checkpoint', type=str, help="checkpoint to load")
  parser.add_argument('--load_checkpoint_hparams', action="store_true", help="also load hparams from checkpoint file")
  args = parser.parse_args()

  env = QBladeAdapter("../sample_projects/NREL_5MW_STR.wpa")
  #env = GymAdapter('Pendulum-v0')

  # Get default hparams
  hparams = DDPG.get_default_hparams()

  # Check if we should load hparams from checkpoint file
  if(args.load_checkpoint_hparams):
    assert(args.load_checkpoint)
    with open(args.load_checkpoint, 'r') as f:
      metadata = json.loads(f.read())
    print('parsing hparams from checkpoint: %s' % metadata['hparams'])
    hparams.override_from_dict(metadata['hparams'])

  # Override with arguments in command line
  if(args.hparams):
    hparams.parse(args.hparams)


  # Set environment-specific hparams
  hparams.obs_dim = env.get_obs_dim()
  hparams.act_dim = env.get_act_dim()
  hparams.act_high = env.get_act_high()
  hparams.act_low = env.get_act_low()
  hparams.act_max_grad = env.get_act_max_grad()
  hparams.act_labels = env.get_act_labels()
  hparams.obs_labels = env.get_obs_labels()

  checkpoint_steps = hparams.checkpoint_steps
  checkpoint_dir = hparams.checkpoint_dir
  
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  print(hparams)

  # Initialize the agent
  agent = DDPG(hparams)

  start_time = 0

  # If we shall load a checkpoint, do so now
  if(args.load_checkpoint):
    start_time = agent.load_checkpoint(args.load_checkpoint)
    print("Loaded checkpoint from %s, starting at step %d" % (args.load_checkpoint, start_time))
  # Reset the environment
  o = env.reset()
  a = agent.prepare(o)

  execution_time_env = 0
  execution_time_agent = 0

  for t in range(start_time, hparams.steps_per_epoch * hparams.epochs):
    # do a checkpoint if required
    if(t>0 and t%checkpoint_steps == 0):
      file = agent.save_checkpoint(checkpoint_dir)
      print("Checkpoint written to %s" % file)

    start = time.time()
    o, r, d = env.step(a)
    reset = d
    step = time.time()
    a, reset = agent.step(o, r, d)
    end = time.time()
    execution_time_env += step - start
    execution_time_agent += end - step

    if(t%300 == 0):
      print("Step %d, Execution times env: %f, agent: %f" % (t, execution_time_env, execution_time_agent))
      execution_time_agent = 0
      execution_time_env = 0

    if(reset):
      o = env.reset()
      print('Reset at step %d' % t)
      a = agent.reset_finalize(o)

  # Run a test run after training if wanted
  if(hparams.test_steps):
    print("Starting test run")
    o = env.reset()
    a = agent.prepare(o)
    total_reward = 0
    total_deaths = 0

    for t in range(0, hparams.test_steps):
      o, r, d = env.step(a)

      total_reward += r

      agent.logger.logAction(t, a, 'act_test')
      agent.logger.logObservation(t, o, 'obs_test')
      agent.logger.add_scalar('Reward/test', r, t)

      if(d):
        total_deaths += 1
        o = env.reset()

      a = agent.get_action(o)

    print("Test results:")
    print("Total reward: %d" % total_reward)
    print("Total deaths: %d" % total_deaths)

    if(hparams.test_output):
      with open(hparams.test_output, 'w') as f:
        f.write(json.dumps({
          "total_reward": total_reward,
          "total_deaths": total_deaths,
          "hparams": hparams.values()
        }))


  env.storeProject("checkpoints/sampleproject.wpa")
  env.close()
  agent.close()


main()