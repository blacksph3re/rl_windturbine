import numpy as np
import tensorflow as tf
import time

from . import core
from .utils.logx import EpochLogger
from .utils.run_utils import setup_logger_kwargs


class ReplayBuffer:
  """
  A simple FIFO experience replay buffer for DDPG agents.
  """

  def __init__(self, obs_dim, act_dim, size):
    self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
    self.rews_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.float32)
    self.ptr, self.size, self.max_size = 0, 0, size

  def store(self, obs, act, rew, next_obs, done):
    self.obs1_buf[self.ptr] = obs
    self.obs2_buf[self.ptr] = next_obs
    self.acts_buf[self.ptr] = act
    self.rews_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    self.ptr = (self.ptr+1) % self.max_size
    self.size = min(self.size+1, self.max_size)

  def sample_batch(self, batch_size=32):
    idxs = np.random.randint(0, self.size, size=batch_size)
    return dict(obs1=self.obs1_buf[idxs],
          obs2=self.obs2_buf[idxs],
          acts=self.acts_buf[idxs],
          rews=self.rews_buf[idxs],
          done=self.done_buf[idxs])


class DDPG:
  """
  An implementation of DDPG, copied from spinningup's ddpg implementation
  """
  def __init__(self, hparams):
    self.hparams = hparams
    self.logger = EpochLogger(**setup_logger_kwargs(self.hparams.name, self.hparams.seed))
    #self.logger.save_config(dict(locals()))

    tf.set_random_seed(hparams.seed)
    np.random.seed(hparams.seed)

    # Inputs to computation graph
    self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(
      hparams.obs_dim, hparams.act_dim, hparams.obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
      self.pi, self.q, self.q_pi = core.mlp_actor_critic(self.x_ph, self.a_ph, self.hparams)
    
    # Target networks
    with tf.variable_scope('target'):
      # Note that the action placeholder going to actor_critic here is 
      # irrelevant, because we only need q_targ(s, pi_targ(s)).
      self.pi_targ, _, self.q_pi_targ = core.mlp_actor_critic(self.x2_ph, self.a_ph, self.hparams)

    # Experience buffer
    self.replay_buffer = ReplayBuffer(obs_dim=self.hparams.obs_dim, act_dim=self.hparams.act_dim, size=self.hparams.replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n'%var_counts)


    # Bellman backup for Q function
    self.backup = tf.stop_gradient(self.r_ph + self.hparams.gamma*(1-self.d_ph)*self.q_pi_targ)

    # DDPG losses
    self.pi_loss = -tf.reduce_mean(self.q_pi)
    self.q_loss = tf.reduce_mean((self.q-self.backup)**2)

    # Separate train ops for pi, q
    self.pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.hparams.pi_lr)
    self.q_optimizer = tf.train.AdamOptimizer(learning_rate=self.hparams.q_lr)
    self.train_pi_op = self.pi_optimizer.minimize(self.pi_loss, var_list=core.get_vars('main/pi'))
    self.train_q_op = self.q_optimizer.minimize(self.q_loss, var_list=core.get_vars('main/q'))

    # Polyak averaging for target variables

    self.target_update = tf.group([tf.assign(v_targ, self.hparams.polyak*v_targ + (1-self.hparams.polyak)*v_main)
                  for v_main, v_targ in zip(core.get_vars('main'), core.get_vars('target'))])

    # Initializing targets to match main variables
    self.target_init = tf.group([tf.assign(v_targ, v_main)
                  for v_main, v_targ in zip(core.get_vars('main'), core.get_vars('target'))])

    # Initialize tensorflow session
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    self.sess.run(self.target_init)

    # Setup model saving
    self.logger.setup_tf_saver(self.sess, inputs={'x': self.x_ph, 'a': self.a_ph}, outputs={'pi': self.pi, 'q': self.q})

  # Prepare for training run
  # Returns an initial random action from the action space
  def prepare(self, initial_observation):
    self.start_time = time.time()
    self.o, self.r, self.d, self.ep_ret, self.ep_len, self.t = initial_observation, 0, False, 0, 0, 0

    # Return an initial action (except for zero initial training steps, random)
    return self.get_action(initial_observation)

  # If a reset has happened, call this function before returning to stepping the controller
  def reset_finalize(self, observation):
    self.o, self.r, self.d, self.ep_ret, self.ep_len = observation, 0, False, 0, 0
    return self.get_action(observation)

  # Get an action from the main actor
  def get_action(self, observation):
    if self.t >= self.hparams.start_steps:
      self.a = self.sess.run(self.pi, feed_dict={self.x_ph: observation.reshape(1,-1)})[0]
      self.a += self.hparams.act_noise * np.random.randn(self.hparams.act_dim)
    else:
      self.a = np.random.uniform(-self.hparams.act_limit, self.hparams.act_limit, self.hparams.act_dim)
    
    return np.clip(self.a, -self.hparams.act_limit, self.hparams.act_limit)

  # Execute a training run
  # Returns an action from the action space, which is to be executed, and a reset signal
  # If the reset signal is True, return the environment to a halfway stable state
  # For the moment it could also be ignored
  def step(self, observation, reward, death):
    reset_signal = False

    self.ep_ret += reward
    self.ep_len += 1
    self.t += 1

    # Ignore the "done" signal if it comes from hitting the time
    # horizon (that is, when it's an artificial terminal signal
    # that isn't based on the agent's state)
    self.d = False if self.ep_len==self.hparams.max_ep_len else death

    # Store experience to replay buffer
    self.replay_buffer.store(self.o, self.a, reward, observation, death)

    # Update recent observation
    self.o = observation

    # Train the agent after one trajectory
    if self.d or (self.ep_len == self.hparams.max_ep_len):
      """
      Perform all DDPG updates at the end of the trajectory,
      in accordance with tuning done by TD3 paper authors.
      """
      for _ in range(self.ep_len):
        batch = self.replay_buffer.sample_batch(self.hparams.batch_size)
        feed_dict = {self.x_ph: batch['obs1'],
               self.x2_ph: batch['obs2'],
               self.a_ph: batch['acts'],
               self.r_ph: batch['rews'],
               self.d_ph: batch['done']
              }

        # Q-learning update
        outs = self.sess.run([self.q_loss, self.q, self.train_q_op], feed_dict)
        self.logger.store(LossQ=outs[0], QVals=outs[1])

        # Policy update
        outs = self.sess.run([self.pi_loss, self.train_pi_op, self.target_update], feed_dict)
        self.logger.store(LossPi=outs[0])

      self.logger.store(EpRet=self.ep_ret, EpLen=self.ep_len)
      self.r, self.d, self.ep_ret, self.ep_len = 0, False, 0, 0
      # At this place in the original code, the env was resetted
      # I would try to avoid that, as resetting the env is quite costly
      # However, mistakes can propagate from before...
      reset_signal = True

    # Check for end
    if self.t > 0 and self.t % self.hparams.steps_per_epoch == 0:
      epoch = self.t // self.hparams.steps_per_epoch

      # Save model
      if (epoch % self.hparams.save_freq == 0) or (epoch == self.hparams.epochs-1):
        self.logger.save_state({}, None)

      # Test the performance of the deterministic version of the agent.
      #test_agent()

      # Log info about epoch
      self.logger.log_tabular('Epoch', epoch)
      self.logger.log_tabular('EpRet', with_min_and_max=True)
      self.logger.log_tabular('EpLen', average_only=True)
      self.logger.log_tabular('TotalEnvInteracts', self.t)
      self.logger.log_tabular('QVals', with_min_and_max=True)
      self.logger.log_tabular('LossPi', average_only=True)
      self.logger.log_tabular('LossQ', average_only=True)
      self.logger.log_tabular('Time', time.time()-self.start_time)
      self.logger.dump_tabular()

    # Draw a new action
    self.a = self.get_action(observation)
    return self.a, reset_signal

  """

  Args:
    env_fn : A function which creates a copy of the environment.
      The environment must satisfy the OpenAI Gym API.

    actor_critic: A function which takes in placeholder symbols 
      for state, ``x_ph``, and action, ``a_ph``, and returns the main 
      outputs from the agent's Tensorflow computation graph:

      ===========  ================  ======================================
      Symbol       Shape             Description
      ===========  ================  ======================================
      ``pi``       (batch, act_dim)  | Deterministically computes actions
                       | from policy given states.
      ``q``        (batch,)          | Gives the current estimate of Q* for 
                       | states in ``x_ph`` and actions in
                       | ``a_ph``.
      ``q_pi``     (batch,)          | Gives the composition of ``q`` and 
                       | ``pi`` for states in ``x_ph``: 
                       | q(x, pi(x)).
      ===========  ================  ======================================

    ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
      function you provided to DDPG.

    seed (int): Seed for random number generators.

    steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
      for the agent and the environment in each epoch.

    epochs (int): Number of epochs to run and train agent.

    replay_size (int): Maximum length of replay buffer.

    gamma (float): Discount factor. (Always between 0 and 1.)

    polyak (float): Interpolation factor in polyak averaging for target 
      networks. Target networks are updated towards main networks 
      according to:

      .. math:: \\theta_{\\text{targ}} \\leftarrow 
        \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

      where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
      close to 1.)

    pi_lr (float): Learning rate for policy.

    q_lr (float): Learning rate for Q-networks.

    batch_size (int): Minibatch size for SGD.

    start_steps (int): Number of steps for uniform-random action selection,
      before running real policy. Helps exploration.

    act_noise (float): Stddev for Gaussian exploration noise added to 
      policy at training time. (At test time, no noise is added.)

    max_ep_len (int): Maximum length of trajectory / episode / rollout.

    logger_kwargs (dict): Keyword args for EpochLogger.

    save_freq (int): How often (in terms of gap between epochs) to save
      the current policy and value function.

    obs_dim (int): Dimension of the observation space

    act_dim (int): Dimension of the action space

    act_limit (float): Bound of the action space (same for all
      dimensions, upper bound=act_limit, lower bound=-act_limit)
  
    ac_hidden_size (int, int): Number of hidden layers in the actor critic

    ac_activation (activation): Activation function for the actor critic

    ac_output_activation (activation): Activation for the output layer in the actor critic

  """
  def get_default_hparams():
    return tf.contrib.training.HParams(
      hid=300,
      l=1,
      gamma=0.99,
      seed=0,
      epochs=10,
      name="ddpg",
      steps_per_epoch=5000,
      replay_size=int(1e6),
      polyak=0.995,
      pi_lr=1e-3,
      q_lr=1e-3,
      batch_size=100,
      start_steps=5000,
      act_noise=0.1,
      max_ep_len=1000,
      save_freq=1,
      obs_dim=20,
      act_dim=1,
      act_limit=1.0,
      ac_hidden_sizes=(400,300),
      ac_activation=tf.nn.relu,
      ac_output_activation=tf.tanh,
      test_steps=500,
      )
