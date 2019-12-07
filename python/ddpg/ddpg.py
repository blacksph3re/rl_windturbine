import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import os
import json
from datetime import datetime
from collections import deque

# For hparams
from .hparams import HParams
from .models import Critic, Actor
from .utils import *
from .normalizer import Normalizer

class DDPG:
  
  def __init__(self, hparams):
    self.device = torch.device("cuda" if torch.cuda.is_available() and hparams.use_gpu else "cpu")
    self.dtype = torch.float

    self.hparams = hparams
    self.obs_dim = hparams.obs_dim
    self.act_dim = hparams.act_dim
    self.act_high = np.array(hparams.act_high)
    self.act_low = np.array(hparams.act_low)
    self.batch_size = hparams.batch_size
    self.gamma = hparams.gamma
    self.tau = hparams.tau

    # If we should feed in past data, multiply obs dim by that
    if(self.hparams.feed_past):
      self.obs_dim *= self.hparams.feed_past + 1
      self.past_obs = CapacityBuffer(self.hparams.feed_past)

    # In case of action gradients, reduce action high and low and extend the observation space
    if(self.hparams.action_gradients):
      self.real_obs_dim = self.obs_dim
      self.obs_dim += self.act_dim
      self.real_act_high = self.act_high
      self.real_act_low = self.act_low
      self.act_high = np.ones(self.act_dim)
      self.act_low = -np.ones(self.act_dim)
      self.real_last_action = self.hparams.starting_action
      assert(len(self.real_last_action) == self.act_dim)
      self.action_grad_denormalizer = (self.real_act_high - self.real_act_low) * self.hparams.action_gradient_stepsize

      assert(np.all(self.real_last_action <= self.real_act_high))
      assert(np.all(self.real_last_action >= self.real_act_low))
      assert(not np.any(self.action_grad_denormalizer == 0))
    
    # initialize actor and critic networks
    self.critic = Critic(self.obs_dim, self.act_dim, hparams.critic_sizes[0], hparams.critic_sizes[1], hparams.init_weight_limit, hparams.critic_simple).to(self.device)
    self.critic_target = Critic(self.obs_dim, self.act_dim, hparams.critic_sizes[0], hparams.critic_sizes[1], hparams.init_weight_limit, hparams.critic_simple).to(self.device)
    
    self.actor = Actor(self.obs_dim, self.act_dim, hparams.actor_sizes[0], hparams.actor_sizes[1], hparams.init_weight_limit, hparams.actor_simple).to(self.device)
    self.actor_target = Actor(self.obs_dim, self.act_dim, hparams.actor_sizes[0], hparams.actor_sizes[1], hparams.init_weight_limit, hparams.actor_simple).to(self.device)

    # Copy target parameters
    for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
      target_param.data.copy_(param.data)
    for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
      target_param.data.copy_(param.data)

    # optimizers
    if(self.hparams.optimizer == 'adam'):
      self.optimizer = optim.Adam
    else:
      self.optimizer = optim.SGD
    self.critic_optimizer = self.optimizer(self.critic.parameters(), lr=hparams.critic_lr, weight_decay=hparams.critic_weight_decay)
    self.actor_optimizer  = self.optimizer(self.actor.parameters(), lr=hparams.actor_lr)

    # If in twin critic mode, add another critic
    if(self.hparams.twin_critics):
      self.critic2 = Critic(self.obs_dim, self.act_dim, hparams.critic_sizes[0], hparams.critic_sizes[1], hparams.init_weight_limit, hparams.critic_simple).to(self.device)
      self.critic2_target = Critic(self.obs_dim, self.act_dim, hparams.critic_sizes[0], hparams.critic_sizes[1], hparams.init_weight_limit, hparams.critic_simple).to(self.device)
      for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
        target_param.data.copy_(param.data)

      self.critic2_optimizer = self.optimizer(self.critic2.parameters(), lr=hparams.critic_lr)

    if(self.hparams.critic_loss == 'huber'):
      loss = torch.nn.SmoothL1Loss(reduction="none")
    else:
      loss = torch.nn.MSELoss(reduction="none")

    def generic_loss_function(target, pred, weights):
      td_error = loss(target, pred)
      return (td_error * weights).mean(), td_error

    self.critic_loss_function = lambda target, pred, weights: generic_loss_function(target, pred, weights)

    # Buffor for experience replay
    self.replay_buffer = BasicBuffer(hparams.buffer_maxlen, self.device, self.obs_dim, self.act_dim, hparams.prioritized_experience_replay_alpha, self.dtype)

    # Noises
    # Random exploration
    if(self.hparams.random_exploration_type == 'correlated'):
      assert(len(self.hparams.random_exploration_mu) == self.act_dim)
      self.random_exploration_noise = OUNoise(
        self.act_dim,
        self.hparams.random_exploration_theta,
        self.hparams.random_exploration_sigma,
        self.hparams.random_exploration_mu)
    elif(self.hparams.random_exploration_type == 'uncorrelated'):
      self.random_exploration_noise = GaussianNoise(
        self.act_dim,
        self.hparams.random_exploration_mean,
        self.hparams.random_exploration_variance)
    else:
      self.random_exploration_noise = NoNoise(
        self.act_dim)

    # Action noise
    if(self.hparams.action_noise_type == 'correlated'):
      self.action_noise = OUNoise(
        self.act_dim,
        self.hparams.action_noise_theta,
        self.hparams.action_noise_sigma)
    elif(self.hparams.action_noise_type == 'correlated-decreasing'):
      self.action_noise = OUNoiseDec(
        self.act_dim,
        self.hparams.action_noise_theta,
        self.hparams.action_noise_sigma,
        self.hparams.random_exploration_steps,
        self.hparams.action_noise_sigma_decayed,
        self.hparams.action_noise_decay_steps)
    elif(self.hparams.action_noise_type == 'uncorrelated'):
      self.action_noise = UniformNoise(
        self.act_dim,
        self.hparams.action_noise_sigma)
    elif(self.hparams.action_noise_type == 'uncorrelated-decreasing'):
      self.action_noise = UniformNoiseDec(
        self.act_dim,
        self.hparams.action_noise_sigma,
        self.hparams.random_exploration_steps,
        self.hparams.action_noise_sigma_decayed,
        self.hparams.action_noise_decay_steps)
    else:
      self.action_noise = NoNoise(
        self.act_dim)

    # We can already calc action normalization based on parameter space limits
    self.normalizer = Normalizer(self.obs_dim,
                                 self.act_dim,
                                 self.hparams.act_low,
                                 self.hparams.act_high,
                                 self.device,
                                 self.dtype)

    self.logger = QBladeLogger(self.hparams.logdir, self.hparams.log_steps, self.hparams.run_name, self.hparams.obs_labels, self.hparams.act_labels)
    for key, value in self.hparams.values().items():
      self.logger.writer.add_text(key, str(value), 0)

    self.time = 0
    self.last_state = np.zeros(self.obs_dim)
    self.last_action = np.zeros(self.act_dim)
    self.epoch_reward = 0
    self.killcount = 0
    self.epoch_killcount = 0

  def clip_action(self, action):
    if(self.hparams.clip_action_gradients):
      max_threshold = np.array(self.hparams.act_max_grad)
      action_grads = action - self.last_action

      # Limit higher values
      indices = action_grads > max_threshold
      action[indices] = self.last_action[indices] + max_threshold[indices]
    
      # Limit smaller values
      indices = action_grads < -max_threshold
      action[indices] = self.last_action[indices] - max_threshold[indices]

    action = np.clip(action, self.act_low, self.act_high)
    return action


  def get_action(self, obs, add_noise=False):
    # Send the observation to the device, normalize it
    # Then calculate the action and denormalize it
    assert(not np.any(np.isnan(obs)))
    state = torch.tensor(obs, dtype=self.dtype).unsqueeze(0).to(self.device)

    if (self.hparams.normalize_observations):
      state = self.normalizer.normalize_state(state, True)
    
    # Run the network
    action = self.actor.forward(state)

    # Add noise
    if(add_noise):
      noise = torch.tensor(self.action_noise.get_noise(self.time), dtype=self.dtype).to(self.device)
      action = action + noise
    
    if(self.hparams.normalize_actions):
      action = self.normalizer.denormalize_action(action, True)
    
    # Get it to the cpu
    action = action.squeeze(0).cpu().detach().numpy()

    if(np.any(np.isnan(action))):
      print([(x, y) for (x, y) in self.actor.cpu().named_parameters()])
      assert(False)

    self.logger.logAction(self.time, action, 'act-unclipped')

    action = self.clip_action(action)

    assert(not np.any(np.isnan(action)))

    return action

  # Returns an unnormalized expert action from the expert controller
  def get_expert_action(self, state):
    if(self.act_dim == 2):
      return [2e7*state[0]-12e6,128.6*state[0]-128.6]
    else:
      return np.zeros(self.act_dim)

  def prepare(self, state):
    assert(not np.any(np.isnan(state)))

    self.last_action = np.zeros(self.act_dim)
    state = self.feed_past_obs(state)
    if(self.hparams.action_gradients):
      return self.real_last_action
    action = self.get_action(state, False)
    self.last_action = action
    self.last_state = state
    return action

  def reset_finalize(self, obs):
    action = self.prepare(obs)
    return action
  
  def feed_past_obs(self, state):
    # Append last observations to observation if using past feeding
    if(self.hparams.feed_past):
      # Check if past feeding is long enough already
      if(len(self.past_obs) < self.hparams.feed_past):
        self.reset_past_obs(state)

      state = np.concatenate([state, *self.past_obs.get_buffer()])

      self.past_obs.append(state)

    assert(len(state) == self.obs_dim)
    return state

  def pretrain_policy(self):
    optimizer = self.optimizer(self.actor.parameters(), lr=self.hparams.pretrain_policy_lr)
    validation_size = int(len(self.replay_buffer) / 0.03)
    validation_indices = np.random.choice(len(self.replay_buffer), validation_size)
    self.replay_buffer.update_priorities(validation_indices, np.ones(validation_size)*1e-10)

    validation_states, _, _, _, _ = zip(*self.replay_buffer.get_buffer()[validation_indices])

    validation_actions = [self.get_expert_action(state) for state in validation_states]

    validation_states = torch.tensor(validation_states, dtype=self.dtype).to(self.device)
    validation_actions = torch.tensor(validation_actions, dtype=self.dtype).to(self.device)

    if(self.hparams.normalize_observations):
      validation_states = self.normalizer.normalize_state(validation_states, True)

    if(self.hparams.normalize_actions):
      validation_actions = self.normalizer.normalize_action(validation_actions, True)

    for i in range(0, self.hparams.pretrain_policy_steps):
      state_batch, _, _, _, _, _ = self.replay_buffer.sample(self.hparams.pretrain_policy_batch_size)

      action_batch = [self.get_expert_action(state) for state in state_batch]
      action_batch = torch.tensor(action_batch, dtype=self.dtype).to(self.device)
      
      if(self.hparams.normalize_observations):
        state_batch = self.normalizer.normalize_state(state_batch, True)

      if(self.hparams.normalize_actions):
        action_batch = self.normalizer.normalize_action(action_batch, True)

      optimizer.zero_grad()
      prediction = self.actor.forward(state_batch)
      loss = F.mse_loss(prediction, action_batch)
      loss.backward()
      optimizer.step()

      validation_loss = F.mse_loss(self.actor.forward(validation_states), validation_actions)

      self.logger.add_scalar('Loss/pretrain', loss, i)
      self.logger.add_scalar('Loss/pretrain_val', validation_loss, i)

    self.replay_buffer.update_priorities(validation_indices, np.ones(validation_size) * self.replay_buffer.max_priority)

    for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
      target_param.data.copy_(param.data)

  def sample(self, batch_size, uniform=False):

    state_batch, action_batch, reward_batch, next_state_batch, masks, indices = self.replay_buffer.sample(batch_size, uniform)

    masks = 1 - masks

    # Add replay noise if desired
    if(self.hparams.replay_noise):
      state_noise = [np.random.uniform(-np.ones(self.obs_dim), np.ones(self.obs_dim), self.obs_dim) for i in range(0, batch_size)]
      state_noise = torch.tensor(state_noise, dtype=self.dtype).to(self.device)
      state_noise = self.normalizer.denormalize_state(state_noise, True)
      action_noise = [np.random.uniform(-np.ones(self.act_dim), np.ones(self.act_dim), self.act_dim) for i in range(0, batch_size)]
      action_noise = torch.tensor(action_noise, dtype=self.dtype).to(self.device)
      action_noise = self.normalizer.denormalize_action(action_noise, True)
      next_state_noise = [np.random.uniform(-np.ones(self.obs_dim), np.ones(self.obs_dim), self.obs_dim) for i in range(0, batch_size)]
      next_state_noise = torch.tensor(next_state_noise, dtype=self.dtype).to(self.device)
      next_state_noise = self.normalizer.denormalize_state(next_state_noise, True)

      state_batch = state_batch * (1 - self.hparams.replay_noise) + state_noise * self.hparams.replay_noise
      action_batch = action_batch * (1 - self.hparams.replay_noise) + action_noise * self.hparams.replay_noise
      next_state_batch = next_state_batch * (1 - self.hparams.replay_noise) + next_state_noise * self.hparams.replay_noise

    # Normalize all states and actions if desired
    if(self.hparams.normalize_observations):
      state_batch = self.normalizer.normalize_state(state_batch, True)
      next_state_batch = self.normalizer.normalize_state(next_state_batch, True)

    if(self.hparams.normalize_rewards):
      reward_batch = self.normalizer.normalize_reward(reward_batch, True)

    if(self.hparams.normalize_actions):
      action_batch = self.normalizer.normalize_action(action_batch, True)

    return (state_batch, action_batch, reward_batch, next_state_batch, masks, indices)

  def update(self, batch_size, time = None):
    time = time or self.time

    
    (state_batch, action_batch, reward_batch, next_state_batch, masks, indices) = self.sample(batch_size, False)


    # When using PER, calculate IS-weights
    if(self.hparams.prioritized_experience_replay):
      weights = self.replay_buffer.get_probabilities(indices)
      weights = (weights*batch_size) ** (-self.hparams.prioritized_experience_replay_beta)
      weights = weights / np.max(weights)
      assert(not np.any(np.isnan(weights)))
      assert(not np.any(weights > 1))
      weights = torch.tensor(weights, dtype=self.dtype).to(self.device)
    else:
      weights = torch.ones(batch_size, dtype=self.dtype, device=self.device)

    # Run the critic
    next_actions = self.actor_target.forward(next_state_batch).detach()
    # If using target policy smoothing, add a bit of noise to next actions
    if(self.hparams.target_policy_smoothing):
      sigma = torch.ones_like(next_actions) * self.hparams.target_policy_smoothing_sigma
      mu = torch.zeros_like(next_actions)

      next_actions += torch.clamp(torch.normal(mu, sigma), -self.hparams.target_policy_smoothing_clip, self.hparams.target_policy_smoothing_clip)
      next_actions = torch.clamp(next_actions, -1, 1)

    next_Q = self.critic_target.forward(next_state_batch, next_actions).detach()
    expected_Q = reward_batch + self.gamma * next_Q * masks
    
    # If using twin q, take minimum of the two next_Q estimations and replace that with the current one
    if(self.hparams.twin_critics):
      
      next_Q2 = self.critic2_target.forward(next_state_batch, next_actions)
      expected_Q = reward_batch + self.gamma * torch.min(next_Q, next_Q2) * masks

      self.critic2_optimizer.zero_grad()
      curr_Q2 = self.critic2.forward(state_batch, action_batch)
      q2_loss, td_error2 = self.critic_loss_function(curr_Q2, expected_Q.detach(), weights.detach())
      q2_loss.backward()

      if(self.hparams.clip_gradients):
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.hparams.clip_gradients)

      self.critic2_optimizer.step()
      self.logger.add_scalar('Loss/q2', q2_loss.detach(), time)
      self.logger.add_scalar('Loss/next-q2-mean', next_Q2.mean().detach().cpu(), time)


    # update critic
    self.critic_optimizer.zero_grad()
    curr_Q = self.critic.forward(state_batch, action_batch)
    q_loss, td_error = self.critic_loss_function(curr_Q, expected_Q.detach(), weights.detach())
    q_loss.backward() 

    if(self.hparams.clip_gradients):
      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hparams.clip_gradients)

    self.critic_optimizer.step()
    self.logger.add_scalar('Loss/q', q_loss.detach(), time)
    self.logger.add_scalar('Loss/next_Q_mean', next_Q.mean().detach(), time)

    # Also, for correction of oversampling in later policy replays, redraw samples
    if(self.hparams.prioritized_experience_replay):
      (state_batch, action_batch, reward_batch, next_state_batch, masks, indices) = self.sample(batch_size, True)
      

    # update actor
    policy_loss = 0
    if(time%self.hparams.actor_delay == 0):
      self.actor_optimizer.zero_grad()
      policy_loss = -torch.mean(self.critic.forward(state_batch, self.actor.forward(state_batch)))

      policy_loss.backward()

      self.actor_optimizer.step()
      self.logger.add_scalar('Loss/policy', policy_loss.detach(), time)


    # Update priorities if using prioritized experience replay
    if(self.hparams.prioritized_experience_replay):
      # In twin critic mode, take the max value of the two, 
      # Thus, if any of the two nets is performing badly on that sample,
      # we'll sample it more often.
      if(self.hparams.twin_critics):
        td_error = td_error + td_error2

      td_error = td_error.detach().abs().cpu().flatten()
      td_error = torch.add(td_error, torch.tensor(np.ones(batch_size) * 1e-6, dtype=self.dtype))

      self.replay_buffer.update_priorities(indices, td_error.data.numpy())


    # update target networks 
    for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
      target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
     
    for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
      target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    # Add actor parameter noise
    if(self.hparams.parameter_noise != 0):
      for param in self.actor.parameters():
        param.data.copy_(param.data + self.hparams.parameter_noise * np.random.uniform(-1, 1))

    # Log everything if needed
    if(self.hparams.log_net_insights and self.time % self.hparams.log_net_insights == 0):
      critic_state = dict(self.critic.cpu().named_parameters())

      self.logger.add_histogram_nofilter('critic-weights/linear1', critic_state['linear1.weight'], time)
      self.logger.add_scalar_nofilter('critic/l1-dead', np.sum(critic_state['linear1.weight'].data.numpy() == 0), time)
      self.logger.add_histogram_nofilter('critic-grads/linear1', np.log10(np.abs(critic_state['linear1.weight'].grad)+1e-20), time)
      self.logger.add_scalar_nofilter('critic/l1-grad-exp', np.sum(critic_state['linear1.weight'].grad.numpy() > 1e10), time)
      self.logger.add_scalar_nofilter('critic/l1-vanishing-grad', np.sum(critic_state['linear1.weight'].grad.numpy() == 0), time)

      self.logger.add_histogram_nofilter('critic-weights/linear2', critic_state['linear2.weight'], time)
      self.logger.add_scalar_nofilter('critic/l2-dead', np.sum(critic_state['linear2.weight'].data.numpy() == 0), time)
      self.logger.add_histogram_nofilter('critic-grads/linear2', np.log10(np.abs(critic_state['linear2.weight'].grad)+1e-20), time)
      self.logger.add_scalar_nofilter('critic/l2-grad-exp', np.sum(critic_state['linear2.weight'].grad.numpy() > 1e10), time)
      self.logger.add_scalar_nofilter('critic/l2-vanishing-grad', np.sum(critic_state['linear2.weight'].grad.numpy() == 0), time)

      if('linear3.weight' in critic_state):
        self.logger.add_histogram_nofilter('critic-weights/linear3', critic_state['linear3.weight'], time)
        self.logger.add_scalar_nofilter('critic/l3-dead', np.sum(critic_state['linear3.weight'].data.numpy() == 0), time)
        self.logger.add_histogram_nofilter('critic-grads/linear3', np.log10(np.abs(critic_state['linear3.weight'].grad)+1e-20), time)
        self.logger.add_scalar_nofilter('critic/l3-grad-exp', np.sum(critic_state['linear3.weight'].grad.numpy() > 1e10), time)
        self.logger.add_scalar_nofilter('critic/l3-vanishing-grad', np.sum(critic_state['linear3.weight'].grad.numpy() == 0), time)

      self.critic.to(self.device)

      self.logger.add_histogram_nofilter('priorities', self.replay_buffer.get_priorities(), time)

      actor_state = dict(self.actor.cpu().named_parameters())

      self.logger.add_scalar_nofilter('actor/l1-dead', np.sum(actor_state['linear1.weight'].data.numpy() == 0), time)
      self.logger.add_scalar_nofilter('actor/l1-grad-exp', np.sum(actor_state['linear1.weight'].grad.numpy() > 1e10), time)
      self.logger.add_scalar_nofilter('actor/l1-vanishing-grad', np.sum(actor_state['linear1.weight'].grad.numpy() == 0), time)
      self.logger.add_histogram_nofilter('actor-weights/linear1', actor_state['linear1.weight'], time)
      self.logger.add_histogram_nofilter('actor-grads/linear1', np.log10(np.abs(actor_state['linear1.weight'].grad)+1e-20), time)

      self.logger.add_scalar_nofilter('actor/l2-dead', np.sum(actor_state['linear2.weight'].data.numpy() == 0), time)
      self.logger.add_scalar_nofilter('actor/l2-grad-exp', np.sum(actor_state['linear2.weight'].grad.numpy() > 1e10), time)
      self.logger.add_scalar_nofilter('actor/l2-vanishing-grad', np.sum(actor_state['linear2.weight'].grad.numpy() == 0), time)
      self.logger.add_histogram_nofilter('actor-weights/linear2', actor_state['linear2.weight'], time)
      self.logger.add_histogram_nofilter('actor-grads/linear2', np.log10(np.abs(actor_state['linear2.weight'].grad)+1e-20), time)

      if('linear3.weight' in actor_state):
        self.logger.add_scalar_nofilter('actor/l3-dead', np.sum(actor_state['linear3.weight'].data.numpy() == 0), time)
        self.logger.add_scalar_nofilter('actor/l3-grad-exp', np.sum(actor_state['linear3.weight'].grad.numpy() > 1e10), time)
        self.logger.add_scalar_nofilter('actor/l3-vanishing-grad', np.sum(actor_state['linear3.weight'].grad.numpy() == 0), time)
        self.logger.add_histogram_nofilter('actor-weights/linear3', actor_state['linear3.weight'], time)
        self.logger.add_histogram_nofilter('actor-grads/linear3', np.log10(np.abs(actor_state['linear3.weight'].grad)+1e-20), time)
      self.actor.to(self.device)


  def reset_past_obs(self, cur_state):
    if(self.hparams.feed_past):
      self.past_obs.clear()
      for i in range(0, self.hparams.feed_past):
        self.past_obs.append(cur_state)

  def step(self, state, reward, done):

    assert(not np.any(np.isnan(state)))
    assert(not np.isnan(reward))

    state = self.feed_past_obs(state)

    # Append last action to observation if using action gradients
    if(self.hparams.action_gradients):
      state = np.concatenate([state, self.real_last_action])

    self.replay_buffer.push(self.last_state, self.last_action, reward, state, done)

    self.epoch_reward += reward
    epoch_end = self.time > 0 and self.time % self.hparams.steps_per_epoch == 0
    if(done):
      self.killcount += 1
      self.epoch_killcount += 1
      if(self.hparams.feed_past):
        self.past_obs.clear()

    if (self.time == self.hparams.random_exploration_steps):
      print("Random exploration finished, preparing normal run")
      self.normalizer.calc_normalizations(self.replay_buffer.get_buffer(), self.hparams.normalization_extradata)

      # If wanted, pretrain the policy to actions from random exploration
      if(self.hparams.pretrain_policy):
        print('Pretraining the policy for %d steps' % self.hparams.pretrain_policy_steps)
        self.pretrain_policy()

      # do the learning which we missed up on during random exploration
      print('Training on random exploration data for %d steps' % self.hparams.random_exploration_training_steps)
      for i in range(0, self.hparams.random_exploration_training_steps):
        self.update(self.batch_size, i)

      # set all priorities in the replay buffer to maximum, so everything will get sampled (those with prio 1 most likely won't get sampled)
      self.replay_buffer.update_priorities(range(0, len(self.replay_buffer)), 
                         np.ones(len(self.replay_buffer)) * self.replay_buffer.max_priority)

      print('Done, starting normal run')

    if (epoch_end):
      epoch = (self.time // self.hparams.steps_per_epoch)
      print('Epoch %d' % epoch)
      print('Epoch reward %d' % self.epoch_reward)
      print('Step %d' % self.time)
      self.logger.add_scalar_nofilter('Epoch/reward', self.epoch_reward, epoch)
      self.logger.add_scalar_nofilter('Epoch/killcount', self.epoch_killcount, epoch)

      self.epoch_reward = 0
      self.epoch_killcount = 0

    if (self.time > self.hparams.random_exploration_steps and len(self.replay_buffer) > self.batch_size):
      for i in range(0, self.hparams.training_steps_per_env_iteration):
        self.update(self.batch_size, self.time + i/self.hparams.training_steps_per_env_iteration)

    if(self.time < self.hparams.random_exploration_steps):
      action = self.get_expert_action(state)
      action += self.normalizer.denormalize_action(self.random_exploration_noise.get_noise(self.time), False)
      action = self.clip_action(action)
    else:
      action = self.get_action(state, True)

    self.last_action = action
    self.last_state = state

    if(self.hparams.action_gradients):
      self.real_last_action += action * self.action_grad_denormalizer
      self.real_last_action = np.clip(self.real_last_action, self.real_act_low, self.real_act_high)

      self.logger.logAction(self.time, action, 'act_grad')
      action = self.real_last_action

    self.logger.logAction(self.time, action)
    self.logger.logObservation(self.time, state)
    self.logger.add_scalar('Reward/real', reward, self.time)

    if(self.time > self.hparams.random_exploration_steps):
      self.logger.logAction(self.time, self.normalizer.normalize_action(action), 'act_norm')
      self.logger.logObservation(self.time, self.normalizer.normalize_state(state), 'obs_norm')
      self.logger.add_scalar('Reward/norm', self.normalizer.normalize_reward([reward], False), self.time)


    self.time = self.time + 1

    return action, done or (epoch_end and self.hparams.reset_after_epoch)

  def close(self):
    self.logger.close()

  def save_checkpoint(self, directory):
    prefix = 'step_%d_' % self.time

    # Save states
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, prefix))
    torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, prefix))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, prefix))
    torch.save(self.critic_target.state_dict(), '%s/%s_critic_target.pth' % (directory, prefix))
    self.replay_buffer.save("%s/%s_memory.dat" % (directory, prefix))

    # And metadata
    metadata = {
      "step": self.time,
      "prefix": prefix,
      "time": str(datetime.now()),
      "killcount": self.killcount,
      "hparams": str(self.hparams.get_dict()),
      "normalizer": self.normalizer.to_dict(),
      "real_last_action": self.real_last_action.tolist() if self.hparams.action_gradients else None,
    }
    with open('%s/%s_metadata' % (directory, prefix), 'w') as f:
      f.write(json.dumps(metadata, indent=2))

    return '%s/%s_metadata' % (directory, prefix)

  def load_checkpoint(self, metadata_file):
    # Load metadata
    with open(metadata_file, 'r') as f:
      metadata = json.loads(f.read())
    directory = os.path.dirname(metadata_file)
    prefix = metadata['prefix']
    self.time = metadata['step'] + 1 # +1 to avoid saving right after loading
    if('real_last_action' in metadata):
      self.real_last_action = np.array(metadata['real_last_action'])
    if('killcount' in metadata):
      self.killcount = metadata['killcount']
    if('normalizer' in metadata):
      self.normalizer.from_dict(metadata['normalizer'])

    # Load model state
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, prefix)))
    self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, prefix)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, prefix)))
    self.critic_target.load_state_dict(torch.load('%s/%s_critic_target.pth' % (directory, prefix)))
    self.replay_buffer.load("%s/%s_memory.dat" % (directory, prefix))

    # The main loop needs to know which timestep to start with
    return self.time


  def get_default_hparams():
    return HParams(
      # The name of the run to make identification easier
      run_name = '',

      # Whether to use CUDA GPUs for acceleration
      use_gpu = True,

      # Number of steps of interaction (state-action pairs) 
      # for the agent and the environment in each epoch.
      steps_per_epoch = 2000,

      # Whether to reset the environment after an epoch
      reset_after_epoch = True,

      # Number of steps to sample random actions
      # before starting to utilize the policy
      random_exploration_steps = 0,

      # How many steps to train the agent after random exploration
      random_exploration_training_steps = 0,

      # Type of random exploration noise
      # 'correlated' (OU Noise), 'uncorrelated' (gaussian) or 'none'
      random_exploration_type = "none",

      # How strongly it is drawn towards mu
      random_exploration_theta = 0.03,

      # How strongly it wanders around
      random_exploration_sigma = 0.02,

      # The default action to start with in random exploration
      # Also where most of the exploration will happen around
      # -1 being minimum action and 1 maximum
      # None means np.zeros(act_dim) as mu
      # If gradient actions are active, this is in gradient action space
      random_exploration_mu = [-1, -1],

      # Number of steps after which to write out a checkpoint
      checkpoint_steps = 10000,

      # Where to store the checkpoints
      checkpoint_dir = "checkpoints",

      # Number of total epochs to run the training
      epochs = 150,

      # Number of steps to run after the training, testing the policy
      test_steps = 50000,

      # Whether, and where to write test results to
      test_output = 'test_output',

      # Batch size for experience replay
      # The bigger the less it will overfit to specific samples
      # Also the bigger, the less likely vanishing gradients appear
      batch_size = 128,

      # The discounting factor with which experiences in the future are regarded
      # less than experiences now. The higher, the further into the future the value function
      # will look but also the less stable it gets
      gamma = 0.99,

      # The speed in which the target network follows the main network
      # Higher means faster learning but also more likely to fall into local
      # Optima
      tau = 0.005,

      # The maximum size of the replay buffer, i.e. how many steps to store as
      # experience
      buffer_maxlen = 100000,

      # Learning rate of the Q approximator
      critic_lr = 1e-5,

      # Neural network sizes of the critic
      critic_sizes = [128, 64],

      # Whether to use a 4-layer or a 2-layer structure
      # for the critic
      critic_simple = False,

      # Loss function for the critic
      # default is mean square error
      # Other values: "mse", "huber"
      critic_loss = 'huber',

      # Whether to use L2 regularization
      critic_weight_decay = 0.01,

      # Whether to use duelling critics
      # In this variant, two critics try to estimate q
      # and only the lower estimation will be chosen
      # Helps prevent loss explosions and overestimation
      twin_critics = False,

      # Learning rate of the policy
      actor_lr = 1e-5,

      # Network sizes of the policy
      actor_sizes = [32, 16],

      # Whether to use a 2-layer or a 3-layer structure for the actor
      actor_simple = False,

      # Only update the actor every n critic updates
      # 1 for updating every critic update
      # From TD3
      actor_delay = 1,

      # Whether to use target policy smoothing
      # In this, a bit of noise is added to target actions
      # From TD3
      target_policy_smoothing = False,

      # How strongly to apply target policy smoothing
      target_policy_smoothing_sigma = 0.01,

      # Where to clip target policy smoothing noise
      target_policy_smoothing_clip = 0.03,

      # Which optimizer to use for everything
      # adam or sgd
      optimizer = 'adam',

      # How many training steps to do per environment iteration
      training_steps_per_env_iteration = 1,

      # Whether to clip gradients
      # If yes, to which values to clip them to
      clip_gradients = 1.0,

      # Whether to use batch normalization in both the actor and critic
      batch_normalization = True,

      # Network parameters of both the actor and critic will be initialized to
      # a uniform random value between [-init_weight_limit, init_weight_limit]
      init_weight_limit = 0.5,

      # Stuff that will be overwritten by the environment
      obs_dim = 0,
      act_dim = 0,
      act_high = [0.],
      act_low = [0.],
      act_max_grad = [0.],
      act_labels = {},
      obs_labels = {},

      # Where to store tensorboard logs
      logdir = "logs",

      # Log data every n steps
      log_steps = 1,

      # Log net insight histograms every n steps
      # -1 for disabling completely
      log_net_insights = 1000,

      # Action noise
      # The type of action noise can be either
      # 'correlated' OU Noise
      # 'correlated-decreasing' OU Noise that decreases over time
      # 'uncorrelated' Uniform noise
      # 'uncorrelated-decreasing' Uniform noise that decreases over time
      # 'none' No noise
      action_noise_type = 'correlated-decreasing',

      # For correlated noise the sigma (intensity of random movement)
      # For uncorrelated noise the noise_level factor
      # 1 results in noise across the whole action space
      action_noise_sigma = 0.1,

      # For decreasing noises this is the minimum sigma level
      action_noise_sigma_decayed = 0.00002,

      # For correlated noises, how much the noise stays around the policy action
      action_noise_theta = 0.03,

      # For decreasing noise, after how many iterations the noise will be reduced to
      # action_noise_sigma_decayed
      # It will come closer gradually
      action_noise_decay_steps = 50000,

      # Adding a uniform noise to the policy parameters helped facilitate exploration
      parameter_noise = False,

      # Adding noise to experience replay helps to prevent overfitting
      # Requires action and observation normalization
      replay_noise = 1e-5,

      # Action normalization should always be enabled when the action space is not already between -1 and 1.
      # Otherwise noise will not make sense
      normalize_actions = True,

      # Whether to normalize observations
      # Normalization factors will be computed after observing some inputs, i.e.
      # after the random exploration phase
      # After that, observations will mostly be between -1 and 1 internally
      # though min and max from the random exploration phase is assumed
      normalize_observations = False,

      # Whether to normalize rewards
      # Normalization factor will be computed after observing some rewards, i.e.
      # after the random exploration phase
      # After that, rewards will mostly be between -1 and 1 internally
      # Though min and max from the random exploration phase is assumed
      normalize_rewards = False,

      # Additional nrmalization replay data to load when
      # calculating normalizations
      # Set to none if you don't want to load extra data
      #normalization_extradata = 'normalization_data/past_feeding_1.dat',
      normalization_extradata = 'normalization_data/two_observations_hold_power.dat',
      #normalization_extradata = None,

      # Whether to offer gradient action spaces instead of the ones from the environment
      # The policy is then limted to outputting a gradient and cannot change the action from
      # One end of the action space to the other immediately
      # Last actions are added as observations to the observation input
      action_gradients = False,

      # The maximum fraction of the action space to traverse in one step
      action_gradient_stepsize = 1e-4,

      # Starting action in absolute terms
      # Only regarded when using action gradients, otherwise
      # use random_exploration_mu
      starting_action = [0, 30],

      # Feed in this number of past observations as additional data to the net
      # This will increase the observation size by the given factor
      # None if you want to disable it
      feed_past = 0,

      # In prioritized experience replay the probability of a sample to be replayed is proportional to its TD error
      # This allows critical samples to be replayed more often and thus the critic updating faster
      prioritized_experience_replay = False,

      # How strongly to apply prioritized experience replay
      # 1 means sampling happens only according to the priority,
      # 0 means uniform sampling
      prioritized_experience_replay_alpha = 0.5,

      # How strongly to use importance sampling to account for
      # overestimation bias
      # 1 means full compensation, 0 means no compensation
      prioritized_experience_replay_beta = 0.5,

      # Whether to clip action gradients at the maximum level returned from the environment
      clip_action_gradients = True,

      # Whether to pretrain the policy
      pretrain_policy = False,

      # How many steps to pretrain the policy after random exploration to predict the actions of the
      # expert controller
      pretrain_policy_steps = 100,

      # Batch size to use in pretraining
      pretrain_policy_batch_size = 32,

      # The learning rate to use on pretraining
      pretrain_policy_lr = 1e-4,

    )