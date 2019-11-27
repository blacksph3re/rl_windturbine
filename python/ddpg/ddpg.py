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
import tensorflow as tf

from .models import Critic, Actor
from .utils import *


class DDPG:
    
    def __init__(self, hparams):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
            self.action_grad_denormalizer = (self.real_act_high - self.real_act_low) * self.hparams.action_gradient_stepsize

            assert(np.all(self.real_last_action <= self.real_act_high))
            assert(np.all(self.real_last_action >= self.real_act_low))
            assert(not np.any(self.action_grad_denormalizer == 0))
        
        # initialize actor and critic networks
        self.critic = Critic(self.obs_dim, self.act_dim, hparams.critic_sizes[0], hparams.critic_sizes[1], hparams.critic_sizes[2], hparams.init_weight_limit, hparams.critic_simple).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.act_dim, hparams.critic_sizes[0], hparams.critic_sizes[1], hparams.critic_sizes[2], hparams.init_weight_limit, hparams.critic_simple).to(self.device)
        
        self.actor = Actor(self.obs_dim, self.act_dim, hparams.actor_sizes[0], hparams.actor_sizes[1], hparams.init_weight_limit, hparams.actor_simple).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.act_dim, hparams.actor_sizes[0], hparams.actor_sizes[1], hparams.init_weight_limit, hparams.actor_simple).to(self.device)

        # Copy target parameters
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        # optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=hparams.critic_lr)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=hparams.actor_lr)

        # If in twin critic mode, add another critic
        if(self.hparams.twin_critics):
            self.critic2 = Critic(self.obs_dim, self.act_dim, hparams.critic_sizes[0], hparams.critic_sizes[1], hparams.critic_sizes[2], hparams.init_weight_limit, hparams.critic_simple).to(self.device)
            self.critic2_target = Critic(self.obs_dim, self.act_dim, hparams.critic_sizes[0], hparams.critic_sizes[1], hparams.critic_sizes[2], hparams.init_weight_limit, hparams.critic_simple).to(self.device)
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(param.data)
            self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=hparams.critic_lr)

        if(self.hparams.critic_loss == 'huber'):
            self.critic_loss_function = F.smooth_l1_loss
        else:
            self.critic_loss_function = F.mse_loss

        # Buffor for experience replay
        self.replay_buffer = BasicBuffer(hparams.buffer_maxlen)

        # Noises
        # Random exploration
        if(self.hparams.random_exploration_type == 'correlated'):
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
        m = (self.act_low - self.act_high) / (-1 - 1)
        c = self.act_low - m * (-1)
        assert(not np.any(m == 0))
        self.action_normalizer_params = (m, c)
        self.action_normalizer_params_gpu = (torch.FloatTensor(m).to(self.device),
                                             torch.FloatTensor(c).to(self.device))

        # We just set the rest to identity projection
        self.state_normalizer_params = (np.ones(self.obs_dim),
                                        np.zeros(self.obs_dim))
        self.state_normalizer_params_gpu = (torch.FloatTensor(np.ones(self.obs_dim)).to(self.device), 
                                            torch.FloatTensor(np.zeros(self.obs_dim)).to(self.device))

        self.reward_normalizer_params = (np.ones(1),
                                         np.zeros(1))
        self.reward_normalizer_params_gpu = (torch.FloatTensor(np.ones(1)).to(self.device), 
                                             torch.FloatTensor(np.zeros(1)).to(self.device))

        self.logger = QBladeLogger(self.hparams.logdir, self.hparams.log_steps)
        for key, value in self.hparams.values().items():
            self.logger.writer.add_text(key, str(value), 0)

        self.time = 0
        self.last_state = np.zeros(self.obs_dim)
        self.last_action = np.zeros(self.act_dim)
        self.epoch_reward = 0
        self.killcount = 0
        self.epoch_killcount = 0

    def normalize_action(self, action, gpu=False):
        m, c = self.action_normalizer_params_gpu if gpu else self.action_normalizer_params
        return (action - c) / m

    def denormalize_action(self, action, gpu=False):
        m, c = self.action_normalizer_params_gpu if gpu else self.action_normalizer_params
        return action * m + c

    def normalize_state(self, state, gpu=False):
        m, c = self.state_normalizer_params_gpu if gpu else self.state_normalizer_params
        return (state - c) / m

    def denormalize_state(self, state, gpu=False):
        m, c = self.state_normalizer_params_gpu if gpu else self.state_normalizer_params
        return state * m + c

    def normalize_reward(self, reward, gpu=False):
        m, c = self.reward_normalizer_params_gpu if gpu else self.reward_normalizer_params
        return (reward - c) / m

    def denormalize_reward(self, reward, gpu=False):
        m, c = self.reward_normalizer_params_gpu if gpu else self.reward_normalizer_params
        return reward * m + c

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
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        if (self.hparams.normalize_observations):
            state = self.normalize_state(state, True)
        
        # Run the network
        action = self.actor.forward(state)

        # Add noise
        if(add_noise):
            noise = torch.FloatTensor(self.action_noise.get_noise(self.time)).to(self.device)
            action = action + noise
        
        if(self.hparams.normalize_actions):
            action = self.denormalize_action(action, True)
        
        # Get it to the cpu
        action = action.squeeze(0).cpu().detach().numpy()

        action = self.clip_action(action)

        assert(not np.any(np.isnan(action)))

        return action


    def prepare(self, state):
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

    def calc_normalizations(self):
        # Load additional data if wanted
        if(self.hparams.normalization_extradata):
            tmpbuffer = BasicBuffer(100000)
            tmpbuffer.load(self.hparams.normalization_extradata)
            extrastates = [state for state, _, _, _, _ in tmpbuffer.get_buffer()]
            extrarewards = [reward for _, _, reward, _, _ in tmpbuffer.get_buffer()]


        # Calculate state max and min
        states = [state for state, _, _, _, _ in self.replay_buffer.get_buffer()]

        if(self.hparams.normalization_extradata):
            states = states + extrastates
        
        state_max = np.amax(states, 0)
        state_min = np.amin(states, 0)

        # If we do action gradients, we concatenate last actions to observations
        # Thus we already know min and max for these
        if(self.hparams.action_gradients):
            for i in range(0, self.act_dim):
                state_max[self.real_obs_dim + i] = self.real_act_high[i]
                state_min[self.real_obs_dim + i] = self.real_act_low[i]


        state_max[(state_min - state_max) == 0] += 1e-6


        # Define a linear projection
        m = (state_min - state_max) / (-1 - 1)
        c = state_min - m * (-1)

        assert(not np.any(m == 0))

        # Store once as cpu and once as gpu variant
        self.state_normalizer_params = (m, c)
        self.state_normalizer_params_gpu = (torch.FloatTensor(m).to(self.device),
                                            torch.FloatTensor(c).to(self.device))

        # Same for rewards
        rewards = [reward for _, _, reward, _, _ in self.replay_buffer.get_buffer()]

        if(self.hparams.normalization_extradata):
            rewards = rewards + extrarewards

        reward_max = np.amax(rewards, 0)
        reward_min = np.amin(rewards, 0)

        reward_max[(reward_min - reward_max) == 0] += 1e-6

        m = (reward_min - reward_max) / (-1 - 1)
        c = reward_min - m * (-1)

        assert(not np.any(m == 0))

        self.reward_normalizer_params = (m, c)
        self.reward_normalizer_params_gpu = (torch.FloatTensor(m).to(self.device),
                                             torch.FloatTensor(c).to(self.device))
    
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
        optimizer = optim.Adam(self.actor.parameters(), lr=self.hparams.pretrain_policy_lr)
        validation_size = int(len(self.replay_buffer) / 0.03)
        validation_indices = np.random.choice(len(self.replay_buffer), validation_size)
        self.replay_buffer.update_priorities(validation_indices, np.ones(validation_size)*1e-10)

        validation_states, validation_actions, _, _, _ = zip(*self.replay_buffer.get_buffer()[validation_indices])

        validation_states = torch.FloatTensor(validation_states).to(self.device)
        validation_actions = torch.FloatTensor(validation_actions).to(self.device)

        if(self.hparams.normalize_observations):
            validation_states = self.normalize_state(validation_states, True)

        if(self.hparams.normalize_actions):
            validation_actions = self.normalize_action(validation_actions, True)

        for i in range(0, self.hparams.pretrain_policy_steps):
            state_batch, action_batch, _, _, _, _ = self.replay_buffer.sample(self.hparams.pretrain_policy_batch_size)
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            
            if(self.hparams.normalize_observations):
                state_batch = self.normalize_state(state_batch, True)

            if(self.hparams.normalize_actions):
                action_batch = self.normalize_action(action_batch, True)

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

    def update(self, batch_size, time = None):
        time = time or self.time

        state_batch, action_batch, reward_batch, next_state_batch, masks, indices = self.replay_buffer.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)

        # Add replay noise if desired
        if(self.hparams.replay_noise):
            state_noise = [np.random.uniform(-np.ones(self.obs_dim), np.ones(self.obs_dim), self.obs_dim) for i in range(0, batch_size)]
            state_noise = torch.FloatTensor(state_noise).to(self.device)
            state_noise = self.denormalize_state(state_noise, True)
            action_noise = [np.random.uniform(-np.ones(self.act_dim), np.ones(self.act_dim), self.act_dim) for i in range(0, batch_size)]
            action_noise = torch.FloatTensor(action_noise).to(self.device)
            action_noise = self.denormalize_action(action_noise, True)
            next_state_noise = [np.random.uniform(-np.ones(self.obs_dim), np.ones(self.obs_dim), self.obs_dim) for i in range(0, batch_size)]
            next_state_noise = torch.FloatTensor(next_state_noise).to(self.device)
            next_state_noise = self.denormalize_state(next_state_noise, True)

            state_batch = state_batch * (1 - self.hparams.replay_noise) + state_noise * self.hparams.replay_noise
            action_batch = action_batch * (1 - self.hparams.replay_noise) + action_noise * self.hparams.replay_noise
            next_state_batch = next_state_batch * (1 - self.hparams.replay_noise) + next_state_noise * self.hparams.replay_noise

        # Normalize all states and actions if desired
        if(self.hparams.normalize_observations):
            state_batch = self.normalize_state(state_batch, True)
            next_state_batch = self.normalize_state(next_state_batch, True)

        if(self.hparams.normalize_rewards):
            reward_batch = self.normalize_reward(reward_batch, True)

        if(self.hparams.normalize_actions):
            action_batch = self.normalize_action(action_batch, True)

        self.critic_optimizer.zero_grad()
        curr_Q = self.critic.forward(state_batch, action_batch)
        next_actions = self.actor_target.forward(next_state_batch)
        next_Q = self.critic_target.forward(next_state_batch, next_actions.detach())
        expected_Q = reward_batch + self.gamma * next_Q
        
        # If using twin q, take minimum of the two next_Q estimations and replace that with the current one
        if(self.hparams.twin_critics):
            self.critic2_optimizer.zero_grad()
            curr_Q2 = self.critic2.forward(state_batch, action_batch)
            next_Q2 = self.critic2_target.forward(next_state_batch, next_actions.detach())
            expected_Q = reward_batch + self.gamma * torch.min(next_Q, next_Q2)

            q2_loss = self.critic_loss_function(curr_Q2, expected_Q.detach())
            q2_loss.backward()
            self.critic2_optimizer.step()
            self.logger.add_scalar('Loss/q2', q2_loss, time)
            self.logger.add_scalar('Loss/next-q2-mean', next_Q2.mean().detach().cpu(), time)


        # update critic
        q_loss = self.critic_loss_function(curr_Q, expected_Q.detach())

        q_loss.backward() 
        self.critic_optimizer.step()
        self.logger.add_scalar('Loss/q', q_loss, time)
        self.logger.add_scalar('Loss/next_Q_mean', next_Q.mean().detach().cpu(), time)



        # Update priorities if using prioritized experience replay
        # TODO Should we correct importance sampling now?
        if(self.hparams.prioritized_experience_replay):
            # In twin critic mode, take the max value of the two, 
            # Thus, if any of the two nets is performing badly on thet sample,
            # we'll sample it more often.
            if(self.hparams.twin_critics):
                td_error = (expected_Q.detach() - torch.min(curr_Q.detach(), curr_Q2.detach())).abs().cpu().data.numpy()
            else:
                td_error = (expected_Q.detach() - curr_Q.detach()).abs().cpu().data.numpy()
            self.replay_buffer.update_priorities(indices, td_error)

        if(self.hparams.log_net_insights and time % self.hparams.log_net_insights == 0):
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

            if('linear3.weight' in critic_state and 'linear4.weight' in critic_state):
                self.logger.add_histogram_nofilter('critic-weights/linear3', critic_state['linear3.weight'], time)
                self.logger.add_scalar_nofilter('critic/l3-dead', np.sum(critic_state['linear3.weight'].data.numpy() == 0), time)
                self.logger.add_histogram_nofilter('critic-grads/linear3', np.log10(np.abs(critic_state['linear3.weight'].grad)+1e-20), time)
                self.logger.add_scalar_nofilter('critic/l3-grad-exp', np.sum(critic_state['linear3.weight'].grad.numpy() > 1e10), time)
                self.logger.add_scalar_nofilter('critic/l3-vanishing-grad', np.sum(critic_state['linear3.weight'].grad.numpy() == 0), time)

                self.logger.add_histogram_nofilter('critic-weights/linear4', critic_state['linear4.weight'], time)
                self.logger.add_scalar_nofilter('critic/l4-dead', np.sum(critic_state['linear4.weight'].data.numpy() == 0), time)
                self.logger.add_histogram_nofilter('critic-grads/linear4', np.log10(np.abs(critic_state['linear4.weight'].grad)+1e-20), time)
                self.logger.add_scalar_nofilter('critic/l4-grad-exp', np.sum(critic_state['linear4.weight'].grad.numpy() > 1e10), time)
                self.logger.add_scalar_nofilter('critic/l4-vanishing-grad', np.sum(critic_state['linear4.weight'].grad.numpy() == 0), time)


            self.critic.to(self.device)

            self.logger.add_histogram_nofilter('priorities', np.log10(self.replay_buffer.get_priorities()+1e-20), time)


        # update actor
        if(time%self.hparams.actor_delay == 0):
            self.actor_optimizer.zero_grad()
            policy_loss = -torch.mean(self.critic.forward(state_batch, self.actor.forward(state_batch)))

            policy_loss.backward()

            self.actor_optimizer.step()
            self.logger.add_scalar('Loss/policy', policy_loss, time)


        if(self.hparams.log_net_insights and self.time % self.hparams.log_net_insights == 0):
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


        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        # Add actor parameter noise
        if(self.hparams.parameter_noise != 0):
            for param in self.actor.parameters():
                param.data.copy_(param.data + self.hparams.parameter_noise * np.random.uniform(-1, 1))

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
        if(done):
            self.killcount += 1
            self.epoch_killcount += 1
            if(self.hparams.feed_past):
                self.past_obs.clear()

        [norm_reward] = self.normalize_reward([reward], False)

        if (self.time == self.hparams.random_exploration_steps):
            print("Random exploration finished, preparing normal run")
            self.calc_normalizations()

            # If wanted, pretrain the policy to actions from random exploration
            if(self.hparams.pretrain_policy_steps):
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

        if (self.time > 0 and self.time % self.hparams.steps_per_epoch == 0):
            epoch = (self.time // self.hparams.steps_per_epoch)
            print('Epoch %d' % epoch)
            print('Epoch reward %d' % self.epoch_reward)
            print('Step %d' % self.time)
            self.logger.add_scalar_nofilter('Epoch/reward', self.epoch_reward, epoch)
            self.logger.add_scalar_nofilter('Epoch/killcount', self.epoch_killcount, epoch)

            self.epoch_reward = 0
            self.epoch_killcount = 0

        if (self.time > self.hparams.random_exploration_steps and len(self.replay_buffer) > self.batch_size):
            self.update(self.batch_size)

        if(self.time < self.hparams.random_exploration_steps):
            action = [2e7*state[0]-12e6,0]
            action += self.denormalize_action(self.random_exploration_noise.get_noise(self.time), False)
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
            #self.logger.logAction(self.time, self.normalize_action(action), 'act_norm')
            #self.logger.logObservation(self.time, self.normalize_state(state), 'obs_norm')
            self.logger.add_scalar('Reward/norm', norm_reward, self.time)


        self.time = self.time + 1

        return action, done

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
            "hparams": self.hparams.values(),
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

        # Load model state
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, prefix)))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, prefix)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, prefix)))
        self.critic_target.load_state_dict(torch.load('%s/%s_critic_target.pth' % (directory, prefix)))
        self.replay_buffer.load("%s/%s_memory.dat" % (directory, prefix))

        # The main loop needs to know which timestep to start with
        return self.time


    def get_default_hparams():
        return tf.contrib.training.HParams(
            # Number of steps of interaction (state-action pairs) 
            # for the agent and the environment in each epoch.
            steps_per_epoch = 500,

            # Number of steps to sample random actions
            # before starting to utilize the policy
            random_exploration_steps = 2000,

            # How many steps to train the agent after random exploration
            random_exploration_training_steps = 2000,

            # Type of random exploration noise
            # 'correlated' (OU Noise), 'uncorrelated' (gaussian) or 'none'
            random_exploration_type = "correlated",

            # How strongly it is drawn towards mu
            random_exploration_theta = 0.05,

            # How strongly it wanders around
            random_exploration_sigma = 0.02,

            # The default action to start with in random exploration
            # Also where most of the exploration will happen around
            # -1 being minimum action and 1 maximum
            # None means np.zeros(act_dim) as mu
            # If gradient actions are active, this is in gradient action space
            random_exploration_mu = [-1,-1],

            # Number of steps after which to write out a checkpoint
            checkpoint_steps = 10000,

            # Where to store the checkpoints
            checkpoint_dir = "checkpoints",

            # Number of total epochs to run the training
            epochs = 2000,

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
            tau = 1e-2,

            # The maximum size of the replay buffer, i.e. how many steps to store as
            # experience
            buffer_maxlen = 100000,

            # Learning rate of the Q approximator
            critic_lr = 1e-4,

            # Neural network sizes of the critic
            critic_sizes = [64, 32, 16],

            # Whether to use a 4-layer or a 2-layer structure
            # for the critic
            critic_simple = False,

            # Loss function for the critic
            # default is mean square error
            # Other values: "mse", "huber"
            critic_loss = 'huber',

            # Whether to use duelling critics
            # In this variant, two critics try to estimate q
            # and only the lower estimation will be chosen
            # Helps prevent loss explosions and overestimation
            twin_critics = False,

            # Learning rate of the policy
            actor_lr = 1e-5,

            # Network sizes of the policy
            actor_sizes = [32, 8],

            # Whether to use a 2-layer or a 3-layer structure for the actor
            actor_simple = False,

            # Only update the actor every n critic updates
            # 1 for updating every critic update
            actor_delay = 2,

            # Whether to use batch normalization in both the actor and critic
            batch_normalization = True,

            # Network parameters of both the actor and critic will be initialized to
            # a uniform random value between [-init_weight_limit, init_weight_limit]
            init_weight_limit = 1.,

            # Observation and action dimension, will be overwritten by environment obs dim
            obs_dim = 1,
            act_dim = 1,

            # Upper and lower limits for actions, will be overwritten by environment act limits
            act_high = [0.],
            act_low = [0.],
            act_max_grad = [0.],

            # Where to store tensorboard logs
            logdir = "logs",

            # Log data every n steps
            log_steps = 4,

            # Log net insight histograms every n steps
            # -1 for disabling completely
            log_net_insights = 128,

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
            action_noise_sigma = 0.08,

            # For decreasing noises this is the minimum sigma level
            action_noise_sigma_decayed = 1e-8,

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
            normalize_observations = True,

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
            normalization_extradata = 'normalization_data/no_past_feeding.dat',
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

            # Whether to clip action gradients at the maximum level returned from the environment
            clip_action_gradients = True,

            # How many steps to pretrain the policy after random exploration to predict the actions of the
            # expert controller
            pretrain_policy_steps = 5000,

            # Batch size to use in pretraining
            pretrain_policy_batch_size = 32,

            # The learning rate to use on pretraining
            pretrain_policy_lr = 5e-5,

        )