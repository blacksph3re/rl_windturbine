import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np

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

        # In case of action gradients, reduce action high and low and extend the observation space
        if(self.hparams.action_gradients):
            self.real_obs_dim = self.obs_dim
            self.obs_dim += self.act_dim
            self.real_act_high = self.act_high
            self.real_act_low = self.act_low
            self.act_high = np.ones(self.act_dim)
            self.act_low = -np.ones(self.act_dim)
            self.real_last_action = self.real_act_low.astype(float)
            self.action_grad_denormalizer = (self.real_act_high - self.real_act_low) * self.hparams.action_gradient_stepsize
            assert(not np.any(self.action_grad_denormalizer == 0))
        
        # initialize actor and critic networks
        self.critic = Critic(self.obs_dim, self.act_dim, hparams.critic_sizes[0], hparams.critic_sizes[1], hparams.critic_sizes[2], hparams.init_weight_limit).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.act_dim, hparams.critic_sizes[0], hparams.critic_sizes[1], hparams.critic_sizes[2]).to(self.device)
        
        self.actor = Actor(self.obs_dim, self.act_dim, hparams.actor_sizes[0], hparams.actor_sizes[1], hparams.init_weight_limit).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.act_dim, hparams.actor_sizes[0], hparams.actor_sizes[1]).to(self.device)
    
        # Copy target parameters
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        # optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=hparams.critic_lr)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=hparams.actor_lr)
    
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

        self.logger = QBladeLogger(self.hparams.logdir)

        self.time = 0
        self.last_state = np.zeros(self.obs_dim)
        self.last_action = np.zeros(self.act_dim)
        self.epoch_reward = 0

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


    def get_action(self, obs, add_noise=False):
        action = None
        # Do some random exploration at the beginning
        if(self.time < self.hparams.random_exploration_steps and add_noise):
            action = self.denormalize_action(self.random_exploration_noise.get_noise(self.time), False)
        else:
            # Send the observation to the device, normalize it
            # Then calculate the action and denormalize it
            state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            if (self.hparams.normalize_observations):
                state = self.normalize_state(state, True)
            
            # Run the network
            action = self.actor.forward(state)

            # Add noise
            noise = torch.FloatTensor(self.action_noise.get_noise(self.time)).to(self.device)
            action = action + noise
            
            if(self.hparams.normalize_actions):
                action = self.denormalize_action(action, True)
            
            # Get it to the cpu
            action = action.squeeze(0).cpu().detach().numpy()

        action = np.clip(action, self.act_low, self.act_high)
        self.last_action = action
        self.last_state = obs

        return action


    def prepare(self, obs):
        if(self.hparams.action_gradients):
            return self.real_last_action
        return self.get_action(obs, True)
    def reset_finalize(self, obs):
        return self.prepare(obs)

    def calc_normalizations(self):
        # Calculate state max and min
        states = [state for state, action, reward, next_state, mask in self.replay_buffer.buffer]
        
        state_max = np.amax(states, 0)
        state_min = np.amin(states, 0)

        # If we do action gradients, we concatenate last actions to observations
        # Thus we already know min and max for these
        if(self.hparams.action_gradients):
            for i in range(0, self.act_dim):
                state_max[self.real_obs_dim + i] = self.real_act_high[i]
                state_min[self.real_obs_dim + i] = self.real_act_low[i]


        state_max[(state_min - state_max) == 0] += 1e-3

        # Define a linear projection
        m = (state_min - state_max) / (-1 - 1)
        c = state_min - m * (-1)

        assert(not np.any(m == 0))

        # Store once as cpu and once as gpu variant
        self.state_normalizer_params = (m, c)
        self.state_normalizer_params_gpu = (torch.FloatTensor(m).to(self.device),
                                            torch.FloatTensor(c).to(self.device))

        # Same for rewards
        rewards = [reward for state, action, reward, next_state, mask in self.replay_buffer.buffer]

        reward_max = np.amax(rewards, 0)
        reward_min = np.amin(rewards, 0)

        reward_max[(reward_min - reward_max) == 0] += 1e-3

        m = (reward_min - reward_max) / (-1 - 1)
        c = reward_min - m * (-1)

        assert(not np.any(m == 0))

        self.reward_normalizer_params = (m, c)
        self.reward_normalizer_params_gpu = (torch.FloatTensor(m).to(self.device),
                                             torch.FloatTensor(c).to(self.device))
    
    def update(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)

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
        
        # update critic
        q_loss = F.mse_loss(curr_Q, expected_Q.detach())

        q_loss.backward() 
        self.critic_optimizer.step()
        

        # update actor
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        policy_loss = -torch.mean(self.critic.forward(state_batch, self.actor.forward(state_batch)))

        policy_loss.backward()

        self.actor_optimizer.step()

        self.logger.add_scalar('Loss/q', q_loss, self.time)
        self.logger.add_scalar('Loss/policy', policy_loss, self.time)

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        # Add actor parameter noise
        if(self.hparams.parameter_noise != 0):
            for param in self.actor.parameters():
                param.data.copy_(param.data + self.hparams.parameter_noise * np.random.uniform(-1, 1))

    def step(self, state, reward, done):
        self.time = self.time + 1
        self.epoch_reward += reward

        # Amend last action to observation if using action gradients
        if(self.hparams.action_gradients):
            state = np.concatenate([state, self.real_last_action])

        self.replay_buffer.push(self.last_state, self.last_action, reward, state, done)

        self.logger.add_scalar('Reward', reward, self.time)

        if (self.time == self.hparams.random_exploration_steps):
            self.calc_normalizations()

        if (len(self.replay_buffer) > self.batch_size):
            self.update(self.batch_size)

        if (self.time > 0 and self.time % self.hparams.steps_per_epoch == 0):
            epoch = (self.time // self.hparams.steps_per_epoch)
            print('Epoch %d' % epoch)
            print('Epoch reward %d' % self.epoch_reward)
            print('Step %d' % self.time)
            self.logger.add_scalar('Epoch reward', self.epoch_reward, epoch)

            self.epoch_reward = 0

        action = self.get_action(state, True)

        if(self.hparams.action_gradients):
            self.real_last_action += action * self.action_grad_denormalizer
            self.real_last_action = np.clip(self.real_last_action, self.real_act_low, self.real_act_high)

            self.logger.logGradAction(self.time, action)
            action = self.real_last_action

        self.logger.logAction(self.time, action)
        self.logger.logObservation(self.time, state)

        return action, False

    def close(self):
        self.logger.close()

    def save_checkpoint(self, directory, prefix):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, prefix))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, prefix))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, prefix))
        torch.save(self.critic_target.state_dict(), '%s/%s_critic_target.pth' % (directory, prefix))
        self.replay_buffer.save(directory, prefix)

    def load_checkpoint(self, directory, prefix):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, prefix)))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, prefix)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, prefix)))
        self.critic_target.load_state_dict(torch.load('%s/%s_critic_target.pth' % (directory, prefix)))
        self.replay_buffer.load(directory, prefix)


    def get_default_hparams():
        return tf.contrib.training.HParams(
            # Number of steps of interaction (state-action pairs) 
            # for the agent and the environment in each epoch.
            steps_per_epoch = 500,

            # Number of steps to sample random actions
            # before starting to utilize the policy
            random_exploration_steps = 3000,

            # Type of random exploration noise
            # 'correlated' (OU Noise), 'uncorrelated' (gaussian) or 'none'
            random_exploration_type = "correlated",

            # How strongly it is drawn towards mu
            random_exploration_theta = 0.05,

            # How strongly it wanders around
            random_exploration_sigma = 0.05,

            # The default action to start with in random exploration
            # Also where most of the exploration will happen around
            # -1 being minimum action and 1 maximum
            # None means np.zeros(act_dim) as mu
            random_exploration_mu = [-0.01, -0.01, -0.01, -0.01, -0.01],

            # Number of steps after which to write out a checkpoint
            checkpoint_steps = 10000,

            # Where to store the checkpoints
            checkpoint_dir = "checkpoints",

            # Number of total epochs to run the training
            epochs = 500,

            # Number of steps to run after the training, testing the policy
            test_steps = 1000,

            # Batch size for experience replay
            # The bigger the less it will overfit to specific samples
            batch_size = 16,

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
            critic_lr = 1e-3,

            # Neural network sizes of the critic
            critic_sizes = [32, 16, 16],

            # Learning rate of the policy
            actor_lr = 1e-3,

            # Network sizes of the policy
            actor_sizes = [16, 8],

            # Network parameters of both the actor and critic will be initialized to
            # a uniform random value between [-init_weight_limit, init_weight_limit]
            init_weight_limit = 1,

            # Observation and action dimension, will be overwritten by environment obs dim
            obs_dim = None,
            act_dim = None,

            # Upper and lower limits for actions, will be overwritten by environment act limits
            act_high = None,
            act_low = None,

            # Where to store tensorboard logs
            logdir = "logs",

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
            action_noise_sigma_decayed = 1e-8,

            # For correlated noises, how much the noise stays around the policy action
            action_noise_theta = 0.01,

            # For decreasing noise, after how many iterations the noise will be reduced to
            # action_noise_sigma_decayed
            # It will come closer gradually
            action_noise_decay_steps = 50000,

            # Adding a uniform noise to the policy parameters helped facilitate exploration
            parameter_noise = 0.01,

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
            normalize_rewards = True,

            # Whether to offer gradient action spaces instead of the ones from the environment
            # The policy is then limted to outputting a gradient and cannot change the action from
            # One end of the action space to the other immediately
            # Last actions are added as observations to the observation input
            action_gradients = True,

            # The maximum fraction of the action space to traverse in one step
            action_gradient_stepsize = 0.01,
        )