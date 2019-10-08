import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime

# For hparams
import tensorflow as tf

from .models import Critic, Actor
from .utils import BasicBuffer
from .utils import OUNoise


class DDPG:
    
    def __init__(self, hparams):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.hparams = hparams
        self.obs_dim = hparams.obs_dim
        self.action_dim = hparams.act_dim
        self.batch_size = hparams.batch_size
        
        # hyperparameters
        self.gamma = hparams.gamma
        self.tau = hparams.tau
        
        # initialize actor and critic networks
        self.critic = Critic(self.obs_dim, self.action_dim, hparams.critic_sizes[0], hparams.critic_sizes[1], hparams.critic_sizes[2]).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.action_dim, hparams.critic_sizes[0], hparams.critic_sizes[1], hparams.critic_sizes[2]).to(self.device)
        
        self.actor = Actor(self.obs_dim, self.action_dim, hparams.actor_sizes[0], hparams.actor_sizes[1]).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.action_dim, hparams.actor_sizes[0], hparams.actor_sizes[1]).to(self.device)
    
        # Copy target parameters
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        # optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=hparams.critic_lr)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=hparams.actor_lr)
    
        self.replay_buffer = BasicBuffer(hparams.buffer_maxlen)        
        self.noise = OUNoise(hparams.act_dim, hparams.act_low, hparams.act_high)
        self.noise_polynom = np.polyfit([hparams.noise_start, hparams.noise_end], [hparams.noise_start_factor, hparams.noise_end_factor], 1)
        self.action_normalizer = torch.FloatTensor(np.ones(hparams.act_dim)).to(self.device)
        self.state_normalizer = torch.FloatTensor(np.ones(hparams.obs_dim)).to(self.device)

        self.writer = SummaryWriter(hparams.logdir + '/' + str(datetime.now()))

        self.time = 0
        self.last_state = None
        self.last_action = None
        self.epoch_reward = 0
        
    def get_action(self, obs, add_noise=False):
        # Send the observation to the device, normalize it
        # Then calculate the action and denormalize it
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        if (self.hparams.normalize_observations):
            state = state * self.state_normalizer
        
        action = self.actor.forward(state)
        
        if (self.hparams.normalize_actions):
            action = action / self.action_normalizer
        
        action = action.squeeze(0).cpu().detach().numpy()

        if(add_noise):
            if(self.time < self.hparams.random_exploration_steps):
                action = self.random_action()
            elif(self.hparams.noise_type == 'correlated'):
                action = self.noise.get_action(action, self.time)
            else:
                m, c = self.noise_polynom
                x = np.clip(self.time, self.hparams.noise_start, self.hparams.noise_end)
                noise_level = np.clip((m * x + c), 0, 1)
                action = self.random_action() * noise_level + action * (1 - noise_level)

        self.last_action = action
        self.last_state = obs

        return action

    def random_action(self):
        return np.random.uniform(self.hparams.act_low, self.hparams.act_high, self.hparams.act_dim)

    def prepare(self, obs):
        return self.get_action(obs, True)
    def reset_finalize(self, obs):
        return self.get_action(obs, True)

    def calc_normalizations(self):
        # Get the max and minimum from the replay buffer
        state_max, action_max, reward_max, next_state_max, masks_max = np.amax(self.replay_buffer.buffer, 0)
        state_min, action_min, reward_min, next_state_min, masks_min = np.amin(self.replay_buffer.buffer, 0)

        # Take the absolute maximum of min and max
        state_mul = np.maximum(np.abs(state_max), np.abs(state_min))
        action_mul = np.maximum(np.abs(action_max), np.abs(action_min))

        # Make sure no entry is zero
        state_mul[state_mul == 0] = 1e-6
        action_mul[action_mul == 0] = 1e-6

        # Store as tensors
        self.state_normalizer = torch.FloatTensor(state_mul).to(self.device)
        self.action_normalizer = torch.FloatTensor(action_mul).to(self.device)
    
    def update(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)

        # Normalize all states and actions if desired
        if(self.hparams.normalize_observations):
            state_batch = state_batch * self.state_normalizer
            next_state_batch = next_state_batch * self.state_normalizer

        if(self.hparams.normalize_actions):
            action_batch = action_batch * self.action_normalizer

        # Add replay noise if desired
        if(self.hparams.replay_noise):
            state_noise = [np.random.uniform(-np.ones(self.hparams.obs_dim), np.ones(self.hparams.obs_dim), self.hparams.obs_dim) for i in range(0, batch_size)]
            state_noise = torch.FloatTensor(state_noise).to(self.device)
            action_noise = [np.random.uniform(-np.ones(self.hparams.act_dim), np.ones(self.hparams.act_dim), self.hparams.act_dim) for i in range(0, batch_size)]
            action_noise = torch.FloatTensor(action_noise).to(self.device)
            next_state_noise = [np.random.uniform(-np.ones(self.hparams.obs_dim), np.ones(self.hparams.obs_dim), self.hparams.obs_dim) for i in range(0, batch_size)]
            next_state_noise = torch.FloatTensor(next_state_noise).to(self.device)

            state_batch = state_batch * (1 - self.hparams.replay_noise) + state_noise * self.hparams.replay_noise
            action_batch = action_batch * (1 - self.hparams.replay_noise) + action_noise * self.hparams.replay_noise
            next_state_batch = next_state_batch * (1 - self.hparams.replay_noise) + next_state_noise * self.hparams.replay_noise



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
        print([p.grad for p in self.actor.parameters()])
        print(policy_loss)

        print([p.grad for p in self.critic.parameters()])

        self.actor_optimizer.step()

        self.writer.add_scalar('Loss/q', q_loss, self.time)
        self.writer.add_scalar('Loss/policy', policy_loss, self.time)
 

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
        self.replay_buffer.push(self.last_state, self.last_action, reward, state, done)

        self.writer.add_scalar('Reward', reward, self.time)

        if (self.time == self.hparams.random_exploration_steps):
            self.calc_normalizations()

        if (len(self.replay_buffer) > self.batch_size and self.time > self.hparams.random_exploration_steps):
            self.update(self.batch_size)

        if (self.time > 0 and self.time % self.hparams.steps_per_epoch == 0):
            epoch = (self.time // self.hparams.steps_per_epoch)
            print('Epoch %d' % epoch)
            print('Epoch reward %d' % self.epoch_reward)
            print('Step %d' % self.time)
            self.writer.add_scalar('Epoch reward', self.epoch_reward, epoch)

            # Printing weights
            #print(dict(self.actor.state_dict())['linear1.weight'])
            self.writer.add_histogram('actor/linear1', dict(self.actor.state_dict())['linear1.weight'], self.time)
            self.writer.add_histogram('actor/linear2', dict(self.actor.state_dict())['linear2.weight'], self.time)
            self.writer.add_histogram('actor/linear3', dict(self.actor.state_dict())['linear3.weight'], self.time)        

            self.writer.add_histogram('critic/linear1', dict(self.critic.state_dict())['linear1.weight'], self.time)
            self.writer.add_histogram('critic/linear2', dict(self.critic.state_dict())['linear2.weight'], self.time)
            self.writer.add_histogram('critic/linear3', dict(self.critic.state_dict())['linear3.weight'], self.time)
            self.writer.add_histogram('critic/linear4', dict(self.critic.state_dict())['linear4.weight'], self.time)


            self.epoch_reward = 0

        action = self.get_action(state, True)

        return action, False

    def close(self):
        self.writer.close()

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
            steps_per_epoch = 500,
            random_exploration_steps = 500,
            checkpoint_steps = 10000,
            checkpoint_dir = "checkpoints",
            epochs = 500,
            test_steps = 1000,
            batch_size = 16,
            gamma = 0.99,
            tau = 1e-2,
            buffer_maxlen = 100000,
            critic_lr = 1e-3,
            critic_sizes = [128, 64, 32],
            actor_lr = 1e-3,
            actor_sizes = [32, 8],
            obs_dim = 20,
            act_dim = 1,
            act_high = 1,
            act_low = -1,
            logdir = "logs",
            noise_start = 5,            # When to start applying noise
            noise_start_factor = 0.5,     # The factor of noise at the beginning (1 max, 0 min)
            noise_end = 30000,          # When to stop applying noise
            noise_end_factor = 0,       # The factor of noise at the end (1 max, 0 min)
            noise_type = 'uncorrelated',
            parameter_noise = 0,
            replay_noise = 1e-2,
            normalize_actions = True,
            normalize_observations = True,
        )