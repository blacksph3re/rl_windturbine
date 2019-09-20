import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

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
        self.critic = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.action_dim).to(self.device)
        
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.action_dim).to(self.device)
    
        # Copy critic target parameters
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=hparams.critic_lr)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=hparams.actor_lr)
    
        self.replay_buffer = BasicBuffer(hparams.buffer_maxlen)        
        self.noise = OUNoise(hparams.act_dim, hparams.act_low, hparams.act_high)

        self.writer = SummaryWriter(hparams.logdir)

        self.time = 0
        self.last_state = None
        self.last_action = None
        self.epoch_reward = 0
        
    def get_action(self, obs, add_noise=False):
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor.forward(state)
        action = action.squeeze(0).cpu().detach().numpy()
        if(add_noise):
            if(self.hparams.noise_type == 'correlated'):
                action = self.noise.get_action(action, self.time)
            else:
                action += self.random_action() * self.hparams.noise_factor

        self.last_action = action
        self.last_state = obs
        return action

    def random_action(self):
        return np.random.uniform(self.hparams.act_low, self.hparams.act_high, self.hparams.act_dim)

    def prepare(self, obs):
        return self.get_action(obs, True)
    def reset_finalize(self, obs):
        return self.get_action(obs, True)
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.replay_buffer.sample(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)
   
        curr_Q = self.critic.forward(state_batch, action_batch)
        next_actions = self.actor_target.forward(next_state_batch)
        next_Q = self.critic_target.forward(next_state_batch, next_actions.detach())
        expected_Q = reward_batch + self.gamma * next_Q
        
        # update critic
        q_loss = F.mse_loss(curr_Q, expected_Q.detach())
        self.writer.add_scalar('Loss/q', q_loss, self.time)

        self.critic_optimizer.zero_grad()
        q_loss.backward() 
        self.critic_optimizer.step()

        # update actor
        policy_loss = -self.critic.forward(state_batch, self.actor.forward(state_batch)).mean()
        self.writer.add_scalar('Loss/policy', policy_loss, self.time)
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

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

        if (len(self.replay_buffer) > self.batch_size and self.time > self.hparams.random_exploration_steps):
            self.update(self.batch_size)

        if (self.time > 0 and self.time % self.hparams.steps_per_epoch == 0):
            epoch = (self.time // self.hparams.steps_per_epoch)
            print('Epoch %d' % epoch)
            print('Epoch reward %d' % self.epoch_reward)
            print('Step %d' % self.time)
            self.writer.add_scalar('Epoch reward', self.epoch_reward, epoch)

            self.epoch_reward = 0

        action = self.get_action(state, True)
        if (self.time < self.hparams.random_exploration_steps):
            action = self.random_action()
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
            checkpoint_steps = 500,
            epochs = 100,
            test_steps = 1000,
            batch_size = 64,
            gamma = 0.99,
            tau = 1e-2,
            buffer_maxlen = 100000,
            critic_lr = 1e-4,
            actor_lr = 1e-4,
            obs_dim = 20,
            act_dim = 1,
            act_high = 1,
            act_low = -1,
            logdir = "logs",
            noise_factor = 0.2,
            noise_type = 'uncorrelated',
            parameter_noise = 0.05,
        )