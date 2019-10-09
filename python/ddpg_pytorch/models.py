import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_size_1 = 1024, hidden_size_2 = 512, hidden_size_3 = 300, init_w = 1e-3):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.linear1 = nn.Linear(self.obs_dim + self.action_dim, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, 1)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, a):
        xa_cat = torch.cat([x,a], 1)
        xa = F.relu(self.linear1(xa_cat))
        qval = self.linear2(xa)

        return qval

class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_size_1 = 512, hidden_size_2 = 128, init_w = 1e-3):
        super(Actor, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.linear1 = nn.Linear(self.obs_dim, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, self.action_dim)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)


    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = torch.tanh(self.linear2(x))

        return x