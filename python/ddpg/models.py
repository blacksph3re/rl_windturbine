import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size_1 = 1024, hidden_size_2 = 512, hidden_size_3 = 300, init_w = 0.1, simple = False):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.simple = simple

        if(self.simple):
            self.linear1 = nn.Linear(self.obs_dim + self.action_dim, hidden_size_1)
            self.linear2 = nn.Linear(hidden_size_1, 1)
        else:
            self.linear1 = nn.Linear(self.obs_dim, hidden_size_1)
            self.linear2 = nn.Linear(hidden_size_1 + self.action_dim, hidden_size_2)
            self.linear3 = nn.Linear(hidden_size_2, hidden_size_3)
            self.linear4 = nn.Linear(hidden_size_3, 1)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)

        if(not self.simple):
            self.linear3.weight.data.uniform_(-init_w, init_w)
            self.linear4.weight.data.uniform_(-init_w, init_w)

    def forward_complex(self, x, a):
        xa = F.relu(self.linear1(x))
        xa = torch.cat([xa,a], 1)
        xa = F.relu(self.linear2(xa))
        xa = F.relu(self.linear3(xa))
        qval = self.linear4(xa)

        return qval

    def forward_simple(self, x, a):
        xa_cat = torch.cat([x,a], 1)
        xa = F.relu(self.linear1(xa_cat))
        qval = self.linear2(xa)

        return qval

    def forward(self, x, a):
        if(self.simple):
            return self.forward_simple(x, a)
        return self.forward_complex(x, a)

class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_size_1 = 512, hidden_size_2 = 128, init_w = 0.1, simple = False):
        super(Actor, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.simple = simple

        if(self.simple):
            self.linear1 = nn.Linear(self.obs_dim, hidden_size_1)
            self.linear2 = nn.Linear(hidden_size_1, self.action_dim)
        else:
            self.linear1 = nn.Linear(self.obs_dim, hidden_size_1)
            self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
            self.linear3 = nn.Linear(hidden_size_2, self.action_dim)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        if(not self.simple):
            self.linear3.weight.data.uniform_(-init_w, init_w)

    def forward_complex(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        action = torch.tanh(self.linear3(x))

        return action

    def forward_simple(self, obs):
        x = F.relu(self.linear1(obs))
        x = torch.tanh(self.linear2(x))

        return x

    def forward(self, obs):
        if(self.simple):
            return self.forward_simple(obs)
        return self.forward_complex(obs)