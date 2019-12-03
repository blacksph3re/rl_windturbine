import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size_1 = 1024, hidden_size_2 = 512, init_w = 0.1, simple = False, batch_normalization = False):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.simple = simple

        # If batch normalization is enabled, add batch-norm layers
        # Otherwise just put the identity function
        if(batch_normalization):
            self.bn1 = nn.BatchNorm1d(hidden_size_1)
            self.bn2 = nn.BatchNorm1d(hidden_size_2)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x
            self.bn3 = lambda x: x

        if(self.simple):
            self.linear1 = nn.Linear(self.obs_dim + self.action_dim, hidden_size_1)
            self.linear2 = nn.Linear(hidden_size_1, 1)
            print('Initialized Critic with %d, %d, %d' % (self.obs_dim + self.action_dim, hidden_size_1, 1))
        else:
            self.linear1 = nn.Linear(self.obs_dim, hidden_size_1)
            self.linear2 = nn.Linear(hidden_size_1 + self.action_dim, hidden_size_2)
            self.linear3 = nn.Linear(hidden_size_2, 1)
            print('Initialized Critic with %d, %d, %d, %d' % (self.obs_dim,
                                                              hidden_size_1 + self.action_dim,
                                                              hidden_size_2,
                                                              1))

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)

        if(not self.simple):
            self.linear3.weight.data.uniform_(-init_w, init_w)

    def first_layer(self, x, a):
        if(self.simple):
            return F.relu(self.bn1(self.linear1(torch.cat([x, a], 1))))
        else:
            return F.relu(self.bn1(self.linear1(x)))

    def forward_complex(self, x, a):
        xa = F.relu(self.bn1(self.linear1(x)))
        xa = torch.cat([xa,a], 1)
        xa = F.relu(self.bn2(self.linear2(xa)))
        qval = self.linear3(xa)

        return qval

    def forward_simple(self, x, a):
        xa_cat = torch.cat([x,a], 1)
        xa = F.relu(self.bn1(self.linear1(xa_cat)))
        qval = self.linear2(xa)

        return qval

    def forward(self, x, a):
        if(self.simple):
            return self.forward_simple(x, a)
        return self.forward_complex(x, a)

class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_size_1 = 512, hidden_size_2 = 128, init_w = 0.1, simple = False, batch_normalization = False):
        super(Actor, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.simple = simple

        if(batch_normalization):
            self.bn1 = nn.BatchNorm1d(hidden_size_1)
            self.bn2 = nn.BatchNorm1d(hidden_size_2)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x

        if(self.simple):
            self.linear1 = nn.Linear(self.obs_dim, hidden_size_1)
            self.linear2 = nn.Linear(hidden_size_1, self.action_dim)
            print("Initialized actor with %d, %d, %d" % (self.obs_dim, hidden_size_1, self.action_dim))
        else:
            self.linear1 = nn.Linear(self.obs_dim, hidden_size_1)
            self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
            self.linear3 = nn.Linear(hidden_size_2, self.action_dim)
            print("Initialized actor with %d, %d, %d, %d" % (self.obs_dim, hidden_size_1, hidden_size_2, self.action_dim))

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        if(not self.simple):
            self.linear3.weight.data.uniform_(-init_w, init_w)

    def first_layer(self, obs):
        return F.relu(self.bn1(self.linear1(obs)))

    def forward_complex(self, obs):
        x = F.relu(self.bn1(self.linear1(obs)))
        x = F.relu(self.bn2(self.linear2(x)))
        action = torch.tanh(self.linear3(x))

        return action

    def forward_simple(self, obs):
        x = F.relu(self.bn1(self.linear1(obs)))
        x = torch.tanh(self.linear2(x))

        return x

    def forward(self, obs):
        if(self.simple):
            return self.forward_simple(obs)
        return self.forward_complex(obs)