from .env.qbladeadapter import QBladeAdapter
from .ddpg.ddpg import DDPG

import torch

def main():

  environment = 'qblade'
  policy = 'actor'

  hparams = DDPG.get_default_hparams()

  if(environment == 'qblade'):
    env = QBladeAdapter()
    # Set environment-specific hparams
    hparams.obs_dim = env.get_obs_dim()
    hparams.act_dim = env.get_act_dim()
    hparams.act_high = env.get_act_high()
    hparams.act_low = env.get_act_low()
    hparams.act_max_grad = env.get_act_max_grad()

  if(policy == 'actor'):
    pol = Actor(hparams.obs_dim, hparams.act_dim, hparams.actor_sizes[0], hparams.actor_sizes[1], hparams.init_weight_limit, hparams.actor_simple)

  for t in range(0, hparams.steps_per_epoch * hparams.epochs):


main()