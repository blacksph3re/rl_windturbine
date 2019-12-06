from .ddpg.models import Actor

class ActorPolicy:
  def __init__(self, hparams):
    self.actor = Actor(hparams.obs_dim, hparams.act_dim, hparams.actor_sizes[0], hparams.actor_sizes[1], hparams.init_weight_limit, hparams.actor_simple)

  def prepare(self, obs):