from multiprocessing import Pool
import subprocess
import json
import os

'''experiments = [
  {"id": 0, "batch_size": 64},
  {"id": 1, "batch_size": 100},
  {"id": 2, "batch_size": 128},
  {"id": 3, "batch_size": 150},
  {"id": 4, "batch_size": 200},
  {"id": 5, "batch_size": 256},
]'''

'''experiments = [
  {"id": 11, "twin_critics": True},
  {"id": 12, "twin_critics": True, "critic_loss": 'mse'},
  {"id": 13, "critic_simple": False},
  {"id": 14, "actor_simple": False},
  {"id": 15, "actor_simple": False, "critic_simple": False}, # Worked kinda
  {"id": 16, "replay_noise": 0},
  {"id": 17, "replay_noise": 1e-4},
  {"id": 18, "replay_noise": 1e-3},
  {"id": 19, "replay_noise": 1e-2},
  {"id": 20, "tau": 1e-3}, # Worked kinda
  {"id": 21, "tau": 1e-4},
  {"id": 22, "tau": 5e-2},
  {"id": 23, "tau": 0.1},
  {"id": 24, "gamma": 0.9}, # Worked kinda
  {"id": 25, "gamma": 0.999},
  {"id": 26, "gamma": 0.9999},
  {"id": 27, "gamma": 0.8},
]'''

'''experiments = [
  {"id": 30, "critic_lr": 1e-2},
  {"id": 31, "critic_lr": 1e-3},
  {"id": 32, "critic_lr": 1e-5},
  {"id": 33, "critic_lr": 1e-6},
  {"id": 34, "actor_lr": 1e-2},
  {"id": 35, "actor_lr": 1e-3},
  {"id": 36, "actor_lr": 1e-5},
  {"id": 37, "actor_lr": 1e-6},
]'''


'''experiments = [
  #{"id": 48}, # Twin critic, target policy smoothing on, results shitty
  #{"id": 49},
  #{"id": 50},
  #{"id": 51},
  {"id": 52}, # Single critic, critic-lr 1e-3, PER on, converged at first then diverged
  {"id": 53},
  {"id": 54},
  {"id": 55},
]'''

'''experiments = [
  {"id": 56}, # Single critic, PER on, critic lr = 1e-4, hold speed
  {"id": 57},
  {"id": 58},
  {"id": 59},
]'''


'''experiments = [
  {"id": 60}, # Single critic, PER on, critic lr = 1e-4, hold power
  {"id": 61, "critic_lr": 5e-5},
  {"id": 62, "critic_lr": 1e-5},
  {"id": 63, "actor_lr": 1e-5},  # Got closest
  {"id": 64, "prioritized_experience_replay": False}, 
  {"id": 65, "batch_size": 64},
  {"id": 66, "batch_size": 32},
  {"id": 67, "prioritized_experience_replay_alpha": 0.8, "prioritized_experience_replay_beta": 0.8},
  {"id": 68, "prioritized_experience_replay_alpha": 0.3},
  {"id": 69, "prioritized_experience_replay_alpha": 1, "prioritized_experience_replay_beta": 0.8},
]'''

# Hold lower power, actor lr 5e-5
'''experiments = [
  # {"id": 70}, # Disaster
  # {"id": 71, "critic_lr": 5e-5}, # Disaster
  # {"id": 72, "critic_lr": 1e-5}, # Disaster
  # {"id": 73, "actor_lr": 1e-5}, $ Disaster
  {"id": 74, "prioritized_experience_replay": False}, 
  {"id": 75, "batch_size": 64},
  {"id": 76, "batch_size": 32},
  {"id": 77, "prioritized_experience_replay_alpha": 0.8, "prioritized_experience_replay_beta": 0.8},
  {"id": 78, "prioritized_experience_replay_alpha": 0.3},
  {"id": 79, "prioritized_experience_replay_alpha": 1, "prioritized_experience_replay_beta": 0.8},
]'''

experiments = [
  {"id": 81},
  {"id": 82},
  {"id": 83},
  {"id": 84},
]

def run_one(experiment):
  id = experiment['id']
  del experiment['id']

  test_output = 'output_%d' % id
  checkpoint_dir = 'checkpoints/autorun_%d' % id

  experiment['test_output'] = test_output
  experiment['checkpoint_dir'] = checkpoint_dir
  experiment['run_name'] = 'id%d' % id

  os.makedirs(checkpoint_dir, exist_ok=True)

  hparams = ','.join([ ('%s=%s' % (key, val)) for key, val in experiment.items() ])

  print('Launching experiment %d with hparams %s' % (id, hparams))

  try:
    handle = subprocess.run('xvfb-run -a python main.py --hparams=%s' % hparams, shell=True)
    handle.wait()

    with open(test_output, 'r') as f:
      data = json.loads(f.read())

    return (id, data['total_reward'], data['total_deaths'])
  except:
    return (id, -1, -1)

# Set the number of parallel runs here
pool = Pool(4)
results = pool.map(run_one, experiments)

for (id, reward, deaths) in results:
  print("Run %d, reward: %f, deaths: %d" % (id, reward, deaths))