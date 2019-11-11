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

experiments = [
  {"id": 6, "critic_lr": 1e-2},
  {"id": 7, "critic_lr": 1e-3},
  {"id": 8, "critic_lr": 1e-5},
  {"id": 9, "critic_lr": 1e-6},
  {"id": 10, "actor_lr": 1e-2},
  {"id": 10, "actor_lr": 1e-3},
  {"id": 10, "actor_lr": 1e-5},
  {"id": 10, "actor_lr": 1e-6},
]


def run_one(experiment):
  id = experiment['id']
  del experiment['id']

  test_output = 'output_%d' % id
  checkpoint_dir = 'checkpoints/autorun_%d' % id

  experiment['test_output'] = test_output
  experiment['checkpoint_dir'] = checkpoint_dir

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
pool = Pool(3)
results = pool.map(run_one, experiments)

for (id, reward, deaths) in results:
  print("Run %d, reward: %f, deaths: %d" % (id, reward, deaths))