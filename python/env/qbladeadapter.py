import ctypes
import _ctypes
import time
import numpy as np
import itertools
import os

def isLoaded(lib):
  libp = os.path.abspath(lib)
  ret = os.system("lsof -p %d | grep %s" % (os.getpid(), libp))
  print(ret)
  return (ret == 0)

def unloadLib(handle, path):
  tries = 0
  while(isLoaded(path)):
    tries += 1
    _ctypes.dlclose(handle)
    print('Unloading, try %d' % tries)
    if(tries > 150):
      break


class QBladeAdapter:
  def __init__(self):
    self._qbladeLib = None

    print("loading qblade library")
    qbladeLib = ctypes.CDLL('../qblade_build/libQBlade.so')
    self._qbladeLib = qbladeLib
    self.map_functions(qbladeLib)

    print("creating qblade instance")
    self._qbladeLib._Z14createInstancev()
    self.load_default_project()

    print("qblade init done")

  def map_functions(self, qbladeLib):
    self._createInstance = qbladeLib._Z14createInstancev
    self._initializeSimulation = qbladeLib._Z20initializeSimulationi
    self._loadProject = qbladeLib._Z11loadProjectPc
    self._loadProject.argtypes = [ctypes.c_char_p]
    self._storeProject = qbladeLib._Z12storeProjectPc
    self._storeProject.argtypes = [ctypes.c_char_p]
    self._setControlVars = qbladeLib._Z14setControlVarsPd
    self._setControlVars.argtypes = [ctypes.POINTER(ctypes.c_double)]
    self._getControlVars = qbladeLib._Z14getControlVarsPd
    self._getControlVars.argtypes = [ctypes.POINTER(ctypes.c_double)]
    self._advanceSingleTimestep = qbladeLib._Z21advanceSingleTimestepv

  def del_functions(self):
    del self._createInstance
    del self._initializeSimulation
    del self._loadProject
    del self._storeProject
    del self._setControlVars
    del self._getControlVars
    del self._advanceSingleTimestep

  def load_default_project(self):
    x = ctypes.c_char_p(b"../sample_projects/NREL_5MW_STR.wpa")
    print("loading qblade project")
    self._loadProject(x)
    self.steps_since_reload = 0

  def reset(self):
    print('Resetting simulation')
    if(self.steps_since_reload>50000):
      self.load_default_project()
    self._initializeSimulation(ctypes.c_int(0))
    self.lastAction = np.zeros(5)
    self.storeAction(np.zeros(self.get_act_dim()))

    # Do some steps with zero action
    action = np.zeros(self.get_act_dim())
    self.storeAction(action)
    for i in range(0, 100):
      self._advanceSingleTimestep()

    print('reset done')

    return self.maskObservation(self.extractObservation())

  def get_obs_dim(self):
    #return 23
    return 2

  def get_act_dim(self):
    #return 5
    return 2

  def get_act_high(self):
    #return [10, 10, 10, 10, 10]
    return [474029.1, 90]

  # TODO get actual values for this
  def get_act_low(self):
    #return [0, 0, 0, 0, 0]
    return [0, 0]

  def get_act_max_grad(self):
    timestep = 0.1
    return [150000*timestep, 8*timestep]


  def calc_reward(self, observation, action, death):
    if(death):
      death_penalty = 0.5 if observation[0] > 0 else 3
    else:
      death_penalty = 0

    rated_power = 3200
    rated_speed = 0.8
    #return -np.abs(observation[1]-rated_power) - 1e3*(np.abs(observation[16]) + np.abs(observation[17]) + np.abs(observation[18]))
    #return 1-np.abs((observation[1]-rated_power)/rated_power)-death_penalty
    #return np.clip(observation[1], 0, None) - 1e4*(np.abs(observation[16]) + np.abs(observation[17]) + np.abs(observation[18]))
    #return np.clip(5 
    #  - np.abs((observation[1]-rated_power)/rated_power)
    #  - 5e-2*(np.abs(observation[16]) + np.abs(observation[17]) + np.abs(observation[18])), -10, 10)
    return np.clip(1-np.abs((observation[0]-rated_speed)/rated_speed)-death_penalty, -4, 4)

  def calc_death(self, observation):
    # If there are nan values, reload completely
    self.steps_since_reload += 100000

    if(np.any(np.isnan(observation))):
      return True

    observation = np.nan_to_num(observation)

    # 63 is the rotor size, so if anything bends further than that, it's broken off.
    broken_state = 20
    return np.abs(observation[16]) > broken_state or \
           np.abs(observation[17]) > broken_state or \
           np.abs(observation[18]) > broken_state or \
           observation[0] > 3 or \
           observation[0] < -0.5

  def padAction(self, action):
    action = [action[0], 0, action[1], action[1], action[1]]
    return action

  def storeAction(self, action):
    # Copy action to control vars
    action = np.clip(action, self.get_act_low(), self.get_act_high())
    action = np.nan_to_num(self.padAction(action))

    in_data = (ctypes.c_double * 5)(*action)
    self._setControlVars(in_data)

    self.lastAction = action

  def extractObservation(self, remove_nan=True):
    # 23 values are hardcoded in the library
    out_data = (ctypes.c_double * 23)()
    self._getControlVars(out_data)
    observation = [out_data[i] for i in range(0, 23)]

    if(remove_nan):
      observation = np.nan_to_num(observation)

    return observation

  def maskObservation(self, obs):
    return np.array(obs[0:2])

  def step(self, action):
    self.steps_since_reload += 1

    self.storeAction(action)
    self._advanceSingleTimestep()

    observation = self.extractObservation(False)
    death = self.calc_death(observation)
    observation = np.nan_to_num(observation)
    reward = self.calc_reward(observation, action, death)
    observation = self.maskObservation(observation)

    return observation, reward, death

  def render(self):
    pass

  def close(self):
    del self._qbladeLib

  def storeProject(self, filename):
    x = ctypes.c_char_p(bytes(filename, 'utf-8'))
    self._storeProject(x)