import ctypes
import time
import numpy as np
import itertools

class QBladeAdapter:
  def __init__(self):
    qbladeLib = ctypes.cdll.LoadLibrary('../qblade_build/libQBlade.so')
    self._qbladeLib = qbladeLib
    self._createInstance = qbladeLib._Z14createInstancev
    self._initializeSimulation = qbladeLib._Z20initializeSimulationi
    self._loadProject = qbladeLib._Z11loadProjectPc
    self._loadProject.argtypes = [ctypes.c_char_p]
    self._storeProject = qbladeLib._Z12storeProjectPc
    self._storeProject.argtypes = [ctypes.c_char_p]
    self._setControlVars = qbladeLib._Z14setControlVarsPd
    self._setControlVars.argtypes = [ctypes.POINTER(ctypes.c_double)]
    self._getControlVars = qbladeLib._Z14getControlVarsPd
    self._setControlVars.argtypes = [ctypes.POINTER(ctypes.c_double)]
    self._advanceSingleTimestep = qbladeLib._Z21advanceSingleTimestepv

    self._qbladeLib._Z14createInstancev()
    x = ctypes.c_char_p(b"../sample_projects/NREL_5MW_STR.wpa")
    self._loadProject(x)
    self._initializeSimulation(ctypes.c_int(0))

    self.lastAction = np.zeros(self.get_act_dim())

  def reset(self):
    # Resetting means 20 timesteps with constant control inputs
    for _ in range(0, 20):
      self._advanceSingleTimestep()

    return self.extractObservation()

  def get_obs_dim(self):
    #return 23
    return 22

  def get_act_dim(self):
    #return 5
    return 5

  # TODO get actual values for this
  def get_act_high(self):
    #return [10, 10, 10, 10, 10]
    return [3.94e6, 1e-5, 45, 45, 45]

  # TODO get actual values for this
  def get_act_low(self):
    #return [0, 0, 0, 0, 0]
    return [0, 0, 0, 0, 0]

  def calc_reward(self, observation):
    return np.clip(observation[1], 0, None) - 1e-6*(observation[10] + observation[11] + observation[12])

  def calc_death(self, observation):
    return False

  def padAction(self, action):
    def choose(a, b):
      if(a == None):
        return b
      return a
    return [choose(a, null) for a, null in itertools.zip_longest(action, np.ones(5))]

  def storeAction(self, action):
    # Copy action to control vars
    action = np.clip(action, self.get_act_low(), self.get_act_high())
    action = np.nan_to_num(self.padAction(action))

    in_data = (ctypes.c_double * 5)(*action)
    self._setControlVars(in_data)

    self.lastAction = action

  def extractObservation(self):
    # 23 values are hardcoded in the library
    out_data = (ctypes.c_double * 23)()
    self._getControlVars(out_data)
    observation = [out_data[i] for i in range(0, self.get_obs_dim())]
    observation = np.nan_to_num(observation)

    return observation

  def step(self, action):

    self.storeAction(action)
    self._advanceSingleTimestep()

    observation = self.extractObservation()
    return observation, self.calc_reward(observation), self.calc_death(observation)

  def render(self):
    pass

  def close(self):
    del self._qbladeLib

  def storeProject(self, filename):
    x = ctypes.c_char_p(bytes(filename, 'utf-8'))
    self._storeProject(x)