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
    return 1

  # TODO get actual values for this
  def get_act_high(self):
    #return [10, 10, 10, 10, 10]
    return [100]

  # TODO get actual values for this
  def get_act_low(self):
    #return [0, 0, 0, 0, 0]
    return [0]

  def calc_reward(self, observation):
    return observation[0]*self.lastAction[0]

  def calc_death(self, observation):
    return False

  def logObservation(self, writer, step, observation):
    writer.add_scalar('obs/rotational speed', observation[0], step)
    writer.add_scalar('obs/power', observation[1], step)
    writer.add_scalar('obs/HH wind velocity', observation[2], step)
    writer.add_scalar('obs/yaw angle', observation[3], step)
    writer.add_scalar('obs/pitch blade 1', observation[4], step)
    writer.add_scalar('obs/pitch blade 2', observation[5], step)
    writer.add_scalar('obs/pitch blade 3', observation[6], step)
    writer.add_scalar('obs/tower top bending local x', observation[7], step)
    writer.add_scalar('obs/tower top bending local y', observation[8], step)
    writer.add_scalar('obs/tower top bending local z', observation[9], step)
    writer.add_scalar('obs/oop bending blade 1', observation[10], step)
    writer.add_scalar('obs/oop bending blade 2', observation[11], step)
    writer.add_scalar('obs/oop bending blade 3', observation[12], step)
    writer.add_scalar('obs/ip bending blade 1', observation[13], step)
    writer.add_scalar('obs/ip bending blade 2', observation[14], step)
    writer.add_scalar('obs/ip bending blade 3', observation[15], step)
    writer.add_scalar('obs/oop tip deflection blade 1', observation[16], step)
    writer.add_scalar('obs/oop tip deflection blade 2', observation[17], step)
    writer.add_scalar('obs/oop tip deflection blade 3', observation[18], step)
    writer.add_scalar('obs/ip tip deflection blade 1', observation[19], step)
    writer.add_scalar('obs/ip tip deflection blade 2', observation[20], step)
    writer.add_scalar('obs/ip tip deflection blade 3', observation[21], step)
    #writer.add_scalar('obs/current time', observation[22], step)

  def logAction(self, writer, step, action):
    action = self.padAction(action)
    writer.add_scalar('act/generator torque', action[0], step)
    writer.add_scalar('act/yaw angle', action[1], step)
    writer.add_scalar('act/pitch blade 1', action[2], step)
    writer.add_scalar('act/pitch blade 2', action[3], step)
    writer.add_scalar('act/pitch blade 3', action[4], step)

  def padAction(self, action):
    def choose(a, b):
      if(a == None):
        return b
      return a
    return [choose(a, null) for a, null in itertools.zip_longest(action, np.zeros(5))]

  def storeAction(self, action):
    # Copy action to control vars
    my_action = self.padAction(action)
    in_data = (ctypes.c_double * 5)(*my_action)
    self._setControlVars(in_data)

    self.lastAction = action

  def extractObservation(self):
    # 23 values are hardcoded in the library
    out_data = (ctypes.c_double * 23)()
    self._getControlVars(out_data)
    observation = [out_data[i] for i in range(0, self.get_obs_dim())]

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