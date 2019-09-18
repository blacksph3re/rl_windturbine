import ctypes
import time



class QBladeAdapter:
  def __init__(self):
    qbladeLib = ctypes.cdll.LoadLibrary('../qblade_build/libQBlade.so')
    self.qbladeLib = qbladeLib
    self.createInstance = qbladeLib._Z14createInstancev
    self.initializeSimulation = qbladeLib._Z20initializeSimulationv
    self.loadProject = qbladeLib._Z11loadProjectPc
    self.loadProject.argtypes = [ctypes.c_char_p]
    self.setControlVars = qbladeLib._Z14setControlVarsPd
    self.setControlVars.argtypes = [ctypes.POINTER(ctypes.c_double)]
    self.getControlVars = qbladeLib._Z14getControlVarsPd
    self.setControlVars.argtypes = [ctypes.POINTER(ctypes.c_double)]
    self.advanceSingleTimestep = qbladeLib._Z21advanceSingleTimestepv

    self.qbladeLib._Z14createInstancev()
    x = ctypes.c_char_p(b"../sample_projects/NREL_5MW_STR.wpa")
    self.loadProject(x)
    self.initializeSimulation()

  def reset(self):
    # Resetting means 200 timesteps with constant control inputs
    for _ in range(0, 200):
      self.advanceSingleTimestep()

  def get_obs_dim(self):
    return 23

  def get_act_dim(self):
    return 5

  # TODO get actual values for this
  def get_act_high(self):
    return [1, 1, 1, 1, 1]

  # TODO get actual values for this
  def get_act_low(self):
    return [-1, -1, -1, -1, -1]

  def calc_reward(self, observation):
    return observation[1]

  def calc_death(self, observation):
    return False

  def storeAction(self, action):
    # Copy action to control vars
    in_data = (ctypes.c_double * self.get_act_dim())(*action)
    self.setControlVars(in_data)

  def extractObservation(self):
    out_data = (ctypes.c_double * self.get_obs_dim())()
    self.getControlVars(out_data)
    observation = [out_data[i] for i in range(0, self.get_obs_dim())]

    return observation

  def step(self, action):

    self.storeAction(action)
    self.advanceSingleTimestep()

    observation = self.extractObservation()
    return observation, self.calc_reward(observation), self.calc_death(observation)

  def render(self):
    pass

  def close(self):
    return self.env.close()