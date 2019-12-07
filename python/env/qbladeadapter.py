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
  def __init__(self, project="../sample_projects/NREL_5MW_STR.wpa"):
    self.project = project
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
    x = ctypes.c_char_p(bytes(self.project, 'utf-8'))
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
    #for i in range(0, 100):
    #  self._advanceSingleTimestep()

    print('reset done')

    return self.maskObservation(self.extractObservation())

  def get_obs_dim(self):
    #return 23
    return 2

  def get_act_dim(self):
    #return 5
    return 2

  def get_act_high(self):
    gearbox_ratio = 97
    generator_torque = 47402.91
    return [generator_torque*gearbox_ratio, 90]

  # TODO get actual values for this
  def get_act_low(self):
    #return [0, 0, 0, 0, 0]
    return [0,0]

  def get_act_max_grad(self):
    timestep = 0.1
    gearbox_ratio = 97
    torque_step = 15000
    pitch_step = 8

    return [torque_step*gearbox_ratio*timestep, pitch_step*timestep]

  def get_obs_labels(self):
    return {
      0: 'rotational speed [rad/s]',
      1: 'power [kW]',
      2: 'HH wind velocity [m/s]',
      3: 'yaw angle [deg]',
      4: 'pitch blade 1 [deg]',
      5: 'pitch blade 2 [deg]',
      6: 'pitch blade 3 [deg]',
      7: 'tower top bending local x [Nm]',
      8: 'tower top bending local y [Nm]',
      9: 'tower top bending local z [Nm]',
      10: 'oop bending blade 1 [Nm]',
      11: 'oop bending blade 2 [Nm]',
      12: 'oop bending blade 3 [Nm]',
      13: 'ip bending blade 1 [Nm]',
      14: 'ip bending blade 2 [Nm]',
      15: 'ip bending blade 3 [Nm]',
      16: 'oop tip deflection blade 1 [m]',
      17: 'oop tip deflection blade 2 [m]',
      18: 'oop tip deflection blade 3 [m]',
      19: 'ip tip deflection blade 1 [m]',
      20: 'ip tip deflection blade 2 [m]',
      21: 'ip tip deflection blade 3 [m]',
      22: 'current time [s]'
    }

  def get_act_labels(self):
    return {
      0: 'generator torque [Nm]',
      1: 'collective pitch [deg]',
      #1: 'yaw angle [deg]',
      #2: 'pitch blade 1 [deg]',
      #3: 'pitch blade 2 [deg]',
      #4: 'pitch blade 3 [deg]'
    }

  def calc_reward(self, observation, action, death):
    if(death):
      death_penalty = 0.5 if observation[0] > 0 else 3
    else:
      death_penalty = 0

    rated_power = 3500
    rated_speed = 0.8
    #return -np.abs(observation[1]-rated_power) - 1e3*(np.abs(observation[16]) + np.abs(observation[17]) + np.abs(observation[18]))
    return np.clip(1-np.abs((observation[1]-rated_power)/rated_power)-death_penalty, -4, 4)
    #return np.clip(observation[1], 0, None) - 1e4*(np.abs(observation[16]) + np.abs(observation[17]) + np.abs(observation[18]))
    #return np.clip(5 
    #  - np.abs((observation[1]-rated_power)/rated_power)
    #  - 5e-2*(np.abs(observation[16]) + np.abs(observation[17]) + np.abs(observation[18])), -10, 10)
    #return np.clip(1-np.abs((observation[0]-rated_speed)/rated_speed)-death_penalty, -4, 4)

  def calc_death(self, observation):
    # If there are nan values, reload completely
    self.steps_since_reload += 100000

    if(np.any(np.isnan(observation))):
      return True

    observation = np.nan_to_num(observation)

    # 63 is the rotor size, so if anything bends further than that, it's broken off.
    broken_state = 20
    max_rotorspeed = 3
    min_rotorspeed = -0.05
    max_power = 10000
    return np.abs(observation[16]) > broken_state or \
           np.abs(observation[17]) > broken_state or \
           np.abs(observation[18]) > broken_state or \
           observation[0] > max_rotorspeed or \
           observation[0] < min_rotorspeed or \
           observation[1] > max_power

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