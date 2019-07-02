import ctypes
import time


qbladeLib = ctypes.cdll.LoadLibrary('code/trunk/libQBlade.so')
createInstance = qbladeLib._Z14createInstancev
loadProject = qbladeLib._Z11loadProjectPc
loadProject.argtypes = [ctypes.c_char_p]
startAnalysis = qbladeLib._Z13startAnalysisv

createInstance()
x = ctypes.c_char_p(b"sample_projects/NREL_5MW.wpa")
loadProject(x)

print('library loaded')

startAnalysis()