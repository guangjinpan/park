# from park.envs.Position_XR.Position_XR import PositionXREnv
from park.envs.Position_XR.Position_XR import PositionXREnv
import numpy as np

BS_number=4 
User_number = 10

env=PositionXREnv(BS_number=4, User_number = 10)
env.reset()
for i in range(1):
    action=np.random.rand(1, User_number *2 + BS_number+ User_number* BS_number)
    action=action[0,:]    
    env.step(action)
    
