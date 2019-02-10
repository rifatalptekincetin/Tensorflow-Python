## this is the file if i want to edit data i edit it with this file

from grabscreen import grab_screen
from gamescreen import getgamescreen
from getspeed import get_speed
from keylog import getkeys
import cv2, win32api, win32gui
import numpy as np
import random
import time
bdata=[]
data=[]
for i in range(43):
    for x in np.load("data/"+str(i)+".npy"):
        if x[2]==[0,0,0,0,0,0,1]:
            if random.randrange(0,9)==1:
                bdata.append(x)
        else:
            data.append(x)
