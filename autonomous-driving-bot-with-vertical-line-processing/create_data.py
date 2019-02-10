from gamescreen import getgamescreen
from keylog import getkeys
import win32api
import numpy as np
import time

## i prefer to convert each key we will use to one-hot arrays
## that way we can use categorical loss functions for optimizing
W = [1,0,0,0,0,0,0]
A = [0,1,0,0,0,0,0]
S = [0,0,1,0,0,0,0]
D = [0,0,0,1,0,0,0]
WA =[0,0,0,0,1,0,0]
WD =[0,0,0,0,0,1,0]
NT =[0,0,0,0,0,0,1]

def keys():
    output=[0,0,0,0,0,0,1]  #current output is non key pressing
    keys=getkeys()          #get pressing keys
    if 'W' in keys:         #control with if else statements
        if 'A' in keys:
            output=WA
        elif 'D' in keys:
            output=WD
        else:
            output=W
    elif 'A' in keys:
        output=A
    elif 'S' in keys:
        output=S
    elif 'D' in keys:
        output=D
    return output

lt=time.time()  
loop=False
val=0
data=[]
while True:
    if loop:
        img = getgamescreen()                       #get frame from car camera
        pk = keys()                                 #get which key i press
        data.append([img,pk])                       #append [frame,key] to data
        if len(data)==500:
            np.save('data/'+str(val)+'.npy',data)   #save data
            print("Data"+str(val)+" Saved")
            val+=1
            data=[]
        if win32api.GetAsyncKeyState(ord('P')):
            loop=False
            print("Stopped")
            time.sleep(1)
        
    else:
        if win32api.GetAsyncKeyState(ord('P')):
            loop=True
            print("Starting")
            time.sleep(0.2)
    time.sleep(0.2)
