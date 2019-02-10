from grabscreen import grab_screen
import cv2
import win32gui, win32api
import numpy as np

## read the digit images from digits folder
digits=[cv2.imread('digits/'+str(i)+'.jpg',0) for i in range(10)]
dictdig={0:.93,1:.925,2:.87,3:.93,4:.9225,5:.92225,6:.922,7:.93,8:.91845,9:.9285}
def get_speed():
    #get the speed bar from screen convert it to grayscale and threshold it
    img=cv2.cvtColor(grab_screen((687,561,757,596)),cv2.COLOR_BGR2GRAY)
    _,img = cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)
    numinimg=''
    cpt=0
    for i in range(10):
        w, h = digits[i].shape[::-1]
        #search for matches between digit-img and speed screen
        res = cv2.matchTemplate(img,digits[i],cv2.TM_CCOEFF_NORMED)
        loc = np.where( res >= dictdig[i])
        for pt in zip(*loc[::-1]):
            if pt[0]>cpt:
                numinimg+=str(i)
                cpt=pt[0]
            else:
                numinimg=str(i)+numinimg
                cpt=pt[0]
    if numinimg:
        numinimg=int(numinimg)
    else:
        numinimg=0
    return numinimg
