from grabscreen import grab_screen
import cv2

def getgamescreen():
##  get the game screen resize it and turn it to grayscale
    return cv2.cvtColor(cv2.resize(grab_screen((0,300,800,550)),(320,100)),cv2.COLOR_BGR2GRAY)

