import tensorflow as tf
import win32api
import time
from gamescreen import getgamescreen
from directkeys import keypress
from getspeed import get_speed

W = {
    "out": tf.Variable(tf.random_normal([320,7]),name="W3"),
    "filter1":tf.Variable(tf.random_normal([60,320,320]),name="W3"),
    }
b = {
    "out": tf.Variable(tf.random_normal([7]),name="B"),
    }

def getlogits(x):
    x=tf.nn.convolution(x,W["filter1"],"SAME",[60])
    x=tf.reshape(x,[-1,320])
    return tf.matmul(x,W["out"])+b["out"]

x=tf.placeholder(tf.float32,[None,60,320])

logits=getlogits(x)

saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'model2/model')
    loop=False
    print("Press 'P' 4 Start !!")
    lt=time.time()
    curr=1
    speed=0
    while True:
        if loop:
            img = getgamescreen()
            speed=get_speed()
            if (speed!=0 and speed < (curr-9)):
                speed=curr
            if curr!=speed:
                curr=speed
            if speed > 20:
                keypress(2)
                time.sleep(0.05)
            kfp=sess.run(tf.argmax(logits,1),feed_dict={x:[img]})
##            print(kfp)
            keypress(kfp[0])
            time.sleep(0.2)
            keypress(6)
            if win32api.GetAsyncKeyState(ord('P')):
                loop=False
                print("Stopped !")
                time.sleep(1)
            print(time.time()-lt)
            lt=time.time()
        else:
            if win32api.GetAsyncKeyState(ord('P')):
                loop=True
                print("On Loop !")
                time.sleep(1)
