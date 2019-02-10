import tensorflow as tf
from time import sleep

W = {
    0: tf.Variable(tf.random_normal([960,3]),name="W0"),
    1: tf.Variable(tf.random_normal([960,3]),name="W1"),
    2: tf.Variable(tf.random_normal([960,3]),name="W2"),
    3: tf.Variable(tf.random_normal([960,3]),name="W3"),
    "players":tf.Variable(tf.random_normal([440,440]),name="players"),
    "lastgames":tf.Variable(tf.random_normal([40,40]),name="lastgames"),
    "date": tf.Variable(tf.random_normal([3,1]),name="date"),
    }
b = {
    "out": tf.Variable(tf.random_normal([3]),name="out"),
    }
playerteamdatecell=tf.nn.rnn_cell.BasicLSTMCell(40, forget_bias=0.0)
lastgamescell=tf.nn.rnn_cell.BasicLSTMCell(40, forget_bias=0.0)

def RNN(listx,cell):
    hidden_state = tf.zeros([listx[0].shape[0], cell.state_size[0]])
    current_state = tf.zeros([listx[0].shape[0], cell.state_size[0]])
    state = hidden_state, current_state
    for x in listx:
        output,state=cell(x,state)
    return output
    
def getlogits(x,batch):
    # 0 leftteam 1 leftteamplayers 2 leftteamlastgames
    # 3 rightteam 4 rightteamplayers 5 rightteamlastgames 6 date
    bb=tf.unstack(tf.reshape(x[1],[batch,11,1]),11,1)
    cc=tf.unstack(tf.reshape(x[2],[batch,5,1]),5,1)
    ee=tf.unstack(tf.reshape(x[4],[batch,11,1]),11,1)
    ff=tf.unstack(tf.reshape(x[5],[batch,5,1]),5,1)
    gg=tf.matmul(x[6],W["date"])
    
    bb=tf.concat([RNN([i,x[0],gg],playerteamdatecell) for i in bb],1)
    cc=RNN(cc,lastgamescell)
    ee=tf.concat([RNN([i,x[3],gg],playerteamdatecell) for i in ee],1)
    ff=RNN(ff,lastgamescell)

    bb=tf.matmul(bb,W["players"])
    cc=tf.matmul(cc,W["lastgames"])
    ee=tf.matmul(ee,W["players"])
    ff=tf.matmul(ff,W["lastgames"])

    x=tf.concat([bb,cc,ee,ff],1)
    x=tf.nn.l2_normalize(x)
    x4=tf.matmul(tf.pow(x,4),W[2])
    x3=tf.matmul(tf.pow(x,3),W[2])
    x2=tf.matmul(tf.pow(x,2),W[1])
    x=tf.matmul(x,W[0])
    return x4+x3+x2+x+b["out"]

def train(data,epochs=100,batch=50,lr=.1):
    phs=[tf.placeholder("float", [batch,1],name="lt"),
    tf.placeholder("float",[batch,11],name="ltp"),
    tf.placeholder("float",[batch,5],name="ltlg"),
    tf.placeholder("float", [batch,1],name="rt"),
    tf.placeholder("float",[batch,11],name="rtp"),
    tf.placeholder("float",[batch,5],name="rlg"),
    tf.placeholder("float", [batch,3],name="d")]
    y=tf.placeholder("float",[batch,3],name="y")
    logits=getlogits(phs,batch)
    loss_op=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                       labels=y))
    optimizer=tf.train.AdamOptimizer(lr)
    train_op=optimizer.minimize(loss_op)
    prediction=tf.nn.softmax(logits)
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1),tf.argmax(y,1)),tf.float32))
    init=tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess: 
##        sess.run(init)
        saver.restore(sess,'model/model')
        for epoch in range(epochs):
            totacc,totloss=0,0
            for i in range(int(len(data[0])/batch)):
                sess.run(train_op,feed_dict={
                    phs[0]:data[0][i*batch:(i+1)*batch],
                    phs[1]:data[1][i*batch:(i+1)*batch],
                    phs[2]:data[2][i*batch:(i+1)*batch],
                    phs[3]:data[3][i*batch:(i+1)*batch],
                    phs[4]:data[4][i*batch:(i+1)*batch],
                    phs[5]:data[5][i*batch:(i+1)*batch],
                    phs[6]:data[6][i*batch:(i+1)*batch],
                    y:data[7][i*batch:(i+1)*batch]})
            if epoch%10==0:
                for i in range(int(len(data[0])/batch)):
                    [loss,acc]=sess.run([loss_op,accuracy],feed_dict={
                        phs[0]:data[0][i*batch:(i+1)*batch],
                        phs[1]:data[1][i*batch:(i+1)*batch],
                        phs[2]:data[2][i*batch:(i+1)*batch],
                        phs[3]:data[3][i*batch:(i+1)*batch],
                        phs[4]:data[4][i*batch:(i+1)*batch],
                        phs[5]:data[5][i*batch:(i+1)*batch],
                        phs[6]:data[6][i*batch:(i+1)*batch],
                        y:data[7][i*batch:(i+1)*batch]})
                    totacc+=(acc/(len(data[0])/batch))
                    totloss+=(loss/(len(data[0])/batch))
                print("Epoch: {}, Loss: {:.8f}, Acc: {:.4f}".format(epoch,totloss,totacc))
        saver.save(sess,'model/model')

import numpy as np


def data2input(data):
    ndata=[[] for _ in range(8)]
    for i in data:
        for _ in range(8):
            ndata[_].append(i[_])
    return ndata

data=np.load("data/ingiltere-1298.npy")[0:1280]

data=data2input(data)
train(data,epochs=1000,batch=20,lr=.000001)
