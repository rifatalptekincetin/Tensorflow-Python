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
    gg=tf.unstack(x[6],2,1)
    gg=[tf.unstack(i,11,1) for i in gg]
    
    bb=tf.concat([RNN([bb[i],x[0],tf.matmul(gg[0][i],W["date"])],playerteamdatecell) for i in range(11)],1)
    cc=RNN(cc,lastgamescell)
    ee=tf.concat([RNN([ee[i],x[3],tf.matmul(gg[1][i],W["date"])],playerteamdatecell) for i in range(11)],1)
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

def test(data,datedata):
    batch=1
    phs=[tf.placeholder("float", [batch,1],name="lt"),
    tf.placeholder("float",[batch,11],name="ltp"),
    tf.placeholder("float",[batch,5],name="ltlg"),
    tf.placeholder("float", [batch,1],name="rt"),
    tf.placeholder("float",[batch,11],name="rtp"),
    tf.placeholder("float",[batch,5],name="rlg"),
    tf.placeholder("float", [batch,2,11,3],name="d")]
    y=tf.placeholder("float",[batch,3],name="y")
    logits=getlogits(phs,batch)
    loss_op=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                       labels=y))
    prediction=tf.nn.softmax(logits)
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1),tf.argmax(y,1)),tf.float32))
    init=tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess: 
        saver.restore(sess,'model3/model')
        totacc,totloss=0,0
        for i in range(int(len(data[0])/batch)):
            [loss,acc]=sess.run([loss_op,accuracy],feed_dict={
                        phs[0]:[data[0][i]],
                        phs[1]:[data[1][i]],
                        phs[2]:[data[2][i]],
                        phs[3]:[data[3][i]],
                        phs[4]:[data[4][i]],
                        phs[5]:[data[5][i]],
                        phs[6]:[datedata[i]],
                        y:[data[7][i]]})
            totacc+=(acc/(len(data[0])/batch))
            totloss+=(loss/(len(data[0])/batch))
        print("Loss: {:.4f}, Acc: {:.4f}".format(totloss,totacc))

import numpy as np


def data2input(data):
    ndata=[[] for _ in range(8)]
    for i in data:
        for _ in range(8):
            ndata[_].append(i[_])
    return ndata

def getdatedict(data):
    #after data2input
    pdd={}
    for i in range(len(data[1])):
        for j in range(11):
            pdd[data[1][i][j]]=data[6][i]
            pdd[data[4][i][j]]=data[6][i]
    return pdd

data=np.load("data/ingiltere-1298.npy")[0:1280]
data=data2input(data)

playerdatedict=getdatedict(data)

data=np.load("data/ingiltere-1298.npy")[1280:1298]
data=data2input(data)

datedata=[]
for i in range(len(data[1])):
	l=[]
	r=[]
	for j in range(11):
		try:
		     l.append(playerdatedict[data[1][i][j]])
		except:
		     l.append(data[6][i])
		try:
		     r.append(playerdatedict[data[4][i][j]])
		except:
		     r.append(data[6][i])
	datedata.append([l,r])

test(data,datedata)
