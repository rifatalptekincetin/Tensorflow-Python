import tensorflow as tf
import numpy as np
from readtext import text2data,loaddict
from random import shuffle,randrange

def data2xsys(data):
    data=[[data[i:i+seqlen],data[i+seqlen]] for i in range(len(data)-seqlen)]
    xs,ys=[],[]
    for i in data:
        xs.append(i[0])
        ys.append(i[1])
    return xs,ys

nclasses=int(len(loaddict())/2)
seqlen=5
batch=100
rnnlayers=3
rnnlayers=[nclasses*seqlen for i in range(rnnlayers)]

data=text2data()

x=tf.placeholder(tf.float32,[batch,seqlen])
y=tf.placeholder(tf.int32,[batch])

W = {
    "fc1": tf.Variable(tf.random_normal([rnnlayers[-1],nclasses*seqlen]),name="fc1"),
    "fc2": tf.Variable(tf.random_normal([nclasses*seqlen,nclasses*seqlen]),name="fc2"),
    1: tf.Variable(tf.random_normal([seqlen,rnnlayers[0]]),name="W1"),
    0: tf.Variable(tf.random_normal([nclasses*seqlen,nclasses]),name="W0"),
    }

b = {
    1: tf.Variable(tf.random_normal([rnnlayers[0]]),name="b1"),
    0: tf.Variable(tf.random_normal([nclasses]),name="b0"),
    }

cell=tf.contrib.rnn.MultiRNNCell(
    [tf.contrib.rnn.LSTMBlockCell(rnnsize,forget_bias=0.0,reuse=tf.AUTO_REUSE) for rnnsize in rnnlayers]
    )

states=tuple([(tf.placeholder(tf.float32,[batch, rnnlayers[-1]]),
               tf.placeholder(tf.float32,[batch, rnnlayers[-1]])) for rnnsize in rnnlayers])

x1=tf.matmul(x,W[1])
x2,states_ret=cell(x1,states)
x3=tf.nn.dropout(x2,.99)
fc1=tf.matmul(x3,W["fc1"])
fc2=tf.matmul(fc1,W["fc2"])
logits=tf.matmul(fc2,W[0])+b[0]

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=tf.one_hot(y,nclasses),
    logits=logits))

lr=tf.placeholder(tf.float32,[])
optimizer=tf.train.AdamOptimizer(lr)
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()
saver=tf.train.Saver()

with tf.Session() as sess:
##    sess.run(init)
    saver.restore(sess, "model/model.ckpt")
    xs,ys=data2xsys(data)
    lrate=0.0001
    for epoch in range(1,1000):
        ss=tuple([(np.zeros([batch, i]),np.zeros([batch, i])) for i in rnnlayers])
        le=0
        for i in range(int((len(data)-seqlen)/batch)):
##            if randrange(0,50)==0:
##                ss=tuple([(np.zeros([batch, i]),np.zeros([batch, i])) for i in rnnlayers])
            _,ss,lb=sess.run([train,states,loss],
                feed_dict={
##                x:[data[j:j+seqlen] for j in range(i*batch,(i+1)*batch)],
##                y:[data[j+seqlen] for j in range(i*batch,(i+1)*batch)],
                x:xs[i*batch:(i+1)*batch],
                y:ys[i*batch:(i+1)*batch],
                states:ss,
                lr:lrate})
            le+=lb/int((len(data)-1)/batch)
        print("epoch: ",epoch,"loss: ",le)
        if epoch%10==0:
            print("Model saved in ",saver.save(sess, "model/model.ckpt"))
            if epoch%500==0:
                lrate=lrate/10
    print("Model saved in ",saver.save(sess, "model/model.ckpt"))
