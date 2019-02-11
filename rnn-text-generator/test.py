import tensorflow as tf
import numpy as np
from readtext import text2data,loaddict,data2text

nclasses=int(len(loaddict())/2)
seqlen=5
batch=1
textlen=500
rnnlayers=3
rnnlayers=[nclasses*seqlen for i in range(rnnlayers)]

data=text2data(open("data.txt").read()[0:seqlen],False)

x=tf.placeholder(tf.float32,[batch,seqlen])

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

saver=tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "model/model.ckpt")
    ss=tuple([(np.zeros([batch, i]),np.zeros([batch, i])) for i in rnnlayers])
    for i in range(textlen):
        ret,ss=sess.run([tf.argmax(logits,1),states],
            feed_dict={
            x:[data[-seqlen:]],
            states:ss
            })
        data.append(ret[0])
        text=data2text(data)
    print(text)
        
