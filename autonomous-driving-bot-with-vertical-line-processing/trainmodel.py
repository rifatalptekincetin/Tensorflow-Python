import tensorflow as tf
import numpy as np

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
##    x=tf.nn.dropout(x,keep_prob=.8)
    return tf.matmul(x,W["out"])+b["out"]

def train(xs,ys,epochs=100,batch=50,lr=.001,restore=False):
    x = tf.placeholder(tf.float32, [None, 60, 320])
    y = tf.placeholder(tf.float32, [None, 7])
    logits=getlogits(x)
    loss_op=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                      labels=y))
    
    correct_pred=tf.equal(tf.argmax(tf.nn.softmax(logits),1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    train_op=tf.train.AdamOptimizer(lr).minimize(loss_op)

    init=tf.global_variables_initializer()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        if !restore:
            sess.run(init)
        else:
            saver.restore(sess,"model/model")
        for epoch in range(epochs):
            for sl in range(int(len(xs)/batch)):
                sess.run(train_op,feed_dict={
                    x:xs[sl*batch:(sl+1)*batch],
                    y:ys[sl*batch:(sl+1)*batch]
                    })
            if epoch%10==0:
                loss,acc=sess.run([loss_op,accuracy],feed_dict={
                    x:xs,
                    y:ys
                    })
                print("epoch={}, acc={:.4f}, loss={:.4f}".format(epoch,acc,loss))
        saver.save(sess,"model/model")
        print("Session Saved!")
        
def data2xsys(data):
    xs=[]
    ys=[]
    for x in data:
        xs.append(x[0])
        ys.append(x[2])
    return xs,ys

data=np.load('data.npy')
xs,ys=data2xsys(data)

train(xs,ys,epochs=100,batch=50,lr=.001,restore=False)
