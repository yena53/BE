import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import math
import random
import os

tf.set_random_seed(777)

batch_size = 100
# sequence_length = 25
chop_len = 198
mel_bin = 256
num_classes = 4
learning_rate = 0.0005
iterations = 100
ckpt_path = 'ckpt/'
data_path = ''

train = np.load(data_path+'train_256mel.npy')
# train = train[:200]
np.random.shuffle(train)
test = np.load(data_path+'test_256mel.npy')
testx,testy = np.hsplit(test,[-1])
testx = np.reshape(testx,[-1,256,chop_len,1])

data_size = train.shape[0]

tf.reset_default_graph()

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32,[None,mel_bin,chop_len])
    x = tf.image.per_image_standardization(x)
    x = tf.reshape(x,[-1,mel_bin,chop_len,1])
    y = tf.placeholder(tf.float32,[None,4])
    phase = tf.placeholder(tf.bool,[])

    w1 = tf.Variable(tf.random_normal([3,3,1,64],stddev=0.01))
    l1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding='SAME')
    # l1 = tf.contrib.layers.batch_norm(l1,center=True,scale=True,is_training=phase)
    l1 = tf.nn.relu(l1)
    l1 = tf.nn.max_pool(l1,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME')
    l1 = tf.layers.dropout(l1,0.1,training=phase)

    w2 = tf.Variable(tf.random_normal([3,3,64,64],stddev=0.01))
    l2 = tf.nn.conv2d(l1,w2,strides=[1,1,1,1],padding='SAME')
    # l2 = tf.contrib.layers.batch_norm(l2,center=True,scale=True,is_training=phase)
    l2 = tf.nn.relu(l2)
    l2 = tf.nn.max_pool(l2,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME')
    # l2 = tf.layers.dropout(l2,0.1,training=phase)
    
    w3 = tf.Variable(tf.random_normal([3,3,64,64],stddev=0.01))
    l3 = tf.nn.conv2d(l2,w3,strides=[1,1,1,1],padding='SAME')
    # l3 = tf.contrib.layers.batch_norm(l3,center=True,scale=True,is_training=phase)
    l3 = tf.nn.max_pool(l3,ksize=[1,6,1,1],strides=[1,6,1,1],padding='SAME')
    # l3 = tf.layers.dropout(l3,0.2,training=phase)
    l3_transpose = tf.transpose(l3,perm=[0,1,3,2])
    temp_shape = l3_transpose.get_shape().as_list()
    l3_reshaped = tf.reshape(l3_transpose,[-1,temp_shape[1]*temp_shape[2],temp_shape[3]])
    l3_reshaped = tf.transpose(l3_reshaped,perm=[0,2,1])
    
    fw_hidden_size=temp_shape[1]*temp_shape[2]//4
    bw_hidden_size=fw_hidden_size//2
    cell_fw = rnn.BasicLSTMCell(num_units=fw_hidden_size,state_is_tuple=True)
    cell_bw = rnn.BasicLSTMCell(num_units=bw_hidden_size,state_is_tuple=True)

    # l4_squeeze = tf.squeeze(l4_max_pool,squeeze_dims=1)
    outputs, states= tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,l3_reshaped,dtype=tf.float32) #[N,25,8]

    temp_shape1 = outputs[1].get_shape().as_list()
    X_for_fc = tf.reshape(outputs[1],[-1,temp_shape1[1]*temp_shape1[2]])

    # FC layer
    y_pred = tf.contrib.layers.fully_connected(X_for_fc,num_classes, activation_fn=None)
    y_pred = tf.layers.dropout(y_pred,0.1,training=phase)
    y_pred = tf.reshape(y_pred, [-1,num_classes])
    result = tf.argmax(y_pred,axis=1)
    target = tf.argmax(y,axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(result,target),"float"))

    # crossentropy loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=y))
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    last_epoch = tf.Variable(0, name='last_epoch')


    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            try:
                saver.restore(sess,ckpt.model_checkpoint_path)
                print("Successfully loaded:", ckpt.model_checkpoint_path)
            except:
                print("Error on loading old network weights")
        else:
            print("Could not find old network weights")
        start_from = sess.run(last_epoch)

        for epoch in range(start_from,iterations):
            for j in range(int(data_size/batch_size)):
                batch = train[j*batch_size:(j+1)*batch_size]
                batchx, batchy = np.hsplit(batch,[-1])
                batchx = np.reshape(batchx,[-1,256,198,1])
                batchy = sess.run(tf.one_hot(batchy[:,0,0]-1,4))                
                _, step_loss, step_accuracy = sess.run([train_op,loss,accuracy],feed_dict={x:batchx,y:batchy,phase:True})
                if j%25==0:
                    print("[step: {}/{}] loss: {} accuracy: {}".format(j,epoch, step_loss,step_accuracy))
                if j%25==0:
                    testyoh = sess.run(tf.one_hot(testy[:,0,0]-1,4))
                    test_accuracy,test_loss,test_result, test_target = sess.run([accuracy,loss,result,target],feed_dict={x:testx,y:testyoh,phase:False})
                    confusionMatrix = np.zeros((4,4),dtype=int)
                    for i in range(len(test_result)):
                        confusionMatrix[test_target[i],test_result[i]]=confusionMatrix[test_target[i],test_result[i]]+1
                    print('test accuracy',test_accuracy,'\n','test_loss',test_loss,'\n',confusionMatrix)
                if j%300==0:
                    if not os.path.exists(ckpt_path):
                        os.makedirs(ckpt_path)
                    saver.save(sess,ckpt_path+"/model",global_step=epoch)
            if epoch%2 == 0:
                np.random.shuffle(train)
            sess.run(last_epoch.assign(epoch))

print('learning_finished!')    
