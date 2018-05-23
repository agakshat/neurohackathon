import tensorflow as tf
from tensorflow import layers
import pdb
import numpy as np
import tensorboard
sess = tf.Session()

def print_metrics(Y, y_pred):
  tp = ((Y==y_pred)*(Y==1)).sum()
  fp = ((Y!=y_pred)*(y_pred==1)).sum()
  fn = ((y_pred==0)*(Y==1)).sum()
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1_score = 2 * precision * recall / (precision + recall)
  accuracy = np.sum(y_pred == Y)/len(Y)
  print('Precision: %4f' %(precision))
  print('Recall: %4f' %(recall))
  print('f1_score: %4f' %(f1_score))
  print('accuracy: %4f' %(accuracy))

tss_b = np.load('tss1_binary.npy')
reach = np.load('reachStart1_np.npy')
amp = np.load('amp_unbinned.npy')

stop_time = 100000
test_tss_b = tss_b[:,2*stop_time:3*stop_time]
tss_b = tss_b[:,:2*stop_time]
tss_mask = np.zeros(tss_b[:1,:].shape)
pos_weight = 15
lr = 0.001
batch_size = 100
inp_size = 658
test_set=[]
def normal(idx, std):
    val = np.exp(-np.square(idx-1000)/(2*std*std))/np.sqrt(2*np.pi*std*std)
    return val
for r in reach:
    if r>stop_time*2:
        if r>2*stop_time+inp_size and r<3*stop_time-inp_size:
            test_set.append(test_tss_b[:,int(r-2*stop_time-inp_size//2):int(r-2*stop_time+inp_size//2)].transpose())
        continue
    for i in np.arange(max(0,r-1000),min(stop_time*2, r+1000)):
        tss_mask[0,int(i)]=1#normal(i,std=400)
        test_set = np.stack(test_set)

###############  Defining the Convolutional Neural network #######################

tss_mask = tss_mask/np.max(tss_mask)
x = tf.placeholder(tf.float32,[None,inp_size, 255])
y = tf.placeholder(tf.float32,[None,1,1])
relu = tf.nn.relu
x1 = relu(layers.conv1d(x,250,[10],padding='valid'))
x1 = layers.max_pooling1d(x1,100,strides=1,padding='valid')
x1 = relu(layers.conv1d(x1,250,[10],padding='valid'))
x1 = layers.max_pooling1d(x1,100,strides=1,padding='valid')
x1 = relu(layers.conv1d(x1,250,[10],padding='valid'))
x1 = layers.max_pooling1d(x1,100,strides=1,padding='valid')
x1 = relu(layers.conv1d(x1,250,[10],padding='valid'))
x1 = layers.max_pooling1d(x1,100,strides=1,padding='valid')
x1 = relu(layers.conv1d(x1,250,[10],padding='valid'))
x1 = layers.max_pooling1d(x1,100,strides=1,padding='valid')
x1 = relu(layers.conv1d(x1,250,[10],padding='valid'))
x1 = layers.max_pooling1d(x1,100,strides=1,padding='valid')
x1 = layers.conv1d(x1,1,[10],padding='valid')
x1_prob = tf.nn.sigmoid(x1)

#### Defining the loss and optimizers ####################
loss = tf.reduce_mean(tf.square(y-x1)*(y*pos_weight+1))
tf.summary.scalar('cross_entropy', loss)
pdb.set_trace()
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

####  Reshaping the training and test set for easy convolutions ####

win_size=inp_size//2
tss_trainx = []
tss_trainy = []
tss_b = tss_b.transpose()
for i, tss in enumerate(tss_b):
    mini = max(0,i-win_size)
    maxi = min(stop_time, i+win_size)
    if mini==0 or maxi==stop_time:
        continue
    tss_trainx.append(tss_b[mini:maxi])
    tss_trainy.append(tss_mask[0,i])
tss_trainx = np.stack(tss_trainx)
tss_trainy = np.stack(tss_trainy)
tss_testx = []
tss_testy = []
for i, tss in enumerate(tss_b[stop_time:2*stop_time]):
    mini = max(0,i-win_size)
    maxi = min(stop_time, i+win_size)
    if mini==0 or maxi==stop_time:
        continue
    tss_testx.append(tss_b[stop_time+mini:stop_time+maxi])
    tss_testy.append(tss_mask[0,stop_time+i])

tss_testx = np.stack(tss_testx)
tss_testy = np.stack(tss_testy)

perm = np.random.permutation(tss_trainx.shape[0])
tss_trainx = tss_trainx[perm]
tss_trainy = tss_trainy[perm]
prob1 = []
saver = tf.train.Saver(max_to_keep=50)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./saved_models' + '/cnn-models/train_npclip',sess.graph)
test_writer = tf.summary.FileWriter('./saved_models'+'/cnn-models/test_noclip',sess.graph)

######### Training and Testing the model ####################################

for j in range(10000):
    i=j%(tss_trainx.shape[0]//batch_size)
    trainx = tss_trainx[i*batch_size:(i+1)*batch_size]
    trainy = tss_trainy[i*batch_size:(i+1)*batch_size]
    if trainx.shape[0]<batch_size:
        continue
    summary,loss_np,_ = sess.run([merged, loss, train_op], feed_dict={x:trainx, y:np.expand_dims(np.expand_dims(trainy,1),2)})
    train_writer.add_summary(summary, i)
    if j%100==0:
        print('train',loss_np)
        testx = tss_testx[i*batch_size:(i+1)*batch_size]
        testy = tss_testy[i*batch_size:(i+1)*batch_size]
        summary,loss_np= sess.run([merged, loss], feed_dict={x:testx, y:np.expand_dims(np.expand_dims(testy,1),2)})
        print('iter',i,'test', loss_np)
        test_writer.add_summary(summary, i)
        saver.save(sess,'./saved_models/cnn-models/no_clip/reg',global_step=i)
        true_probs=[]
        prob1=[]
        for k in range(1000):
            testx = tss_testx[k*batch_size:(k+1)*batch_size]
            testy = tss_testy[k*batch_size:(k+1)*batch_size]
            if testx.shape[0]<batch_size:
                continue
            prob = sess.run([x1_prob], feed_dict={x:testx, y:np.expand_dims(np.expand_dims(testy,1),2)})
            prob1.append(prob[0])  
            true_probs.append(testy)
        true_probs = np.stack(true_probs)
        prob1 = np.stack(prob1)
        pred = prob1>0.5
        print_metrics(true_probs.reshape(-1), pred.reshape(-1))

pdb.set_trace()
