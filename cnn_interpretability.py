import tensorflow as tf
from tensorflow import layers
import pdb
import numpy as np
import tensorboard
from deepexplain.tensorflow import DeepExplain
import sys
import scipy.ndimage.filters as filters

sys.path.insert(0, '../repos/DeepExplain/examples/')

from utils import plot, plt
sess = tf.Session()

def print_metrics(Y, y_pred):
  
  tp = ((Y==y_pred)*(Y==1)).sum()
  fp = ((Y!=y_pred)*(y_pred==1)).sum()
  fn = ((y_pred==0)*(Y==1)).sum()
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1_score = 2 * precision * recall / (precision + recall)
  accuracy = np.sum(y_pred == Y)/len(Y)
  #pdb.set_trace()
  print('Precision: %4f' %(precision))
  print('Recall: %4f' %(recall))
  print('f1_score: %4f' %(f1_score))
  print('accuracy: %4f' %(accuracy))

tss_b = np.load('tss1_binary.npy')
reach = np.load('reachStart1_np.npy')


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
loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y,x1,pos_weight))
tf.summary.scalar('cross_entropy', loss)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)
win_size=inp_size//2
tss_trainx = []
tss_trainy = []
tss_b = tss_b.transpose()

####  Reshaping the training and test set for easy convolutions ####

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
saver = tf.train.Saver()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./saved_models' + '/cnn-models/train_npclip',sess.graph)
test_writer = tf.summary.FileWriter('./saved_models'+'/cnn-models/test_noclip',sess.graph)
saver.restore(sess,tf.train.latest_checkpoint('./saved_models/'))#./saved_models/cnn-models/no_clip/best_model/reg-700')
grads =[]
i=0
idx = 418
for j in range(50):
    idx = int(reach[j])
    trainx = tss_trainx[idx+i*100:idx+1+i*100]
    trainy = tss_trainy[idx+i*100:idx+1+i*100]
    if not trainy:
        trainy=-1
    ############ We use the DeepExplain repository (https://github.com/marcoancona/DeepExplain) to interpret the CNN model predictions##########
    with DeepExplain(session=sess) as de:
        # We run `explain()` several time to compare different attribution methods
        attributions = {
            # Gradient-based
            'Saliency maps':        de.explain('saliency', x1 * trainy, x, trainx),
            #'Gradient * Input':     de.explain('grad*input', x1 * trainy, x, trainx),
            'Integrated Gradients': de.explain('intgrad', x1 * trainy, x, trainx),
            #'Epsilon-LRP':          de.explain('elrp', logits * yi, X, xi),
            #'DeepLIFT (Rescale)':   de.explain('deeplift', logits * yi, X, xi),
            #Perturbation-based
            #'_Occlusion [1x1]':      de.explain('occlusion', logits * yi, X, xi),
            #'_Occlusion [3x3]':      de.explain('occlusion', logits * yi, X, xi, window_shape=(3,))
        }
        print ('Done', i)
    grads.append(attributions['Integrated Gradients'][0])
grads = np.stack(grads)
grads_sum = np.mean(grads,axis=0)
im = filters.gaussian_filter(grads_sum, 4)
pdb.set_trace()
plt.imshow(im)
plt.show()

