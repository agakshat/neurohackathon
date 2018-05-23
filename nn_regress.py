#!/usr/bin/env python
import pdb
import keras
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Activation, Input,Lambda,Dropout
from keras.models import model_from_json
import keras.backend as K

import tensorflow as tf
import numpy as np
import sys, copy, argparse
import os.path
from collections import deque
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


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


maxtimesteps = 1000000
def compute_amplitude(jx, jy):
    # return jy
    return np.sqrt( jx **2 + jy** 2)


def bin_data(amp, neuron_data):  ##bin raw binary data into bins
    bin_size = 1000
    nbins = maxtimesteps // bin_size
    amp_splits = np.split(amp, nbins)
    neuron_splits = np.split(neuron_data, nbins, axis=1)

    am = [x.sum() / bin_size for x in amp_splits]
    nm = [x.sum(axis=1) for x in neuron_splits]
    # nm = [x.max(axis=1) for x  in neuron_splits]
    amp_final = np.array(am)
    neu_final = np.vstack(nm)
    # pdb.set_trace()
    return amp_final, neu_final

def bin_tss(neuron_data):
    bin_size = 1000
    nbins = maxtimesteps // bin_size
    # amp_splits = np.split(amp, nbins)
    neuron_splits = np.split(neuron_data, nbins, axis=1)

    # am = [x.sum() / bin_size for x in amp_splits]
    nm = [x.sum(axis=1) for x in neuron_splits]
    # nm = [x.max(axis=1) for x  in neuron_splits]
    # amp_final = np.array(am)
    neu_final = np.vstack(nm)
    # pdb.set_trace()
    return neu_final


def plot_test_and_true(Y_test, y_pred_test):
    plt.figure(10)
    plt.plot(Y_test, 'r')
    plt.plot(y_pred_test, 'b')
    plt.show()


data_folder = './'
jx_filename = data_folder + 'jx_np.npy'
jy_filename = data_folder + 'jy_np.npy'
neuron_filename = data_folder + 'tss1_binary.npy'
jx = np.load(jx_filename)
jy = np.load(jy_filename)
neuron_data = np.load(neuron_filename)
# amp = compute_amplitude(jx, jy)
# amp = amp[:maxtimesteps]
# each row represents a neuron
neuron_data = neuron_data[:, :maxtimesteps] * 1.0
# amp_final, neu_final = bin_data(amp, neuron_data)
neu = bin_tss(neuron_data=neuron_data)
X = neu.T

num_vecs = 50
n_cat = X.shape[1]/num_vecs
n_cat =20


X_train = X.reshape(-1,num_vecs)

depths = np.load('depth1_np.npy')
y_train = np.repeat(depths,n_cat)

y_train[y_train<1.5] =1
y_train[y_train>=1.5] =0
# y_train -=2
# pdb.set_trace()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=123)

# numsamples = X.shape[0]
# feature_size = X.shape[1]

lrs = [5e-5,1e-4]
##try out various learning rates

for lr in lrs:
    input_layer = Input(shape=(num_vecs,))
    hl = Dense(64, use_bias=True, activation='relu')(input_layer)
    hl = Dropout(rate=0.5)(hl)
    # self.model.add(Activation('relu'))
    hl = Dense(64, use_bias=True, activation='relu')(hl)
    hl = Dropout(rate=0.2)(hl)
    # hl = Dense(64, use_bias=True, activation='relu')(hl)
    hl = Dense(1, use_bias=True, activation='softmax')(hl)
    ##setup model


    model = Model(input=[input_layer], output=[hl])

    # Add optimizers here, initialize your variables, or alternately compile your model here.
    # lr = lr
    # optimizer = keras.optimizers.RMSprop(lr=self.lr, decay=1e-5)
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    model.summary()

    model.fit(X_train,y_train,batch_size=32,epochs=75,validation_data=(X_val,y_val))
    model.save_weights("nn_tss1_classify_depth" + str(lr))
    # print_metrics(y_train,np.round(model.predict(X_train)).squeeze())
    # print_metrics(y_val,np.round(model.predict(X_val)).squeeze())
    pdb.set_trace()
