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


def print_metrics(Y, y_pred): #print precision,recall
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

x_data = np.load('Bin100X.npy')
y_data = np.load('Bin100_y_leading_to_reach.npy')

X = x_data.T
X_copy = X.copy().astype('float64')
i=0
while i < X.shape[0]:
    factor = np.sum(X_copy[i:i+100,:])/100.0
    X_copy[i:i+100,:] =X_copy[i:i+100,:]/ factor
    i+=100
X_copy[-51:,:] = X_copy[-51:,:] /factor

X_train, X_val, y_train, y_val = train_test_split(X_copy, y_data, test_size=0.15, random_state=123)

numsamples = X.shape[0]
feature_size = X.shape[1]

lrs = [5e-5,1e-4]

for lr in lrs:
    input_layer = Input(shape=(feature_size,))
    hl = Dense(64, use_bias=True, activation='relu')(input_layer)
    hl = Dropout(rate=0.5)(hl)
    # self.model.add(Activation('relu'))
    hl = Dense(64, use_bias=True, activation='relu')(hl)
    hl = Dropout(rate=0.2)(hl)
    # hl = Dense(64, use_bias=True, activation='relu')(hl)
    hl = Dense(1, use_bias=True, activation='sigmoid')(hl)

#setup model

    # self.model.add(Dense(numActions))
    class_weights = {0:0.15, 1:1.0}
    model = Model(input=[input_layer], output=[hl])

    # Add optimizers here, initialize your variables, or alternately compile your model here.
    # lr = lr
    # optimizer = keras.optimizers.RMSprop(lr=self.lr, decay=1e-5)
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    model.summary()

    model.fit(X_train,y_train,batch_size=32,epochs=75,class_weight=class_weights,validation_data=(X_val,y_val))
    model.save_weights("nn_tss1_" + str(lr))
    print_metrics(y_train,np.round(model.predict(X_train)).squeeze())
    print_metrics(y_val,np.round(model.predict(X_val)).squeeze())
    pdb.set_trace()

