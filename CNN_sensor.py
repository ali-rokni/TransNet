# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 17:40:36 2017

@author: ali
"""


from __future__ import print_function
from OPPDataset import OPPDataset
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras import utils
from keras.layers import Reshape
from keras.layers import concatenate
import keras
from SomeMetrics import *
# from read_activations import *
from keras.callbacks import ModelCheckpoint
from glob import glob
from keras.layers import LSTM
from IRHDataset import IRHDataset
from WISDMDataset import WISDMDataset
from SADDataset import SADDataset

import utils as ut
import numpy as np

from keras import backend as K
from keras.layers import merge
from keras.layers import Merge
import configparser


config = configparser.ConfigParser()
config.read('config.ini')
# max_features =150
batch_size = int(config['TransNet']['batch_size'])
embedding_dims = int(config['TransNet']['embedding_dims'])
filters = int(config['TransNet']['filters'])
kernel_size = int(config['TransNet']['kernel_size'])
hidden_dims = int(config['TransNet']['hidden_dims'])
epochs = int(config['TransNet']['epochs'])
epochs2 = int(config['TransNet']['epochs2'])
translite = bool(config['TransNet']['epochs2'])
translite = False

def make_list(X, num_channels, segment_length):
    if K.image_data_format() == 'channels_first':
        X = X.reshape(X.shape[0], num_channels, segment_length)
    else:
        X = X.reshape(X.shape[0], segment_length, num_channels)
    Xs = list()
    for j in range(num_channels):
        Xs.append(X[:, :, j].reshape(len(X), segment_length))
    return Xs


def train_deep_model(dataset, XT, YT, XS, YS, location=False):
    num_channels = dataset.num_channels * dataset.num_loc
    if location:
        num_channels = dataset.num_channels

    segment_length = dataset.segment_length

    YT = utils.to_categorical(YT, dataset.num_classes)
    YS = utils.to_categorical(YS, dataset.num_classes)

    ems = list()
    for j in range(num_channels):
        embed2 = Sequential()
        embed2.add(Embedding(dataset.max_features,
                             embedding_dims,
                             input_length=segment_length))
        ems.append(embed2)

    feature_layers = [Merge(ems, mode='concat', concat_axis=1),
                      Dropout(0.2),
                      Conv1D(filters,
                             kernel_size,
                             padding='same',
                             activation='relu',
                             strides=1),
                      MaxPooling1D(pool_size=3),
                      Conv1D(filters,
                             kernel_size,
                             padding='same',
                             activation='relu',
                             strides=1),
                      MaxPooling1D(pool_size=3),
                      Dropout(0.2)
                      # ,Flatten()
                      ]

    if not translite:
        feature_layers.append(Flatten())

    lstm_output_size = num_channels * hidden_dims
    # classification_layers = [Dense(num_channels * hidden_dims),
    #                          #  Dropout(0.2),
    #                          Activation('relu'),
    #                          # Reshape(-1,-1 ),
    #                          LSTM(lstm_output_size, return_sequences=True),
    #                          Dense(dataset.num_classes),
    #                          Activation('softmax')
    #                          ]

    model = Sequential(feature_layers)

    # model.add(Dense(num_channels * hidden_dims))
    # model.add(Activation('relu'))
    # model.layers
    # model.add(Reshape((1,num_channels * hidden_dims)))
    if translite:
        model.add(LSTM(lstm_output_size))
    else:
        model.add(Dense(num_channels * hidden_dims))
        model.add(Activation('relu'))

    model.add(Dense(dataset.num_classes))
    model.add(Activation('softmax'))
    print(model.summary())
    # model = Sequential(feature_layers + classification_layers)

    # opt = keras.optimizers.adam(lr=0.01, decay=1e-2)
    opt = keras.optimizers.adam()
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy', fmeasure, precision, recall])

    XTs = make_list(XT, num_channels, segment_length)
    XSs = make_list(XS, num_channels, segment_length)

    hist = model.fit(XTs, YT,
                     verbose=2,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_data=(XSs, YS))

    for l in feature_layers:
        l.trainable = False

    model.save_weights(dataset.name + '_weights.h5')

    return ut.get_best_results(hist.history), ut.get_hist_results(hist.history), model


def retrain(dataset, model_in, Xt, Yt, Xs, Ys, retrain_list, location = False):
    num_channels = dataset.num_channels * dataset.num_loc
    if location:
        num_channels = dataset.num_channels

    afters = []
    percentage = retrain_list[0] < 1
    Ys = utils.to_categorical(Ys, dataset.num_classes)
    smallXS = make_list(Xs, num_channels, dataset.segment_length)
    model = model_in
    for j in retrain_list:
        model.load_weights(dataset.name + '_weights.h5')

        if percentage:
            Xtr, Ytr = ut.get_up_to_percent(Xt, Yt, j)
        else:
            Xtr, Ytr = ut.getUpToNth(Xt, Yt, j)

        Ytr = utils.to_categorical(Ytr, dataset.num_classes)
        smallXT = make_list(Xtr, num_channels, dataset.segment_length)
        hist = model.fit(smallXT, Ytr,
                         batch_size=batch_size,
                         verbose=2,
                         epochs=epochs2,
                         validation_data=(smallXS, Ys))
        score2 = ut.get_best_results(hist.history)

        afters.append(score2)
        print(score2)
    model.load_weights(dataset.name + '_weights.h5')
    return afters, model


