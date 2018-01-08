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
from keras.layers.merge import concatenate
import keras
from SomeMetrics import *
# set parameters:
#SAD_max =
#WISDM_max
#
# max_features =150
batch_size = 64
embedding_dims = 50
filters = 20
kernel_size = 3
hidden_dims = 10
epochs = 10
epochs2 = 60


import utils as ut
import numpy as np

from keras import backend as K
from keras.layers import merge
from keras.layers import Merge




def tranfer_learning_method(dataset, XT, YT, XS, YS, Xt, Yt, max_tran, overall=False):
    num_channels = dataset.num_channels * dataset.num_loc
    segment_length = dataset.segment_length


    if K.image_data_format() == 'channels_first':
        XT = XT.reshape(XT.shape[0], num_channels, segment_length)
        XS = XS.reshape(XS.shape[0], num_channels, segment_length)
        Xt = Xt.reshape(Xt.shape[0], num_channels, segment_length)
        input_shape = (num_channels, segment_length)
    else:
        XT = XT.reshape(XT.shape[0], segment_length, num_channels)
        XS = XS.reshape(XS.shape[0], segment_length, num_channels)
        Xt = Xt.reshape(Xt.shape[0], segment_length, num_channels)
        input_shape = (segment_length, num_channels)


    # c = len(Y)
    seed = 123
    np.random.seed(seed)
    np.random.shuffle(XT)
    np.random.seed(seed)
    np.random.shuffle(YT)

    #Xt, Yt, Xs, Ys = ut.extract_random_instances_per_sub(dataset, XS, YS, numTran, SS, test_subjects)

    #REDO
    # XT = XT/max_features
    # XS = XS/max_features
    # Xt = Xt/max_features
    print('Loading data...')
    #(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    #print(len(x_train), 'train sequences')
    #print(len(x_test), 'test sequences')
    #
    #print('Pad sequences (samples x time)')
    #REDO
    # XT = sequence.pad_sequences(XT, maxlen=maxlen)
    # XS = sequence.pad_sequences(XS, maxlen=maxlen)
    YT = utils.to_categorical(YT, dataset.num_classes)
    YS = utils.to_categorical(YS, dataset.num_classes)


    #REDO
    # Xt = sequence.pad_sequences(Xt, maxlen=maxlen)
    # Xs = sequence.pad_sequences(Xs, maxlen=maxlen)

    # Ys = utils.to_categorical(Ys, dataset.num_classes)

    print('....0')

    ems = list()

    for j in range(num_channels):
        embed2 = Sequential()
        embed2.add(Embedding(dataset.max_features,
                        embedding_dims,
                        input_length=segment_length))
        ems.append(embed2)
    #REDO
    feature_layers = [Merge(ems, mode='concat', concat_axis = 1),
                      Dropout(0.2),
                      # Conv1D(filters,
                      #        kernel_size,
                      #        padding='same',
                      #        activation='relu',
                      #        strides=1),
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
                      Dropout(0.2),
                     Flatten()
                     ]


    # feature_layers = [Conv1D(filters,
    #                         kernel_size,
    #                         padding='same',
    #                         activation='relu',
    #                         input_shape=input_shape,
    #                         ),
    #                  MaxPooling1D(pool_size=3),
    #
    #                  Conv1D(filters,
    #                         kernel_size,
    #                         padding='same',
    #                         activation='relu',
    #                         strides=1),
    #                  MaxPooling1D(pool_size=3),
    #                   Dropout(0.2),
    #                   # Conv1D(filters,
    #                   #        kernel_size,
    #                   #        padding='same',
    #                   #        activation='relu',
    #                   #        input_shape=input_shape,
    #                   #        ),
    #                   # MaxPooling1D(pool_size=3),
    #                   # Conv1D(filters,
    #                   #        kernel_size,
    #                   #        padding='same',
    #                   #        activation='relu',
    #                   #        strides=1),
    #                   # MaxPooling1D(pool_size=3),
    #                   # Dropout(0.2),
    #                  Flatten()
    #                  ]

    classification_layers = [Dense(num_channels * hidden_dims),
                            Dropout(0.2),
                            Activation('relu'),
                            Dense(dataset.num_classes),
                            Activation('softmax')
                            ]


    model = Sequential(feature_layers + classification_layers)
    # opt = keras.optimizers.adam(lr=0.01, decay=1e-2)
    opt = keras.optimizers.adam()
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy', fmeasure, precision, recall])

    validationSet = XT[:,:,0].reshape(len(XT), segment_length)

    # for i in range(1, num_channels+1):
    #     np.concatenate(validationSet
    #REDO
    XTs = list()
    XSs = list()
    for j in range(num_channels):
        XTs.append(XT[:, :, j].reshape(len(XT), segment_length))
        XSs.append(XS[:, :, j].reshape(len(XS), segment_length))

    hist = model.fit(XTs, YT,
              batch_size=batch_size,
              epochs=epochs,
    validation_data=(XSs, YS))

    # score1 = ut.get_best_results(hist.history)
    score1 = model.evaluate(XSs, YS, verbose=0)

    if overall:
        return ut.get_best_results(hist.history), ut.get_hist_results(hist.history)


    for l in feature_layers:
        l.trainable = False

    from keras.models import model_from_json

    json_string = model.to_json()
    model.save_weights('my_model_weights.h5')

    # import copy
    # fixed_model = copy.deepcopy(model)
    smallXT = list()
    remainingX = list()
    # befores[j - 1, i - 1, :, :] = before
    # afters[j - 1, i - 1, :, :] = before
    afters = []

    for j in range(1, max_tran + 1):
    # for j in range(3, 4):
        Xtr, Ytr = ut.getUpToNth(Xt, Yt, j)
        Ytr = utils.to_categorical(Ytr, dataset.num_classes)
        score2 = doRetrain(model, num_channels, segment_length, Xtr, Ytr, XSs, YS)
        afters.append(score2)
        print(score2)
        # model = model_from_json(json_string)
        model.load_weights('my_model_weights.h5')
#     for j in range(num_channels):
#         smallXT.append(Xt[:,:,j].reshape(len(Xt), segment_length))
#         remainingX.append(Xs[:, :, j].reshape(len(Xs), segment_length))
#
#     # model.fit(XT, YT,
#     #           batch_size=batch_size,
#     #           epochs=epochs,
#     #           validation_data=(XS, YS))
#     #
#     # score1 = model.evaluate(XS, YS, verbose=0)
#
#     # freeze feature layers and rebuild model
#
# #REDO
#     model.fit(smallXT, Yt,
#               batch_size=batch_size,
#               epochs=epochs2,
#     validation_data=(remainingX, Ys))
#     AA = model.predict(remainingX, batch_size)
#
#     score2 = model.evaluate(remainingX, Ys, verbose=0)
#     # model.fit(Xt, Yt,
#     #           batch_size=batch_size,
#     #           epochs=epochs2,
#     # validation_data=(Xs, Ys))
#     #AA = model.predict(Xs, batch_size)
#
#     # score2 = model.evaluate(Xs, Ys, verbose=0)
#     print(score2[1])
    return score1, afters

# subjects = range(1,6)
# mvts = range(1,20)
# segments = range(1,61)
# motes = range(1,2)
print('here')
#X, Y = read_simple3D()
#X = X.astype(np.float64, copy=False)
#Y = Y.astype(np.int, copy=False)
#test_subjects = np.random.randint(1, 9, 1)
scores=[]

#subjects = np.setdiff1d(range(1, 37), 19)
#dataset = Dataset(name='WISDM', num_classes=6, segment_length=200, num_channels=3, num_loc=1, subjects=subjects)
from IRHDataset import IRHDataset
from WISDMDataset import WISDMDataset
from SADDataset import SADDataset
#datasets = [IRHDataset(), OPPDataset(), WISDMDataset(), SADDataset()]


# for j in range(1, 6):
#     for dataset in datasets:
#         befores = []
#         afters = []
#         for i in dataset.subjects:
#             XT, YT, ST, XS, YS, SS = ut.create_test_and_train(dataset, np.setdiff1d(dataset.subjects,[i]),[i])
#             before, after = tranfer_learning_method(dataset, XT, YT, XS, YS, SS, [i], j)
#             befores.append(before)
#             afters.append(after)
#         ut.writeScores('results/Deep_'+ dataset.name + '_Before_' + str(j) + '_' , befores)
#         ut.writeScores('results/Deep_'+ dataset.name + '_After_' + str(j) + '_', afters)


def doRetrain(model,num_channels, segment_length, Xt, Yt, XSs, YS):
    smallXT = list()

    for j in range(num_channels):
        smallXT.append(Xt[:, :, j].reshape(len(Xt), segment_length))

    # seed = 123
    # np.random.seed(seed)
    # np.random.shuffle(smallXT)
    # np.random.seed(seed)
    # np.random.shuffle(Yt)

    hist = model.fit(smallXT, Yt,
              batch_size=batch_size,
              epochs=epochs2,
              validation_data=(XSs, YS))
    # AA = model.predict(remainingX, batch_size)
    score2 = ut.get_best_results(hist.history)
    #score2 = model.evaluate(remainingX, Ys, verbose=0)
    return score2


def deepTester():
    max_tran = 5
    num_methods = 1
    num_metrics = 4
     #   for j in range(1, 6):
    # dataset = WISDMDataset()
    dataset= SADDataset()
    befores = np.empty(shape=(1, len(dataset.subjects), num_metrics))
    afters = np.empty(shape=(max_tran, len(dataset.subjects), num_metrics))
    c = 0
    for i in dataset.subjects:
    # for i in range(4,5):
    #     XT, YT, ST, XS, YS, SS = ut.create_test_and_train(dataset, np.setdiff1d(dataset.subjects, [i]), [i], True)
        XT, YT, ST, XS, YS, SS = ut.separateY_into_2(dataset, [i], True)
        seed = 123
        np.random.seed(seed * 2)
        np.random.shuffle(XS)
        np.random.seed(seed * 2)
        np.random.shuffle(YS)
        if len(np.unique(YS)) < 2:
            continue
        # Xt, Yt, Xs, Ys = ut.create_transfer_set(dataset, XS, YS, max_tran, SS, [i])
        Xt, Yt, Xs, Ys = ut.create_transfer_set(dataset, XS, YS, max_tran, SS, [i])
        score1, score2 = tranfer_learning_method(dataset, XT, YT, Xs, Ys, Xt, Yt, max_tran)
        befores[0, c, :] = score1[1:]
        afters[:, c, :] = score2
        with open('log.csv', 'a+') as file:
            file.write('%03d\n' % c)
        c += 1

       # befores.append(before)
       # afters.append(after)

    for i in range(max_tran):
        #ut.writeScores('results2/' + dataset.name + '_Before_' + str(i + 1) + '_', befores[i, :, :, :])
        ut.writeScores('results6/Deep_' + dataset.name + '_After_' + str(i + 1) + '_', afters[i,...])
    ut.writeScores('results6/Deep_' + dataset.name + '_Before_', befores[0,...])


def testOverall():
    dataset = OPPDataset()
    XT, YT, ST, XS, YS, SS = dataset.getSplitedXY(True, 0.2)
    score1, history = tranfer_learning_method(dataset, XT, YT, XS, YS, XS, YS, 0, True)
    print(score1)
    print(history)
    ut.writeScores('results3/Deep_' + dataset.name + '_Overall_', score1[np.newaxis, ...])


# testOverall()
deepTester()