# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:21:40 2017

@author: ali
"""
import numpy as np
from numpy import genfromtxt
import utils as ut
import random
# deep = True
# num_classes = 19
# num_features = 5
# segment_length = 125
# num_channels = 3
# num_loc = 5
# def feature_extract(sig):
#     # med = sig.median(axis=0)
#     mnvalue = sig.mean(axis=0)
#     maxvalue = sig.max(axis=0)
#     minvalue = sig.min(axis=0)
#     #amp = minvalue - mnvalue
#     p2p = maxvalue - minvalue
#     variance = sig.var(axis=0)
#     #stdvalue = sig.std(axis=0)
#     rms = np.sqrt(sum(np.square(sig))/ len(sig))
#     # s2e = sig(x,:) - sig(1,:)
#     a, b = sig.shape
#     if b > 1:
#         return np.concatenate([ mnvalue, minvalue, maxvalue, p2p, variance])
#     else:
#         return [mnvalue, minvalue, maxvalue, p2p, variance];

# def extract_random_instances_per_sub(X, Y, n, S, subs):
#     mask = (S == subs[0])
#     XT, YT, XS, YS = extract_random_instance(X[mask], Y[mask], n)
#     for i in range(1, len(subs)):
#         mask = (S == subs[i])
#         Xt, Yt, Xs, Ys = extract_random_instance(X[mask], Y[mask], n)
#         XT = np.vstack((XT, Xt))
#         YT = np.vstack((YT, Yt))
#         XS = np.vstack((XS, Xs))
#         YS = np.vstack((YS, Ys))
#     return XT, YT, XS, YS
# def extract_random_instance( X, Y, n):
#     random.seed(123)
#     a, b, c = X.shape
#     Xt = np.empty(shape=(n*num_classes,b, c), dtype=object)
#     Yt = np.empty(shape=(n*num_classes), dtype=int)
#     Xs = np.empty(shape=(len(Y)-(n*num_classes),b, c), dtype=object)
#     Ys = np.empty(shape=(len(Y)-(n*num_classes)), dtype=int)
#     c1 = 0
#     c2 = 0
#     for i in range(num_classes):
#         idx = np.where(Y == i)
#         idx = list(idx[0])
#         t1 = random.sample(idx, n)
#         t2 = np.setdiff1d(idx, t1)
#         for m in idx:
#             if m in t1:
#                 Xt[c1] = X[m,:,:]
#                 Yt[c1] = Y[m]
#                 c1 += 1
#             else:
#                 Xs[c2] = X[m,:,:]
#                 Ys[c2] = Y[m]
#                 c2 += 1
#         # Xt[s:e] = X[t2]
#         # Yt[s:e] = Y[t2]
#     return Xt, Yt, Xs, Ys

# def extract_feature3D(X):
#     Z = np.empty(shape=(len(X), num_features * num_channels * num_loc))
#     for i in range(len(X)):
#         Z[i] = feature_extract(X[i, :, :].T)
#     return Z

# def read_simple_feature3D():
#     X, Y = read_simple3D()
#     Z = ut.extract_feature3D(X)
#     return Z, Y
# def read_simple_features():
#     X, Y = read_simple()
#     Z = np.empty(shape=(len(X),num_features))
#     for i in range(len(X)):
#         Z[i] = feature_extract(X[i,:])
#     return Z, Y



# def read_simple3D():
#     from scipy.signal import savgol_filter
#     X1 = genfromtxt('X1.csv', delimiter=',')
#     X2 = genfromtxt('X2.csv', delimiter=',')
#     X3 = genfromtxt('X3.csv', delimiter=',')
#     m = max(np.max(abs(X1)), np.max(abs(X2)), np.max(abs(X3)))
#     X1 = X1 + m
#     X2 = X2 + m
#     X3 = X3 + m
#     X = np.empty(shape=(len(X1), num_channels, segment_length))
#     for i in range(len(X1)):
#         B = np.empty(shape=(num_channels, segment_length))
#         B[0, :] = preprocess(X1[i, :], deep)
#         B[1, :] = preprocess(X2[i, :], deep)
#         B[2, :] = preprocess(X3[i, :], deep)
#         X[i,:] = B
#     Y = genfromtxt('Y.csv', delimiter=',')
#     Y = Y - 1
#     return X, Y

# def preprocess(X, deep=True):
#     from scipy.signal import savgol_filter
#     A = savgol_filter(np.asarray(X).reshape(segment_length, ), 9, 2)
#     if deep:
#         A *= 2
#         A = A.astype(int).reshape(segment_length, )
#     return A
# def read_simple():
#     X = genfromtxt('X1.csv', delimiter=',')
#     X = X + np.max(np.abs(X))
#     for i in range(len(X)):
#         X[i,:] = preprocess(X[i,:])
#     Y = genfromtxt('Y.csv', delimiter=',')
#     Y = Y - 1
#     return X, Y

def write_to_file(dataset, mvts, subjects, motes, segments):
    n = len(mvts) * len(subjects) * len(motes) * len(segments);
    # Xs = np.empty(shape=(3,n), dtype=object)
    # X = np.empty(shape=(n,segment_length), dtype=float)
    X1 = np.empty(shape=(n, dataset.segment_length), dtype=float)
    X2 = np.empty(shape=(n, dataset.segment_length), dtype=float)
    X3 = np.empty(shape=(n, dataset.segment_length), dtype=float)
    Y = np.empty(shape=(n,1), dtype=int)
    Z = np.empty(shape=(n,1), dtype=int)
    S = np.empty(shape=(n,1), dtype=int)
    i = 0
    for sub in subjects:
        for mvt in mvts:
            for seg in segments:
                for mote in motes:
                    filename = 'C:\D\WSU\Research\Experiments\SAD\\a' + str(mvt).zfill(2) + '\\p' + str(
                        sub) + '\\s' + str(seg).zfill(2) + '.txt'
                    sig = genfromtxt(filename, delimiter=',')
                    start = (mote - 1) * 9 + 1
                    #c = 3
                    A = sig[:, start:start + 1]
                    X1[i,:] = A.reshape(dataset.segment_length,)
                    A = sig[:, start + 1:start + 2]
                    X2[i, :] = A.reshape(dataset.segment_length, )
                    A = sig[:, start + 2:start + 3]
                    X3[i, :] = A.reshape(dataset.segment_length, )
                    Y[i] = mvt
                    Z[i] = mote
                    S[i] = sub
                    i = i + 1
    np.savetxt('SAD_allX1.csv', X1, delimiter=',')
    np.savetxt('SAD_allX2.csv', X2, delimiter=',')
    np.savetxt('SAD_allX3.csv', X3, delimiter=',')
    np.savetxt('SAD_allY.csv', Y, delimiter=',')
    np.savetxt('SAD_allZ.csv', Z, delimiter=',')
    np.savetxt('SAD_allS.csv', S, delimiter=',')
    #np.savetxt('X3.csv', X, delimiter=',')
    #np.savetxt('Y.csv', Y, delimiter=',')
    #np.savetxt('Z.csv', Z, delimiter=',')


def read_joined_channels(dataset,mvts, subjects, motes, segments):
    n = len(mvts) * len(subjects)  * len(segments)
    X = np.empty(shape=(n,dataset.num_channels * len(motes), dataset.segment_length), dtype=float)
    Y = np.empty(shape=(n,), dtype=float)
    i = 0
    for sub in subjects:
        print('sub: ', sub)
        for mvt in mvts:
            print('mvt: ', mvt)
            for seg in segments:
                j = 0
                for mote in motes:
                    filename = 'C:\D\WSU\Research\Experiments\SAD\\a' + str(mvt).zfill(2) + '\\p' + str(
                        sub) + '\\s' + str(seg).zfill(2) + '.txt'
                    sig = genfromtxt(filename, delimiter=',')
                    start = (mote - 1) * 9
                    for c in range(dataset.num_channels):
                        A = sig[:, start+c:start + 1 + c]
                        A = A.astype(int).reshape(dataset.segment_length,)
                        A = ut.preprocess(A, deep)
                        X[i, j, :] = A
                        j += 1
        Y[i] = mvt
        i += 1
    return (X, Y)

def create_joined_test_and_train(train_mvts, train_subs, train_motes, test_mvts, test_subs, test_motes):
    XT, YT = read_joined_channels(train_mvts, train_subs, train_motes, range(1,61))
    XS, YS = read_joined_channels(test_mvts, test_subs, test_motes, range(1,61))
    return XT, YT, XS, YS

def read_data(dataset,mvts, subjects, motes, segments):
    n = len(mvts) * len(subjects) * len(motes) * len(segments)
    #X = np.empty(shape=(n,3, segment_length), dtype=object)
    X = np.empty(shape=(n,3, dataset.segment_length), dtype=float)
    Y = np.empty(shape=(n,), dtype=int)
    Z = np.empty(shape=(n,), dtype=int)
    L = np.empty(shape=(n,), dtype=int)
    i = 0
    for sub in subjects:
        print('sub: ' , sub)
        for mvt in mvts:
            print('mvt: ', mvt)
            for seg in segments:
                for mote in motes:
                    filename = 'C:\D\WSU\Research\Experiments\SAD\\a' + str(mvt).zfill(2) + '\\p' + str(
                        sub) + '\\s' + str(seg).zfill(2) + '.txt'
                    sig = genfromtxt(filename, delimiter=',')
                    start = (mote - 1) * 9
                    # X = np.vstack((X, feature_extract(sig[:, start:start+3])))
                    B = np.empty(shape= (3,dataset.segment_length))
                    for c in range(3):
                        A = sig[:, start+c:start + 1 + c]
                        A = ut.preprocess(A, False)
                        X[i, c,:] = A
                    #X[i] = B
                    # X[i,:] = B
                    Y[i] = mvt
                    Z[i] = mote
                    # L[i].append(X[i])
                    # L[i].append(Y[i])
                    # L[i].append(Z[i])
                    i = i + 1
    # np.save('all.txt', L)
    return X, Y, Z


if __name__ == "__main__":
    import utils as ut
    deep = False
    seed = 123
    subjects = range(1, 9)

    from Dataset import Dataset

    dataset = Dataset(name='SAD', num_classes=19, segment_length=125, num_channels=3, num_loc=5, subjects=subjects)


    import datetime
    print('before: ', datetime.datetime.now())
    for j in range(1, 6):
        befores = []
        afters = []
        for i in dataset.subjects:
           XT, YT, ST, XS, YS, SS = ut.create_test_and_train(dataset, np.setdiff1d(dataset.subjects, [i]), [i], False)
           before, after = ut.testAccuracy(dataset, XT, YT, XS, YS, SS, [i])
           befores.append(before)
           afters.append(after)
        ut.writeScores('results/'+ dataset.name + '_Before_' + str(j) + '_', befores)
        ut.writeScores('results/'+ dataset.name + '_After_' + str(j) + '_' , afters)
