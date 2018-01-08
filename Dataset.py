import numpy as np
import utils as ut
import random
class Dataset:
    def __init__(self, name, num_classes, num_channels, segment_length, num_loc, subjects):
        self.name = name
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.segment_length = segment_length
        self.num_loc = num_loc
        self.subjects = subjects
        self.YR = []
        self.SR = []
        self.XR = []
        self.X = []
        self.Y = []
        self.S = []


    def computeXYS(self, deepy):
        X1 = self.XR[:, 0, :]
        X2 = self.XR[:, 1, :]
        X3 = self.XR[:, 2, :]
        m = max(np.max(abs(X1)), np.max(abs(X2)), np.max(abs(X3)))
        self.max_features = int(m * 5)
        X1 = X1 + m
        X2 = X2 + m
        X3 = X3 + m
        X = np.empty(
            shape=(int(len(X1) / self.num_loc + 0.1), self.num_channels * self.num_loc, self.segment_length))
        S = np.empty(shape=(len(X)))
        Y = np.empty(shape=(len(X)))
        i = 0
        m = 0

        while i < len(X1):
            Y[m] = self.YR[i]
            S[m] = self.SR[i]
            for j in range(self.num_loc):
                X[m, j, :] = ut.preprocess(self, X1[i, :], deepy)
                X[m, self.num_loc + j, :] = ut.preprocess(self, X2[i, :], deepy)
                X[m, 2 * self.num_loc + j, :] = ut.preprocess(self, X3[i, :], deepy)
                i += 1
            m += 1

        Y = Y.reshape(len(Y), )
        Y = np.asarray(Y, dtype=int)
        Y = Y - 1
        if not deepy:
            X = ut.extract_feature3D(self, X)
            X = ut.feature_select(X, Y, k=max(10, 3 * self.num_loc))
        return X, Y, S

    def getXYS(self, deepy):
        if len(self.X) == 0:
            self.X, self.Y, self.S = self.computeXYS(deepy)
        return self.X, self.Y, self.S

    def getSplitedXY(self, deepy, percent):
        X, Y, S = self.getXYS(deepy)
        np.random.seed(123)
        np.random.shuffle(X)
        np.random.seed(123)
        np.random.shuffle(Y)
        np.random.seed(123)
        np.random.shuffle(S)
        a = len(X)
        b = int(a*percent)
        XS = X[0:b]
        YS = Y[0:b]
        SS = S[0:b]
        XT = X[b+1:a]
        YT = Y[b+1:a]
        ST = S[b+1:a]
        return XT, YT, ST, XS, YS, SS





