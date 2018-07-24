import numpy as np
import random
import datetime
num_features = 18
seed = 123

deep = True
from numpy import genfromtxt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def writeScores(name, scores, start=0):
    metrics =['ACC', 'F1', 'Pre', 'Rec']

    scores = np.asarray(scores)
    for i in range(len(metrics)):
        file = open(name + '_' + metrics[i] +'.csv', 'a+b')
        np.savetxt(file, scores[...,start + i], delimiter=',', fmt='%.4f')
        file.close()

def preprocess(dataset, X, deep=True):
    #return X
    from scipy.signal import savgol_filter
    A = savgol_filter(np.asarray(X).reshape(dataset.segment_length, ), 9, 2)
    A = np.where(A < 0, np.zeros(A.shape, dtype=int), A)
    if deep:
        A *= 2
        A = A.astype(int).reshape(dataset.segment_length, )
    return A


def feature_extract(sig):
    med = np.median(sig,axis=0)
    mnvalue = sig.mean(axis=0)
    maxvalue = sig.max(axis=0)
    minvalue = sig.min(axis=0)
    #amp = minvalue - mnvalue
    p2p = np.ptp(sig, axis=0)
    variance = sig.var(axis=0)
    percent = np.percentile(sig, axis=0, q=[25, 50, 75])
    #stdvalue = sig.std(axis=0)
#    rms = np.sqrt(int(sum(np.square(sig)))/ len(sig))
    a, b = sig.shape
    hists = []
    for i in range(b):
        hists.extend(np.histogram(sig[:,i], bins = np.arange(0, 99, 10))[0])
    if b > 1:
        return np.concatenate([ med, mnvalue, minvalue, maxvalue, p2p, percent.ravel(), hists, variance])
    else:
        return [mnvalue, minvalue, maxvalue, p2p, variance]


def feature_select(X, Y, k=10):
    return SelectKBest(chi2, k).fit_transform(X, Y)


def extract_feature3D(dataset, X):
    Z = np.empty(shape=(len(X), num_features * dataset.num_channels * dataset.num_loc))
    for i in range(len(X)):
        Z[i] = feature_extract(X[i, :, :].T)
    return Z


def extract_random_instances_per_sub(dataset, X, Y, n, S, subs):
    mask = (S == subs[0])
    print("sub " + str(subs[0]))
    XT, YT, XS, YS = extract_random_instance(dataset, X[mask], Y[mask], n)
    for i in range(1, len(subs)):
        mask = (S == subs[i])
        Xt, Yt, Xs, Ys = extract_random_instance(dataset, X[mask], Y[mask], n)
        XT = np.vstack((XT, Xt))
        YT = np.vstack((YT, Yt))
        XS = np.vstack((XS, Xs))
        YS = np.vstack((YS, Ys))
    return XT, YT, XS, YS
def extract_random_instance(dataset, X, Y, n):
    random.seed(123)

    # num_classes = len(np.unique(Y))
    # shape = list(X.shape)
    # a = n*num_classes
    # shape[0] = a
    # Xt = np.empty(shape=shape, dtype=object)
    # Yt = np.empty(shape=a, dtype=int)
    # shape[0] = len(Y) - a
    # Xs = np.empty(shape=shape, dtype=object)
    # Ys = np.empty(shape=shape[0], dtype=int)
    # c1 = 0
    # c2 = 0
    Xt = []
    Yt = []
    Xs = []
    Ys = []
    print('unique: ', np.unique(Y))
    for i in np.unique(Y):
        idx = np.where(Y == i)
        idx = list(idx[0])
        t1 = random.sample(idx, min(n, len(idx)))
        t2 = np.setdiff1d(idx, t1)
        for m in idx:
            if m in t1:
                if len(Xt) == 0:
                    Xt = X[np.newaxis,m,...]
                    Yt = Y[m]
                else:
                    Xt = np.vstack((Xt, X[np.newaxis,m,...]))
                    Yt = np.vstack((Yt, Y[m]))
            else:
                if len(Xs) == 0:
                    Xs = X[np.newaxis,m,...]
                    Ys = Y[m]
                else:
                    Xs = np.vstack((Xs, X[np.newaxis,m, ...]))
                    Ys = np.vstack((Ys, Y[m]))
    return Xt, Yt.ravel(), Xs, Ys.ravel()

def getUpToNth(Xt, Yt, n):
    resX = []
    resY = []
    for i in np.unique(Yt):
        mask = (Yt == i)
        tempX = Xt[mask]
        tempY = Yt[mask]
        if len(resX) == 0:
            resX = tempX[0:min(len(tempX), n)]
        else:
            resX =np.vstack((resX, tempX[0:min(len(tempX), n)]))
        resY.extend(tempY[0:min(len(tempX), n)].tolist())
    resY = np.asarray(resY)
    resY = resY.reshape(len(resY),)
    return resX, resY
#
# def getXsYRSR(dataset):
#
#     if dataset.name == 'SAD':
#         X1, X2, X3, YR, SR = dataset.getAll()
#     else:
#         XR, YR, SR = dataset.XR, dataset.YR, dataset.SR
#         #XR, YR, SR = dataset.segment(XR, YR, SR)
#         # TODO: fix this
#         X1 = XR[:, 0, :]
#         X2 = XR[:, 1, :]
#         X3 = XR[:, 2, :]
#     m = max(np.max(abs(X1)), np.max(abs(X2)), np.max(abs(X3)))
#     X1 = X1 + m
#     X2 = X2 + m
#     X3 = X3 + m
#     return X1, X2, X3, YR, SR
#

# def getXY(dataset, deepy = deep):
#     X1, X2, X3, YR, SR = getXsYRSR()
#     X = np.empty(
#         shape=(int(len(X1) / dataset.num_loc + 0.1), dataset.num_channels * dataset.num_loc, dataset.segment_length))
#     S = np.empty(shape=(len(X)))
#     Y = np.empty(shape=(len(X)))
#     i = 0
#     m = 0
#
#     while i < len(X1):
#         Y[m] = YR[i]
#         S[m] = SR[i]
#         for j in range(dataset.num_loc):
#             X[m, j, :] = preprocess(dataset, X1[i, :], deepy)
#             X[m, dataset.num_loc + j, :] = preprocess(dataset, X2[i, :], deepy)
#             X[m, 2 * dataset.num_loc + j, :] = preprocess(dataset, X3[i, :], deepy)
#             i += 1
#         m += 1
#
#     Y = Y.reshape(len(Y), )
#     Y = np.asarray(Y, dtype=int)
#     Y = Y - 1
#     # mask = (SZ == 1)
#     if not deepy:
#         X = extract_feature3D(dataset, X)
#         X = feature_select(X, Y, k=max(10, 3 * dataset.num_loc))
#     return X, Y, S


# def create_test_and_train_for_loc(dataset, train_loc, test_loc, train_subjects, test_subjects, deepy = deep):
#     X1, Y1, S1 = dataset.getXYS(deepy)
#
#     X[:, j, :] = ut.preprocess(self, X1[i, :], deepy)
#     X[m, self.num_loc + j, :] = ut.preprocess(self, X2[i, :], deepy)
#     X[m, 2 * self.num_loc + j, :] = ut.preprocess(self, X3[i, :], deepy)
#
#     mask2 = False
#     for item in train_subjects:
#         mask2 |= S == item
#     # mask &= mask2
#     XT = X[mask2]
#     YT = Y[mask2]
#     ST = S[mask2]
#     # XT = X[mask]
#     # YT = Y[mask]
#     # ST = S[mask]
#     # mask = (SZ == 1)
#     mask2 = False
#     for item in test_subjects:
#         mask2 |= S == item
#     # mask &= mask2
#     XS = X[mask2]
#     YS = Y[mask2]
#     # ZS = Z[mask]
#     SS = S[mask2]
#     # XS = X[mask]
#     # YS = Y[mask]
#     # # ZS = Z[mask]
#     # SS = S[mask]
#
#     return XT, YT, ST, XS, YS, SS


def create_test_and_train(dataset, train_subjects, test_subjects, deepy = deep):
    X, Y, S = dataset.getXYS(deepy)
    mask2 = False
    for item in train_subjects:
        mask2 |= S == item
    # mask &= mask2
    XT = X[mask2]
    YT = Y[mask2]
    ST = S[mask2]
    # XT = X[mask]
    # YT = Y[mask]
    # ST = S[mask]
    # mask = (SZ == 1)
    mask2 = False
    for item in test_subjects:
        mask2 |= S == item
    # mask &= mask2
    XS = X[mask2]
    YS = Y[mask2]
    # ZS = Z[mask]
    SS = S[mask2]
    # XS = X[mask]
    # YS = Y[mask]
    # # ZS = Z[mask]
    # SS = S[mask]

    return XT, YT, ST, XS, YS, SS



def create_test_and_train_per_loc(dataset, train_subjects, test_subjects, train_loc, test_loc, deepy = deep):
    X, Y, S = dataset.getXYS(deepy)
    mask2 = False
    for item in train_subjects:
        mask2 |= S == item
    # mask &= mask2
    XT = X[mask2]
    YT = Y[mask2]
    ST = S[mask2]
    # XT = X[mask]
    # YT = Y[mask]
    # ST = S[mask]
    # mask = (SZ == 1)
    mask2 = False
    for item in test_subjects:
        mask2 |= S == item
    # mask &= mask2
    XS = X[mask2]
    YS = Y[mask2]
    # ZS = Z[mask]
    SS = S[mask2]
    # XS = X[mask]
    # YS = Y[mask]
    # # ZS = Z[mask]
    # SS = S[mask]
    a, b, c = XT.shape
    XTL = np.empty(shape=(a, dataset.num_channels * len(train_loc), c))
    for i in range(len(train_loc)):
        XTL[:, i*3, :] = XT[:,train_loc[i],: ]
        XTL[:, i*3+1, :] = XT[:,dataset.num_loc + train_loc[i],:]
        XTL[:,i*3+2 , :] = XT[:,2 * dataset.num_loc + train_loc[i],:]

    XSL = np.empty(shape=(a, dataset.num_channels * len(test_loc), c))
    for i in range(len(test_loc)):
        XSL[:, i * 3, :] = XS[:, test_loc[i], :]
        XSL[:, i * 3 + 1, :] = XS[:, dataset.num_loc + test_loc[i], :]
        XSL[:, i * 3 + 2, :] = XS[:, 2 * dataset.num_loc + test_loc[i], :]

    return XTL, YT, ST, XSL, YS, SS

def testAccuracy(dataset, XT, YT, XS, YS, Xtr, Ytr, subs, j):
    print(subs, np.unique(YT), np.unique(YS))
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import svm
    from sklearn import neighbors
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    np.random.seed(seed)
    np.random.shuffle(XT)
    np.random.seed(seed)
    np.random.shuffle(YT)
    np.random.seed(seed)
    np.random.shuffle(Xtr)
    np.random.seed(seed)
    np.random.shuffle(Ytr)
    np.random.seed(seed * 2)
    np.random.shuffle(XS)
    np.random.seed(seed * 2)
    np.random.shuffle(YS)

    # XT = extract_feature3D(dataset, XT)
    # XS = extract_feature3D(dataset, XS)
    # Xt = extract_feature3D(dataset, Xt)
    # Xs = extract_feature3D(dataset, Xs)
    models = list()
    models.append(DecisionTreeClassifier())
    models.append(LogisticRegression())
  #
    models.append(svm.LinearSVC())
    models.append(svm.SVC())
   # models.append(neighbors.KNeighborsClassifier(3))
    models.append(QuadraticDiscriminantAnalysis())
    models.append(RandomForestClassifier())
    models.append(GradientBoostingClassifier())
    before = []
    scores2 = []
    after = []
    for model in models:
        if j == 1:
            model.fit(XT, YT)
            labels = model.predict(XS)
            before.append([accuracy_score(YS, labels), f1_score(YS, labels, average='macro'), precision_score(YS, labels, average='macro'), recall_score(YS, labels, average='macro')] )
       # model.fit(Xt, Yt)
        # labels = model.predict(Xs)
        # scores2.append(sum(labels == Ys) / len(Ys))
        #scores2.append(model.score(Xs, Ys))
        model.fit(np.concatenate((XT, Xtr)), np.concatenate((YT, Ytr)))
        labels = model.predict(XS)
        # scores3.append(sum(labels == Ys) / len(Ys))
        after.append([accuracy_score(YS, labels), f1_score(YS, labels, average='macro'), precision_score(YS, labels, average='macro'), recall_score(YS, labels, average='macro')])
   # print('time: ', datetime.datetime.now())
    print(before, after)
    return before, after
def create_transfer_set(dataset, XS, YS, max_trans, SS, subs):
    return extract_random_instances_per_sub(dataset, XS, YS, max_trans, SS, subs)

def computeResults(dataset):
    # delResults('results2')
    max_tran = 5
    num_methods = 7
    num_metrics = 4
 #   for j in range(1, 6):
    befores = np.empty(shape=(1, len(dataset.subjects),num_methods, num_metrics) )
    afters = np.empty(shape=(max_tran, len(dataset.subjects),num_methods, num_metrics) )
    c = 0
    for i in dataset.subjects:
    # for i in range(1,4):
       XT, YT, ST, XS, YS, SS = create_test_and_train(dataset, np.setdiff1d(dataset.subjects, [i]), [i], False)
       if len(np.unique(YS)) < 2:
           continue
       Xt, Yt, Xs, Ys = create_transfer_set(dataset, XS, YS, max_tran, SS, [i])
       for j in range(1,max_tran + 1):
           Xtr, Ytr = getUpToNth(Xt, Yt, j)
           before, after = testAccuracy(dataset, XT, YT, Xs, Ys, Xtr, Ytr,[i], j)
           if j == 1:
               befores[0, c,:, :] = before
           afters[j-1, c,:, :] = after
       c += 1
       # befores.append(before)
       # afters.append(after)
    for i in range(max_tran):
        writeScores('results7/' + dataset.name + '_After_' + str(i + 1) + '_', afters[i,:, :, :])
    writeScores('results7/' + dataset.name + '_Before_' + str(1) + '_', befores[0, :, :, :])


def delResults(foldername):
    import os, shutil
    folder = foldername
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def get_best_results(history):
    res = []
    keys = ['val_acc', 'val_fmeasure', 'val_precision', 'val_recall']
    for key in keys:
        res.append(max(history[key]))
    return res


def get_hist_results(history):
    res = []
    keys = ['val_acc', 'val_fmeasure', 'val_precision', 'val_recall']
    for key in keys:
        if len(res) == 0:
            res = history[key]
        else:
            res = np.vstack((res, history[key]))
    return res

def separateY_into_2(dataset, train_subjects, deepy=True):
    X, Y, S = dataset.getXYS(deepy)
    mask2 = False
    for item in train_subjects:
        mask2 |= S == item
    # mask &= mask2
    X = X[mask2]
    Y = Y[mask2]
    S = S[mask2]
    random.seed(123)
    uY = np.unique(Y)
    s1 = random.sample(list(uY), int(len(uY)/2))
    mask = np.in1d(Y, s1)
    XT = np.asarray(X[~mask])
    YT = np.asarray(Y[~mask])
    ST = np.asarray(S[~mask])
    ST = ST.reshape(len(ST),)
    YT = YT.reshape(len(YT),)
    XS = np.asarray(X[mask])
    YS = np.asarray(Y[mask])
    YS.reshape(len(YS),)
    SS = np.asarray(S[mask])
    SS = SS.reshape(len(SS,))
    return XT, YT, ST, XS, YS, SS


def testOverallAccuracy(dataset, percent):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import svm
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    XT, YT, ST, XS, YS, SS = dataset.getSplitedXY(False, percent)
    models = list()
    models.append(DecisionTreeClassifier())
    models.append(LogisticRegression())
    models.append(svm.LinearSVC())
    models.append(svm.SVC())
    models.append(QuadraticDiscriminantAnalysis())
    models.append(RandomForestClassifier())
    models.append(GradientBoostingClassifier())
    scores = []
    for model in models:
        model.fit(XT, YT)
        labels = model.predict(XS)
        scores.append([accuracy_score(YS, labels), f1_score(YS, labels, average='macro'), precision_score(YS, labels, average='macro'), recall_score(YS, labels, average='macro')] )
        # labels= model.predict(XT)
        # print(accuracy_score(YT, labels))
    print(scores)
    scores = np.asarray(scores)
    scores = scores[np.newaxis,...]
    writeScores('results7/' + dataset.name + '_Overall__', scores)
    return scores

def create_transfer_set(dataset, XS, YS, max_trans, SS, subs):
    return extract_random_instances_per_sub(dataset, XS, YS, max_trans, SS, subs)


def create_sequential_transfer_set(dataset, XS, YS, max_trans, SS, subs):
    unique_ys = np.unique(YS)
    d = dict()
    for y in unique_ys:
        d[y] = []

    for c, v in enumerate(YS):
        if len(d[v]) < max_trans:
            d[v].append(c)

    transfer_indexes = [i for x in d.values() for i in x]
    test_mask = np.ones(len(YS), dtype=bool)
    test_mask[transfer_indexes] = False

    return XS[~test_mask], YS[~test_mask], XS[test_mask], YS[test_mask]

def transfer_percent(XS, YS, percent):
    np.random.seed(seed)
    np.random.shuffle(XS)
    np.random.seed(seed)
    np.random.shuffle(YS)
    split_point = int(percent * len(YS))
    Xt = XS[:split_point]
    Yt = YS[:split_point]
    Xs = XS[split_point:len(YS)]
    Ys = YS[split_point:len(YS)]
    return Xt, Yt, Xs, Ys


def get_up_to_percent(X, Y, p):
    split_point = int(p * len(Y))
    return X[:split_point], Y[:split_point]

def divide_in_3(X, Y):
    d1 = 0.35
    d2 = 0.7
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(Y)
    split_point1 = int(d1 * len(Y))
    split_point2 = int(d2 * len(Y))
    X1 = X[:split_point1]
    Y1 = Y[:split_point1]
    X2 = X[split_point1:split_point2]
    Y2 = Y[split_point1:split_point2]
    X3 = X[split_point2:len(Y)]
    Y3 = Y[split_point2:len(Y)]

    return X1, Y1, X2, Y2, X3, Y3