from Dataset import Dataset
import numpy as np
import configparser


class IRHDataset(Dataset):
    def __init__(self):
        subjects = np.setdiff1d(range(1, 22), [1, 2, 10, 12, 13, 14, 15, 17, 20, 22])
        Dataset.__init__(self, name='IRH', num_classes=24, segment_length=100, num_channels=3, num_loc=5,
                         subjects=subjects)
        self.XR, self.YR, self.SR = self.read_data()

    def read_data(self):
        XR = []
        YR = []
        SR = []
        import datetime
        import os.path
        X = []
        for i in self.subjects:
            numAc = 0
            print('before: ', i, datetime.datetime.now())
            for mvt in range(1, 27):
                for loc in range(1, 6):
                    filename = self.config[self.name]['path'] + 'S' + str(i) + '\sub' + str(i) + '-mvt' + str(
                        mvt) + '-loc' + str(loc) + '.csv'
                    if not os.path.exists(filename):
                        continue
                    sig = np.genfromtxt(filename, delimiter=',', dtype=str)
                    if loc == 1:
                        X = sig[:, 0:3]
                    else:
                        X = np.hstack((X, sig[:, 0:3]))
                # print(mvt, X.shape)
                # self.removeBlanks(X)
                Y = (np.ones(shape=len(X)) * mvt).tolist()
                X1, Y1 = self.segment(np.asarray(X, dtype=float), np.asarray(Y, dtype=int))
                XR += X1
                YR += Y1
                numAc += len(Y1)
            SR += (np.ones(shape=numAc) * i).tolist()
        XR = np.asarray(XR, dtype=float)
        YR = np.asarray(YR, dtype=int)
        SR = np.asarray(SR, dtype=int)
        YR[YR == 16] = 14
        YR[YR == 17] = 15
        YR[YR == 25] = 16
        YR[YR == 26] = 17
        return self.removeZeros(XR, YR, SR)

    def segment(self, X, Y):
        from scipy import stats
        window = 2 * 50  # 3-sec (50 Hz) window similar to the orignial paper
        overlap = 25  # 1.5 seconds overlap
        start = 0
        Xs = []
        Ys = []

        while (start + window) < len(X):
            if np.any(X[start: start + window, :] == -999999.0):
                start += window - overlap
                continue
            for i in range(self.num_loc):
                currentX = X[start: start + window, i * 3:(i * 3) + 3]
                currentY = stats.mode(Y[start: start + window])[0][0]
                Xs.append(currentX.T)
                # print(i, np.asarray(Xs).shape)
                Ys.append(currentY)
            start += window - overlap
        # Xs = np.array(Xs)
        # Ys = np.asarray(Ys)

        return Xs, Ys

    def removeZeros(self, X, Y, S):
        # idx = np.where(Y == 0)
        # idx = idx[0]
        mask = (Y == 0)
        return X[~mask], Y[~mask], S[~mask]

    def removeBlanks(self, S):
        # S = self.fillBlankWithZeros(S)

        # S = S.astype(float)
        while np.any(S == '-999999'):
            S2 = list(S)
            S2.append(S2[-1])
            S2 = S2[1:len(S2) + 1]
            S = np.where(S == '-999999', S2, S)
        return S
