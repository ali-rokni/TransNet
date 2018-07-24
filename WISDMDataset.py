from Dataset import Dataset
import numpy as np


class WISDMDataset(Dataset):
    def __init__(self, lazy=False):
        subjects = np.setdiff1d(range(1, 37), 19)
        Dataset.__init__(self, name='WISDM', num_classes=6, segment_length=200, num_channels=3, num_loc=1, subjects=subjects)
        if lazy != True:
            self.XR, self.YR, self.SR = self.read_data()
        # self.XR, self.YR, self.SR = self.segment(XR, YR, SR)
        # TODO: fix this
        # X1 = XR[:, 0, :]
        # X2 = XR[:, 1, :]
        # X3 = XR[:, 2, :]
        #
        # m = max(np.max(abs(X1)), np.max(abs(X2)), np.max(abs(X3)))
        # self.X1 = X1 + m
        # self.X2 = X2 + m
        # self.X3 = X3 + m
        # self.YR = YR
        # self.SR = SR

    def getXYs(self):
        return self.X1, self.X2, self.X3, self.YR, self.SR

    def read_data(self):
        filename = self.config[self.name]['path'] + 'raw_cleaned.csv'
        sig = np.genfromtxt(filename, delimiter=',', dtype=str)
        activity_names = ['Nothing', 'Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']
        S = self.removeBlanks(sig[:, 0])
        Y = self.removeBlanks(sig[:, 1])
        X = self.removeBlanks(sig[:, 3:6])
        Y = [activity_names.index(a) for a in Y]
        print(X.shape)
        S = S.astype(int)
        for i in range(1, 37):
            idx = np.where(S == i)
            i = idx[0].tolist()
            YY = [Y[a] for a in i]
            # print(len(np.unique(YY)))

        return self.segment(X.astype(float), Y, S)

    def fillBlankWithZeros(self,S):
        if type(S) is np.ndarray:
            size = S.shape
        else:
            size = len(S)
        return np.where(S == '', np.zeros(size, dtype=int), S)

    def removeBlanks(self, S):
        S = self.fillBlankWithZeros(S)

        # S = S.astype(float)
        while np.any(S == '0'):
            S2 = list(S)
            S2.append(S2[-1])
            S2 = S2[1:len(S2) + 1]
            S = np.where(S == '0', S2, S)
        return S

    def segment(self, X, Y, S):
        from scipy import stats
        window = 10 * 20  # 10-sec (20 Hz) window similar to the orignial paper
        overlap = 50  # 2.5 seconds overlap
        start = 0
        Xs = []
        Ys = []
        Ss = []

        while (start + window) < len(X):
            currentS = np.unique(S[start: start + window])
            if len(currentS) > 2 or len(currentS) == 0:
                raise Exception('More than 2 or less than 1 subject(s) in one segment!!')
            elif len(currentS) == 2:
                start += np.where(S[start: start + window] == S[start + window - 1])[0][0]
                continue
            else:
                currentX = X[start: start + window, :]
                currentY = stats.mode(Y[start: start + window])[0][0]
                Xs.append(currentX.T)
                if len(currentS) > 1:
                    print(currentS)
                Ss.append(currentS)
                Ys.append(currentY)
                start += window - overlap
        Xs = np.asarray(Xs)
        Ys = np.asarray(Ys)
        Ss = np.asarray(Ss)
        return Xs, Ys, Ss