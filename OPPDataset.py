from Dataset import Dataset
import numpy as np


class OPPDataset(Dataset):
    def __init__(self):
        subjects = range(1,5)
        Dataset.__init__(self, name='OPP2', num_classes=4, segment_length=150, num_channels=3, num_loc=5, subjects=subjects)
        self.XR, self.YR, self.SR = self.read_data(mode=1)

    def read_data(self, mode = 1):
        XR = []
        YR = []
        SR = []
        import datetime
        for i in range(1,5):
            print('before: ', i, datetime.datetime.now())
            filename = 'C:\D\WSU\Research\Experiments\OpportunityUCIDataset\scripts\\benchmark\S' + str(i) + '.csv'
            sig = np.genfromtxt(filename, delimiter=',', dtype=str)
            a, b = sig.shape
            X = sig[:,1:b-2]
            if mode == 1:
                Y = sig[:,b-2]
            else:
                Y = sig[:,b-1]
                Y = np.asarray(Y).astype(float)
            X, Y = self.segment(np.asarray(X, dtype=float),np.asarray(Y).astype(int) )
            XR += X
            YR += Y
            SR += (np.ones(shape=len(Y)) * i).tolist()
        XR = np.asarray(XR, dtype=float) * 9.8 / 1000
        YR = np.asarray(YR, dtype=int)
        SR = np.asarray(SR, dtype=int)
        XR, YR, SR = self.removeZeros(XR, YR, SR)
        Ys = np.unique(YR)
        self.num_classes = len(Ys)
        YR2 = YR
        for j in range(len(Ys)):
            YR2[YR == Ys[j]] = j+1
        YR = YR2
        # YR[YR == 5] = 3
        return XR, YR, SR

    def segment(self, X, Y):
        from scipy import stats
        window = 5 * 30  # 5-sec (30 Hz) window similar to the orignial paper
        overlap = 45  # 1.5 seconds overlap
        start = 0
        Xs = []
        Ys = []

        while (start + window) < len(X):
            for i in range(self.num_loc):
                currentX = X[start: start + window, i*9:(i*9)+3]
                currentY = stats.mode(Y[start: start + window])[0][0]
                Xs.append(currentX.T)
                #print(i, np.asarray(Xs).shape)
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

