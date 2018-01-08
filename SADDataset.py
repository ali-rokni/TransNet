from Dataset import Dataset
import numpy as np
class SADDataset(Dataset):
    def __init__(self):
        subjects = range(1,9)
        Dataset.__init__(self, name='SAD', num_classes=19, segment_length=125, num_channels=3, num_loc=5, subjects=subjects)
        X1 = np.genfromtxt(self.name + '_allX1.csv', delimiter=',')
        X2 = np.genfromtxt(self.name + '_allX2.csv', delimiter=',')
        X3 = np.genfromtxt(self.name + '_allX3.csv', delimiter=',')
        self.YR = np.genfromtxt(self.name + '_allY.csv', delimiter=',')
        self.SR = np.genfromtxt(self.name + '_allS.csv', delimiter=',')
        a, b = X1.shape
        self.XR = np.empty(shape=(a, 3, b))
        self.XR[:, 0, :] = X1
        self.XR[:, 1, :] = X2
        self.XR[:, 2, :] = X3