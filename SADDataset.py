from Dataset import Dataset
import numpy as np
import os


class SADDataset(Dataset):
    data_folder = 'sad_data'

    def __init__(self):
        subjects = range(1, 9)
        Dataset.__init__(self, name='SAD', num_classes=19, segment_length=125, num_channels=3, num_loc=5,
                         subjects=subjects)
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            self.write_to_file()
        x1 = np.genfromtxt(self.name + '_allX1.csv', delimiter=',')
        x2 = np.genfromtxt(self.name + '_allX2.csv', delimiter=',')
        x3 = np.genfromtxt(self.name + '_allX3.csv', delimiter=',')
        self.YR = np.genfromtxt(self.name + '_allY.csv', delimiter=',')
        self.SR = np.genfromtxt(self.name + '_allS.csv', delimiter=',')
        a, b = x1.shape
        self.XR = np.empty(shape=(a, 3, b))
        self.XR[:, 0, :] = x1
        self.XR[:, 1, :] = x2
        self.XR[:, 2, :] = x3

    def write_to_file(self):
        n = self.num_classes * len(self.subjects) * self.num_loc * 60  # number of segments
        x1 = np.empty(shape=(n, self.segment_length), dtype=float)
        x2 = np.empty(shape=(n, self.segment_length), dtype=float)
        x3 = np.empty(shape=(n, self.segment_length), dtype=float)
        y = np.empty(shape=(n, 1), dtype=int)
        z = np.empty(shape=(n, 1), dtype=int)
        s = np.empty(shape=(n, 1), dtype=int)
        i = 0
        for sub in self.subjects:
            for mvt in range(1, self.num_classes + 1):
                for seg in range(1, 61):
                    for mote in range(1, self.num_loc + 1):
                        filename = self.config[self.name]['path'] + 'a' + str(mvt).zfill(2) + '\\p' + str(
                            sub) + '\\s' + str(seg).zfill(2) + '.txt'
                        sig = np.genfromtxt(filename, delimiter=',')
                        start = (mote - 1) * 9 + 1
                        # c = 3
                        A = sig[:, start:start + 1]
                        x1[i, :] = A.reshape(self.segment_length, )
                        A = sig[:, start + 1:start + 2]
                        x2[i, :] = A.reshape(self.segment_length, )
                        A = sig[:, start + 2:start + 3]
                        x3[i, :] = A.reshape(self.segment_length, )
                        y[i] = mvt
                        z[i] = mote
                        s[i] = sub
                        i = i + 1
        np.savetxt(self.data_folder + '/SAD_allX1.csv', x1, delimiter=',')
        np.savetxt(self.data_folder + '/SAD_allX2.csv', x2, delimiter=',')
        np.savetxt(self.data_folder + '/SAD_allX3.csv', x3, delimiter=',')
        np.savetxt(self.data_folder + '/SAD_allY.csv', y, delimiter=',')
        np.savetxt(self.data_folder + '/SAD_allZ.csv', z, delimiter=',')
        np.savetxt(self.data_folder + '/SAD_allS.csv', s, delimiter=',')
