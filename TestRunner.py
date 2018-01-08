import numpy as np

import utils as ut
from WISDMDataset import WISDMDataset
from OPPDataset import OPPDataset
from IRHDataset import IRHDataset
from SADDataset import SADDataset

if __name__ == "__main__":
    import datetime

    dataset = OPPDataset()
    print('before: ', datetime.datetime.now())
    ut.computeResults(dataset)
    # ut.testOverallAccuracy(dataset, 0.2)