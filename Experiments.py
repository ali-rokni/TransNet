from CNN_sensor import train_deep_model, retrain, make_list
from OPPDataset import OPPDataset
# from WISDMDataset import WISDMDataset
from IRHDataset import IRHDataset
from SADDataset import SADDataset
import numpy as np
import utils as ut
import os

max_tran = 5
num_methods = 1
num_metrics = 4
#   for j in range(1, 6):
dataset = OPPDataset()
# dataset= SADDataset()
# dataset = IRHDataset()
# dataset = WISDMDataset()
befores = np.empty(shape=(1, len(dataset.subjects), num_metrics))
afters = np.empty(shape=(max_tran, len(dataset.subjects), num_metrics))
folder_name = dataset.config['Result']['path']
#directory = os.path.dirname(folder_name)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)


def write_to_file():
    for i in range(max_tran):
        # ut.writeScores('results2/' + dataset.name + '_Before_' + str(i + 1) + '_', befores[i, :, :, :])
        ut.writeScores(folder_name + '/Deep_' + dataset.name + '_After_' + str(i + 1) + '_', afters[i, ...])
    ut.writeScores(folder_name + '/Deep_' + dataset.name + '_Before_', befores[0, ...])


def deepTester():
    c = 0
    for i in dataset.subjects:
        XT, YT, ST, XS, YS, SS = ut.separateY_into_2(dataset, [i], True)
        seed = 123
        np.random.seed(seed * 2)
        np.random.shuffle(XS)
        np.random.seed(seed * 2)
        np.random.shuffle(YS)
        if len(np.unique(YS)) < 2:
            continue
        Xt, Yt, Xs, Ys = ut.create_transfer_set(dataset, XS, YS, max_tran, SS, [i])
        score1, score2, model = train_deep_model(dataset, XT, YT, Xs, Ys)
        score_after, model = retrain(dataset, model, Xt, Yt, Xs, Ys, range(1, max_tran+1))

        befores[0, c, :] = score1
        afters[:, c, :] = score_after
        c += 1

    write_to_file()


def sequential_tester():
    c = 0
    for i in dataset.subjects:
        print('subject :' + str(i))
    # for i in range(1, 7):
        XT, YT, ST, XS, YS, SS = ut.create_test_and_train(dataset, np.setdiff1d(dataset.subjects, [i]), [i], True)

        if len(np.unique(YS)) < 2:
            continue
        Xt, Yt, Xs, Ys = ut.create_sequential_transfer_set(dataset, XS, YS, max_tran, SS, [i])
        score1, score2, model = train_deep_model(dataset, XT, YT, Xs, Ys)
        score2, model = retrain(dataset, model, Xt, Yt, Xs, Ys, range(1, max_tran + 1))
        befores[0, c, :] = score1[1:]
        afters[:, c, :] = score2
        c += 1

    write_to_file()


def location_tester():
    c = 0
    for i in dataset.subjects:
        print('subject :' + str(i))
    # for i in range(1, 7):
        XT, YT, ST, XS, YS, SS = ut.create_test_and_train_per_loc(dataset, [i], [i], [1], [3], True)

        if len(np.unique(YS)) < 2:
            continue
        # Xt, Yt, Xs, Ys = ut.transfer_percent(XS, YS, 0.3)
        Xt, Yt, Xs, Ys = ut.create_transfer_set(dataset, XS, YS, 26, SS, [i])
        score1, score2, model = train_deep_model(dataset, XT, YT, Xs, Ys, location=True)
        # score2, model = retrain(dataset, model, Xt, Yt, Xs, Ys, range(5, 26, 5), location=True)
        score2, model = retrain(dataset, model, Xt, Yt, Xs, Ys, range(len(Xt),len(Xt)+1), location=True)


        befores[0, c, :] = score1
        afters[:, c, :] = score2
        c += 1

    write_to_file()


def ipsn_tester():
    c = 0
    for i in dataset.subjects:
        print('subject :' + str(i))
    # for i in range(1, 7):
        XT, YT, ST, XS, YS, SS = ut.create_test_and_train_per_loc(dataset, [i], [i], [1], [3], True)

        if len(np.unique(YS)) < 2:
            continue

        XT1, YT1, XT2, YT2, XT3, YT3 = ut.divide_in_3(XT, YT)
        XS1, YS1, XS2, YS2, XS3, YS3 = ut.divide_in_3(XS, YS)
        score1, score2, model = train_deep_model(dataset, XT1, YT1, XT2, YT2, location=True)

        predicted_YS2 = model.predict(make_list(XT2, dataset.num_channels, dataset.segment_length))
        predicted_YS2 = np.argmax(predicted_YS2, 1)
        # score2, model = retrain(dataset, model, XS2, predicted_YS2, XS3, YS3, range(len(XS2), len(XS2)+1), location=True)
        score1, score2, model = train_deep_model(dataset, XS2, predicted_YS2, XS3, YS3, location=True)
        befores[0, c, :] = score1
        c += 1
        ut.writeScores(folder_name + '/Deep_' + dataset.name + '_IPSN_', befores[0, ...])
        print(score1)
        # print(score2)


def only_cross_sub():
    c = 0
    for i in dataset.subjects:
        print('subject :' + str(i))
        XT, YT, ST, XS, YS, SS = ut.create_test_and_train(dataset, np.setdiff1d(dataset.subjects, [i]), [i], True)
        score1, score2, model = train_deep_model(dataset, XT, YT, XS, YS)
        befores[0, c, :] = score1
        c += 1
    ut.writeScores('results8/Deep_' + dataset.name + '_Before_', befores[0, ...])


def testOverall():
    # dataset = WISDMDataset()
    # dataset = OPPDataset()
    XT, YT, ST, XS, YS, SS = dataset.getSplitedXY(True, 0.2, shuffle=False)
    # mask = np.in1d(ST, np.asarray([1]))
    # XT = XT[mask]
    # YT = YT[mask]
    # ST = ST[mask]
    # mask = np.in1d(SS, np.asarray([1]))
    # XS = XS[mask]
    # YS = YS[mask]
    # SS = SS[mask]

    score1, score2, model = train_deep_model(dataset, XT, YT, XS, YS)
    print(score1)
    print(score2)
    # ut.writeScores('results3/Deep_' + dataset.name + '_Overall_', score1[np.newaxis, ...])


# location_tester()
testOverall()