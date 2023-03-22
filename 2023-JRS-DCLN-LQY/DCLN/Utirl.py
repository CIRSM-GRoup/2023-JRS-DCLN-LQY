from torch import nn
import torch
import torch.nn.functional as F
from torch import optim
from datass import loadData,applyFA,splitTrainTestSet,padWithZeros,createImageCubes,one_hot,four_rotation,DA,four_rotation_2
import numpy as np
import random
from scipy.stats import entropy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import heapq

windowSize = 11
#windowSize = 9
class_num = 9
K = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 1

def attention(x, y):
    x = torch.from_numpy(x)
    k = 60
    count = 0
    for i in range(0, x.shape[0]):
        if y[i] == 0:
            None
        else:
            x_al = x[i]
            x_al = x_al.reshape(11*11, -1)
            x_al = x_al.numpy()
            center = x_al[60].reshape(1, -1)
            x_center = np.ones((121, 20)) * center
            A = x_al - x_center
            l = np.sum(A * A, axis=1)
            l = np.argsort(l)
            re = l[-24:]
            idx = list(set(re))
            x_al[idx] = 0.2 * x_al[idx]
            x[i] = torch.from_numpy(x_al.reshape((-1, 11, 11, 20)))
        if i % 100 ==0:
            print(i)
    return x


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)

class SE(nn.Module):
    def __init__(self, ch_in, reduction=7):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        a, b, c, _, _ = x.size()
        y = self.avg_pool(x).view(a, b, c)  # squeeze操作
        y = self.fc(y).view(a, b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上

class HybridSN1(nn.Module):
    # 定义各个层的部分
    def __init__(self):
        super(HybridSN1, self).__init__()
        self.S = windowSize
        self.L = K
        self.f = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3)),
                               nn.ReLU(),
                               nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3)),
                               nn.ReLU(),
                               nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3)),
                               nn.ReLU(),
                               Reshape(256, 5, 5),
                               #Reshape(256, 3, 3),
                               nn.Conv2d(256, 64,
                                         kernel_size=(3, 3)),
                               nn.ReLU())

        # self.dense1 = nn.Linear(16384, 256)
        self.g = nn.Sequential(nn.Linear(576, 256),
                               nn.BatchNorm1d(256),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(256, 128),
                               nn.BatchNorm1d(128),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(128, 64),
                               nn.BatchNorm1d(64),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(64, class_num)
                               )
        pass

    def forward(self, x):
        out = self.f(x)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.g(out)
        return out
        pass

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(device))

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax
# def softmax(x):
#     x = x - np.max(x)
#     x = np.exp(x)/np.sum(np.exp(x))
#     return softmax

def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """

    indices = np.nonzero(gt)
    X = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
        train_size = int(train_size)

    if mode == 'random':
        train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y)
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[train_indices] = gt[train_indices]
        test_gt[test_indices] = gt[test_indices]
    elif mode == 'fixed':
        # print("Sampling {} with train size = {}".format(mode, train_size))
        train_indices, test_indices = [], []
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))  # x,y features
            ##================
            classsize = len(X)
            train_size2 = train_size
            if classsize <= train_size:
                train_size2 = classsize - 1
            ##====================
            train, test = sklearn.model_selection.train_test_split(X, train_size=train_size2, random_state=random_seed)
            train_indices += train
            test_indices += test
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[train_indices] = gt[train_indices]
        test_gt[test_indices] = gt[test_indices]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt

def cross_entropy_error(y, t):
    delta=1e-7  #添加一个微小值可以防止负无限大(np.log(0))的发生。
    return -np.sum(t*np.log(y+delta))

def sample_gt1(img, gt, train_gt, test_gt, SAMPLE_PERCENTAGE, M):
    X = img
    Y = gt
    row, col, n_band = X.shape
    num_class = np.max(Y)
    numm = np.zeros((row, col))
    for i in range(1, num_class + 1):
        index = np.where(train_gt == i)
        index2 = np.where(test_gt == i)

        if i == 1:
            array1_train = index[0]
            array2_train = index[1]
            array1_test = index2[0]
            array2_test = index2[1]
        else:
            array1_train = np.concatenate((array1_train, index[0]))
            array2_train = np.concatenate((array2_train, index[1]))
            array1_test = np.concatenate((array1_test, index2[0]))
            array2_test = np.concatenate((array2_test, index2[1]))

    y_train = Y[array1_train, array2_train]
    trueEDtimesSID = []
    trueEDtimesSID2 = []
    sumtrueES = []
    sumfalseES = []
    pseudo_labels3 = np.zeros([row, col, num_class + 1])
    for i in range(0, len(array1_test)):
        if i % 1000 == 0:
            print("i:%d" % (i))
        # if i%200!=0:
        #    continue
        xtest = array1_test[i]
        ytest = array2_test[i]
        labeltest = Y[xtest, ytest]
        specvectortest = X[xtest, ytest]
        EDs = np.zeros(num_class)
        SIDs = np.zeros(num_class)
        EDtimesSIDs = np.zeros(num_class)
        EDtimesSIDs2 = np.zeros(num_class)
        minED = 10000000000
        for j in range(1, num_class + 1):  # 类别循环
            index2 = np.where(y_train == j)  ## 当前类别序号
            index2 = index2[0]
            EDsclass = []
            SIDclass = []
            EDtimesSIDclass = []
            for nn in range(0, len(index2)):  # 类别内训练集循环 nn
                # print(index2[nn])##当前训练样本序号
                ind = index2[nn]
                xtrain = array1_train[ind]
                ytrain = array2_train[ind]
                specvectortrain = X[xtrain, ytrain]
                ED = np.sqrt(np.square(xtest - xtrain) + np.square(ytest - ytrain))
                SID1 = cross_entropy_error(specvectortest, specvectortrain)
                SID2 = cross_entropy_error(specvectortrain, specvectortest)
                # SID1 = entropy(specvectortest, specvectortrain)
                # SID2 = entropy(specvectortrain, specvectortest)
                SID = SID1 + SID2
                EDtimesSID = np.sqrt(ED * SID)
                ED = ED + SID
                EDsclass.append(ED)
                SIDclass.append(SID)
                EDtimesSIDclass.append(EDtimesSID)

                if ED < minED:
                    minED = ED
            # =================================
            inde = np.argsort(EDsclass)

            jiaquan = 0
            for nn in range(0, len(index2)):
                jiaquandis = EDsclass[inde[nn]] * (float(num_class) ** (-nn))  # 类别内训练集循环 nn
                jiaquan = jiaquan + jiaquandis

            EDs[j - 1] = jiaquan
            SIDs[j - 1] = np.min(SIDclass)
            EDtimesSIDs[j - 1] = np.min(EDtimesSIDclass)

        a = np.min(EDtimesSIDs)
        numm[xtest, ytest] = a
    numm = numm.reshape(-1, 1).squeeze(-1)
    index = np.where(numm != 0)[0]
    numm = numm[index]
    numm = np.sort(numm)
    T = numm[round(len(numm) / M)]
    return T

def sample_gt2(img, gt, train_gt, test_gt, SAMPLE_PERCENTAGE, t):
    X = img
    Y = gt
    row, col, n_band = X.shape
    num_class = np.max(Y)
    for i in range(1, num_class + 1):
        index = np.where(train_gt == i)
        index2 = np.where(test_gt == i)

        if i == 1:
            array1_train = index[0]
            array2_train = index[1]
            array1_test = index2[0]
            array2_test = index2[1]
        else:
            array1_train = np.concatenate((array1_train, index[0]))
            array2_train = np.concatenate((array2_train, index[1]))
            array1_test = np.concatenate((array1_test, index2[0]))
            array2_test = np.concatenate((array2_test, index2[1]))

    y_train = Y[array1_train, array2_train]
    trueEDtimesSID = []
    trueEDtimesSID2 = []
    sumtrueES = []
    sumfalseES = []
    pseudo_labels3 = np.zeros([row, col, num_class + 1])
    for i in range(0, len(array1_test)):
        if i % 1000 == 0:
            print("i:%d" % (i))
        #if i%200!=0:
        #    continue
        xtest = array1_test[i]
        ytest = array2_test[i]
        labeltest = Y[xtest, ytest]
        specvectortest = X[xtest, ytest]
        EDs = np.zeros(num_class)
        SIDs = np.zeros(num_class)
        EDtimesSIDs = np.zeros(num_class)
        EDtimesSIDs2 = np.zeros(num_class)
        minED = 10000000000
        for j in range(1, num_class + 1):  # 类别循环
            index2 = np.where(y_train == j)  ## 当前类别序号
            index2 = index2[0]
            EDsclass = []
            SIDclass = []
            EDtimesSIDclass = []
            for nn in range(0, len(index2)):  # 类别内训练集循环 nn
                # print(index2[nn])##当前训练样本序号
                ind = index2[nn]
                xtrain = array1_train[ind]
                ytrain = array2_train[ind]
                specvectortrain = X[xtrain, ytrain]
                ED = np.sqrt(np.square(xtest - xtrain) + np.square(ytest - ytrain))
                SID1 = cross_entropy_error(specvectortest, specvectortrain)
                SID2 = cross_entropy_error(specvectortrain, specvectortest)
                SID = SID1 + SID2
                EDtimesSID = np.sqrt(ED * SID)
                ED = ED + SID
                EDsclass.append(ED)
                SIDclass.append(SID)
                EDtimesSIDclass.append(EDtimesSID)

                if ED < minED:
                    minED = ED
            # =================================
            inde = np.argsort(EDsclass)

            jiaquan = 0
            for nn in range(0, len(index2)):
                jiaquandis = EDsclass[inde[nn]] * (float(num_class) ** (-nn))  # 类别内训练集循环 nn
                jiaquan = jiaquan + jiaquandis

            EDs[j - 1] = jiaquan
            SIDs[j - 1] = np.min(SIDclass)
            EDtimesSIDs[j - 1] = np.min(EDtimesSIDclass)
            ###
            jiaquan2 = 0
            inde2 = np.argsort(EDtimesSIDclass)
            for nn in range(0, len(index2)):
                jiaquandis = EDtimesSIDclass[inde2[nn]] * (float(num_class) ** (-nn))  # 类别内训练集循环 nn
                jiaquan2 = jiaquan2 + jiaquandis
            EDtimesSIDs2[j - 1] = jiaquan2
        ###

        # ========================
        # print("minED:", minED)
        # if minED>2.71:
        if np.min(EDtimesSIDs) > t:
            continue
        else:
            minn = np.min(EDs)
            softm = softmax(16 / EDs)
            softm2 = softmax(-EDs * num_class)
            softm3 = softmax(-EDtimesSIDs2 * num_class*100)
        labeEDtimesSIDs = np.argmin(EDtimesSIDs) + 1
        labeEDtimesSIDs2 = np.argmin(EDtimesSIDs2) + 1
        train_gt[xtest, ytest] = labeEDtimesSIDs
        pseudo_labels3[xtest, ytest][1:17] = softm3


        if labeEDtimesSIDs == labeltest:
            trueEDtimesSID.append(1)
            sumtrueES.append(np.min(EDtimesSIDs))
        else:
            trueEDtimesSID.append(0)
            sumfalseES.append(np.min(EDtimesSIDs))
            print("falseEDtimesSID:", np.min(EDtimesSIDs),"label:", labeEDtimesSIDs)

        if labeEDtimesSIDs2 == labeltest:
            trueEDtimesSID2.append(1)
        else:
            trueEDtimesSID2.append(0)


    accuEDtimesSID2 = np.sum(trueEDtimesSID2) / len(trueEDtimesSID2)
    accuEDtimesSID = np.sum(trueEDtimesSID) / len(trueEDtimesSID)
    pseudo_labels_idx = train_gt
    print("lenEDtimesSID: %d, accurate:%f, truenum:%d" % (
    len(trueEDtimesSID), 100 * accuEDtimesSID, np.sum(trueEDtimesSID)))
    print("lenEDtimesSID2: %d, accurate:%f, truenum:%d" % (
        len(trueEDtimesSID2), 100 * accuEDtimesSID2, np.sum(trueEDtimesSID2)))
    return pseudo_labels_idx, pseudo_labels3

def sample_gt3(img, gt, train_gt, test_gt, SAMPLE_PERCENTAGE):
    X = img
    Y = gt
    row, col, n_band = X.shape
    num_class = np.max(Y)
    o_index = np.where(train_gt != 0)
    for i in range(1, num_class + 1):
        index = np.where(train_gt == i)
        index2 = np.where(test_gt == i)

        if i == 1:
            array1_train = index[0]
            array2_train = index[1]
            array1_test = index2[0]
            array2_test = index2[1]
        else:
            array1_train = np.concatenate((array1_train, index[0]))
            array2_train = np.concatenate((array2_train, index[1]))
            array1_test = np.concatenate((array1_test, index2[0]))
            array2_test = np.concatenate((array2_test, index2[1]))
    y_train = Y[array1_train, array2_train]
    trueEDtimesSID = []
    trueEDtimesSID2 = []
    sumtrueES = []
    sumfalseES = []
    num_gt = np.zeros([row, col])
    for i in range(0, len(array1_test)):
        if i % 1000 == 0:
            print("i:%d" % (i))
        #if i%200!=0:
        #    continue
        xtest = array1_test[i]
        ytest = array2_test[i]
        labeltest = Y[xtest, ytest]
        specvectortest = X[xtest, ytest]
        EDtimesSIDs = np.zeros(num_class)
        for j in range(1, num_class + 1):  # 类别循环
            index2 = np.where(y_train == j)  ## 当前类别序号
            index2 = index2[0]
            EDtimesSIDclass = []
            for nn in range(0, len(index2)):  # 类别内训练集循环 nn
                # print(index2[nn])##当前训练样本序号
                ind = index2[nn]
                xtrain = array1_train[ind]
                ytrain = array2_train[ind]
                specvectortrain = X[xtrain, ytrain]
                ED = np.sqrt(np.square(xtest - xtrain) + np.square(ytest - ytrain))
                SID1 = cross_entropy_error(specvectortest, specvectortrain)
                SID2 = cross_entropy_error(specvectortrain, specvectortest)
                SID = SID1 + SID2
                EDtimesSID = np.sqrt(ED * SID)
                EDtimesSIDclass.append(EDtimesSID)
            # =================================
            EDtimesSIDs[j - 1] = np.min(EDtimesSIDclass) #得到到每一类的平均距离
        # ========================
        ss = sorted(softmax(-EDtimesSIDs)) # 得到每一类的概率,并排序
        num_gt[xtest, ytest] = ss[-1] * (ss[-1] - ss[-2]) #得到每一个样本的BVSB
        labeEDtimesSIDs = np.argmin(EDtimesSIDs) + 1 # 得到每一类的标签
        train_gt[xtest, ytest] = labeEDtimesSIDs

        # 打印准确率
        if labeEDtimesSIDs == labeltest:
            trueEDtimesSID.append(1)
            sumtrueES.append(np.min(EDtimesSIDs))
        else:
            trueEDtimesSID.append(0)
            sumfalseES.append(np.min(EDtimesSIDs))
            print("falseEDtimesSID:", np.min(EDtimesSIDs),"label:", labeEDtimesSIDs)

    accuEDtimesSID = np.sum(trueEDtimesSID) / len(trueEDtimesSID)
    pseudo_labels_idx = train_gt
    print("lenEDtimesSID: %d, accurate:%f, truenum:%d" % (
    len(trueEDtimesSID), 100 * accuEDtimesSID, np.sum(trueEDtimesSID)))
    return num_gt, pseudo_labels_idx, o_index

def sample_choice(num_gt, Y, pseudo_labels_idx, o_index, t):
    # 存一下原来的标签样本
    pl_choice = np.zeros((610, 340))
    pl_choice[o_index] = pseudo_labels_idx[o_index]
    print(len(np.where(pl_choice != 0)[0]))
    # 选取标签样本
    for i in range(1, class_num+1):
        pl_choice = pl_choice.reshape(-1, 1)
        pseudo_labels_idx = pseudo_labels_idx.reshape(-1, 1)
        numm = num_gt.reshape(-1, 1).squeeze(-1)
        o = np.where(pseudo_labels_idx == i)[0]
        id = np.where(pseudo_labels_idx != i)[0]
        numm[id] = 0
        print(len(np.where(numm != 0)[0]))
        sor = sorted(range(len(numm)), key=lambda k: numm[k])  # 返回索引
        index2 = sor[-(round(len(o) / t)):]
        pl_choice[index2] = pseudo_labels_idx[index2]
        print(len(np.where(pl_choice != 0)[0]))
        pl_choice = pl_choice.reshape((610, 340))
        # 打印准确率
        idd = np.where(pl_choice != 0)
        a1 = pl_choice[idd]
        a2 = Y[idd]
        acc = np.sum(a1 == a2) / len(idd[0])
        print(acc)
    return pl_choice

def sample_choice2(num_gt, Y, pseudo_labels_idx, o_index, t):
    # 存一下原来的标签样本
    pl_choice = np.zeros((610, 340))
    pl_choice[o_index] = pseudo_labels_idx[o_index]
    num = num_gt
    num = num.reshape(-1, 1).squeeze(-1)
    numm = np.zeros((610*340,))
    pl_choice = pl_choice.reshape(-1, 1)
    pseudo_labels_idx = pseudo_labels_idx.reshape(-1, 1)
    # numm = num_gt.reshape(-1, 1).squeeze(-1)
    o = np.where(pseudo_labels_idx != 0)[0]
    numm[o] = num[o]
    sor = sorted(range(len(numm)), key=lambda k: numm[k])  # 返回索引
    #index2 = sor[-(round(len(o) / t)):]
    index2 = sor[-t:]
    pl_choice[index2] = pseudo_labels_idx[index2]
    print(len(np.where(pl_choice != 0)[0]))
    pl_choice = pl_choice.reshape((610, 340))
        # 打印准确率
    idd = np.where(pl_choice != 0)
    a1 = pl_choice[idd]
    a2 = Y[idd]
    acc = np.sum(a1 == a2) / len(idd[0])
    print(acc)
    return pl_choice


