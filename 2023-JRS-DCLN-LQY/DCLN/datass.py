import scipy.io as sio
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FactorAnalysis
import torch
#from torchvision.transforms.functional import rotate as rot
from torchvision import transforms
def loadData(name):
    data_path = os.path.join(os.getcwd(), 'data')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'UP':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']

    return data, labels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=10, removeZeroLabels = True):
    margin = int((windowSize-1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin+1 , c - margin:c + margin+1 ]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def applyFA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    fa = FactorAnalysis(n_components=numComponents, random_state=0)
    newX = fa.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, fa

def one_hot(label, depth = 16):
    out = torch.zeros(32, depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim = 1, index = idx, value= 1)
    return out

def four_rotation(matrix_0):
    matrix_90 = np.rot90(matrix_0, k=1, axes=(3, 4))
    matrix_180 = np.rot90(matrix_90, k=1, axes=(3, 4))
    matrix_270 = np.rot90(matrix_180, k=1, axes=(3, 4))
    #train_x_1 = matrix_180
    #train_x_2 = matrix_270
    matrix_180 = matrix_180.copy()
    matrix_270 = matrix_270.copy()
    train_x_1 = torch.tensor(matrix_180)
    train_x_2 = torch.tensor(matrix_270)
    return train_x_1, train_x_2

def Crop_and_resize(data):
    da = transforms.RandomResizedCrop(11, scale=(0.8, 1.0), ratio=(0.7777, 1.3333))
    data = data.transpose(2, 0, 1)
    data = torch.from_numpy(data)
    x = da(data)
    x = x.numpy()
    x = x.transpose(1, 2, 0)
    return x

def four_rotation_2(matrix_0):
    matrix_90 = np.rot90(matrix_0, k=1, axes=(3, 4))
    matrix_180 = np.rot90(matrix_90, k=1, axes=(3, 4))
    matrix_270 = np.rot90(matrix_180, k=1, axes=(3, 4))
    #train_x_1 = matrix_180
    #train_x_2 = matrix_270
    matrix_180 = matrix_180.copy()
    matrix_270 = matrix_270.copy()
    return matrix_180, matrix_270

def DA(x):
    [matrix_0, matrix_90, matrix_180, matrix_270] = four_rotation(x)
    train_x_1 = np.concatenate((matrix_0,matrix_180 ),axis=2)
    train_x_2 = np.concatenate((matrix_90,matrix_270 ),axis=2)
    train_x_1 = torch.tensor(train_x_1)
    train_x_2 = torch.tensor(train_x_2)
    return train_x_1, train_x_2

