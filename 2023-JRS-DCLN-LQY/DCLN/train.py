from torch import nn
import torch
import torch.nn.functional as F
from torch import optim
from datass import loadData, applyFA, splitTrainTestSet, padWithZeros, createImageCubes, one_hot, four_rotation, DA, four_rotation_2
import numpy as np
import random
from scipy.stats import entropy
import sklearn
from Utirl import sample_gt,sample_gt1, sample_gt2, sample_gt3, softmax, Reshape, HybridSN1, ContrastiveLoss, attention,sample_choice2
from sklearn.preprocessing import minmax_scale
import datetime

# Hyper Parameters
windowSize = 11
class_num = 9
K = 20
batch_size = 9
temperature = 0.5

dataset = 'UP'
SAMPLE_PERCENTAGE = 5
X, Y = loadData(dataset)
x = X.astype(np.float32)
y = Y.astype(np.float32)
x,fa = applyFA(x, numComponents=K)
x, y = createImageCubes(x, y, windowSize=windowSize, removeZeroLabels=False)

# attention
x = attention(x)
np.save('x_at.npy', x)
x = np.load('x_at_he.npy')
train_gt, test_gt = sample_gt(Y, SAMPLE_PERCENTAGE, mode='fixed')
# testing datasets
indices2 = np.nonzero(train_gt.reshape(-1,1))
indices2 = indices2[0]
Train_x = x[indices2]
Train_y = y[indices2]

Train_x = Train_x.reshape(-1, windowSize, windowSize, K, 1)
Train_x = Train_x.transpose(0, 4, 3, 1, 2)
# to tensor
Train_x = torch.from_numpy(Train_x)
Train_y = torch.from_numpy(Train_y)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class TrainDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Train_x.shape[0] * 20
        self.Train_x = torch.FloatTensor(Train_x.float())
        self.Train_y = torch.FloatTensor(Train_y.float())
        self.index1 = 0
    def __getitem__(self, index):
        index = index + 1
        if self.index1 == 0:
            self.index1 = index/index
        self.index1 = self.index1 + index/index
        idd = np.where(Train_y == self.index1 - 1)[0]
        idd = list(idd)
        index1 = random.choice(idd)
        index2 = random.choice(idd)
        if self.index1 > 9:
            self.index1 = 0

        return self.Train_x[index1], self.Train_x[index2], self.Train_y[index1]

    def __len__(self):

        return self.len
trainset = TrainDS()
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=9, shuffle=True, num_workers=0)
def cross_entropy_error(y, t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta))
def c_e(y, t):
    to_loss = 0
    for i in range(0,len(y)):
        lossi = cross_entropy_error(y[i, :], t[i, :])
        to_loss = to_loss + lossi
    return to_loss/len(y)
net = HybridSN1().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
Loss_cl = ContrastiveLoss(batch_size, temperature).to(device)
Loss_ce = nn.CrossEntropyLoss().to(device)
net.train()
total_loss = 0
total_acc = 0
for epoch in range(200):
    for i, (inputs_1, inputs_2, y) in enumerate(train_loader):
        #print(inputs.detach().cpu().numpy().shape)
        inputs_1 = inputs_1.to(device)
        inputs_2 = inputs_2.to(device)
        y = y - 1
        y = y.to(device)
        optimizer.zero_grad()
        outputs_1 = net(inputs_1)
        outputs_2 = net(inputs_2)

        loss_1 = Loss_cl(outputs_1, outputs_2)
        loss_2 = (Loss_ce(outputs_1, y.long()) + Loss_ce(outputs_2, y.long()))/2
        loss =  loss_1 + loss_2

        loss.backward(retain_graph=True)
        pre_1 = np.argmax(outputs_1.detach().cpu().numpy(), axis=1)
        pre_2 = np.argmax(outputs_2.detach().cpu().numpy(), axis=1)
        acc_1 = np.sum(pre_1 == y.detach().cpu().numpy()) / batch_size
        acc_2 = np.sum(pre_2 == y.detach().cpu().numpy()) / batch_size

        optimizer.step()
        total_loss += loss.item()
        if i % 10 == 0:
            print('[loss: %.4f] [acc1: %.4f] [acc2: %.4f]' % (loss.item(), acc_1.item(), acc_2.item()))
            #print('[loss: %.4f]   [acc: %.4f]' % (loss.item(), acc.item()))
    print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f] ' % (epoch + 1, total_loss / ((epoch + 1)*len(train_loader)), loss.item() ))
print('Finished Training')
torch.save(net, 'final_5-1.pth')


# class
X, Y = loadData(dataset)
x = X.astype(np.float32)
y = Y.astype(np.float32)
x,fa = applyFA(x, numComponents=K)
x, y = createImageCubes(x, y, windowSize=windowSize, removeZeroLabels=False)
x = np.load('x_at_he.npy')
x = x.reshape(-1, windowSize, windowSize, K, 1)
x = x.transpose(0, 4, 3, 1, 2)
x = torch.from_numpy(x)
y = torch.from_numpy(y)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class XDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = x.shape[0]
        self.Test_x = torch.FloatTensor(x.float())
        self.Test_y = torch.FloatTensor(y.float())
    def __getitem__(self, index):
        return self.Test_x[index], self.Test_y[index]
    def __len__(self):
        return self.len
Xset = XDS()
test_loader = torch.utils.data.DataLoader(dataset=Xset, batch_size=29, shuffle=False, num_workers=0)

net = torch.load('final_5-1.pth')
net.eval()
print(net)
count = 0
for i, (X, _) in enumerate(test_loader):
    # print(inputs.detach().cpu().numpy().shape)
    X = X.to(device)
    outputs = net(X)
    if count == 0:
        fea_x = outputs.detach().cpu().numpy()
        count = 1
    else:
        fea_x = np.concatenate((fea_x, outputs.detach().cpu().numpy()))
print(fea_x.shape)
fea_x = minmax_scale(fea_x, feature_range=(0, 1), axis=0, copy=True)
fea_x = fea_x.reshape(Y.shape[0],Y.shape[1],class_num)
fea_x[Y == 0] = 0
print(fea_x.shape)
print(fea_x)
T = sample_gt1(fea_x, Y, train_gt, test_gt, SAMPLE_PERCENTAGE, 40)
label_pl_idx, label_pl = sample_gt2(fea_x, Y, train_gt, test_gt, SAMPLE_PERCENTAGE, T)
np.save('final_5-1.npy', label_pl_idx)

# re train
X, Y = loadData(dataset)
x = X.astype(np.float32)
y = Y.astype(np.float32)
x,fa = applyFA(x, numComponents=K)
x, y = createImageCubes(x, y, windowSize=windowSize, removeZeroLabels=False)
x = np.load('x_at_he.npy')
indices = np.nonzero(test_gt.reshape(-1,1))
indices = indices[0]
Test_x = x[indices]
Test_y = y[indices] - 1
Test_x = Test_x.reshape(-1, windowSize, windowSize, K, 1)
Test_x = Test_x.transpose(0, 4, 3, 1, 2)
Test_x = torch.from_numpy(Test_x)
Test_y = torch.from_numpy(Test_y)
label_pl = np.load('final_5-1.npy')
label_pl = torch.from_numpy(label_pl)
label_pl = label_pl.reshape(-1,1)
indices2 = np.nonzero(label_pl)
indices2 = indices2.numpy()[:,0]
Train_x = x[indices2]
Train_y = label_pl[indices2]
Train_y = Train_y.numpy().squeeze(-1)
Train_x = Train_x.reshape(-1, windowSize, windowSize, K, 1)
Train_x = Train_x.transpose(0, 4, 3, 1, 2)
Train_x = torch.from_numpy(Train_x)
Train_y = torch.from_numpy(Train_y)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class TrainDS2(torch.utils.data.Dataset):
    def __init__(self):
        self.len = SAMPLE_PERCENTAGE * class_num * 20
        self.Train_x = torch.FloatTensor(Train_x.float())
        self.Train_y = torch.FloatTensor(Train_y.float())
        self.index1 = 0
    def __getitem__(self, index):
        index = index + 1
        if self.index1 == 0:
            self.index1 = index/index
        self.index1 = self.index1 + index/index
        idd = np.where(Train_y == self.index1 - 1)[0]
        idd = list(idd)
        index1 = random.choice(idd)
        index2 = random.choice(idd)
        if self.index1 > 9:
            self.index1 = 0

        return self.Train_x[index1], self.Train_x[index2], self.Train_y[index1]

    def __len__(self):
        return self.len
trainset = TrainDS2()
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=9, shuffle=True, num_workers=0)
net = HybridSN1().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
Loss_cl = ContrastiveLoss(batch_size, temperature).to(device)
Loss_ce = nn.CrossEntropyLoss().to(device)
net.train()
total_loss = 0
total_acc = 0
for epoch in range(200):
    for i, (inputs_1, inputs_2, Y) in enumerate(train_loader):
        #print(inputs.detach().cpu().numpy().shape)
        inputs_1 = inputs_1.to(device)
        inputs_2 = inputs_2.to(device)
        Y = Y - 1
        Y = Y.to(device)
        optimizer.zero_grad()
        outputs_1 = net(inputs_1)
        outputs_2 = net(inputs_2)

        loss_1 = Loss_cl(outputs_1, outputs_2)
        loss_2 = (Loss_ce(outputs_1, Y.long()) + Loss_ce(outputs_2, Y.long()))/2
        loss = loss_1 + loss_2

        loss.backward(retain_graph=True)
        pre_1 = np.argmax(outputs_1.detach().cpu().numpy(), axis=1)
        pre_2 = np.argmax(outputs_2.detach().cpu().numpy(), axis=1)
        acc_1 = np.sum(pre_1 == Y.detach().cpu().numpy()) / batch_size
        acc_2 = np.sum(pre_2 == Y.detach().cpu().numpy()) / batch_size

        optimizer.step()
        total_loss += loss.item()
        if i % 10 == 0:
            print('[loss: %.4f] [acc1: %.4f] [acc2: %.4f]' % (loss.item(), acc_1.item(), acc_2.item()))
            #print('[loss: %.4f]   [acc: %.4f]' % (loss.item(), acc.item()))
    print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f] ' % (epoch + 1, total_loss / ((epoch + 1)*len(train_loader)), loss.item() ))

print('Finished Training')
torch.save(net, 'final_5-2.pth')

# re class
X, Y = loadData(dataset)
x = X.astype(np.float32)
y = Y.astype(np.float32)
x,fa = applyFA(x, numComponents=K)
x, y = createImageCubes(x, y, windowSize=windowSize, removeZeroLabels=False)
x = np.load('x_at_he.npy')
label_pl = np.load('final_5-1.npy')
train_gt = label_pl
test_gt = Y - train_gt
x = x.reshape(-1, windowSize, windowSize, K, 1)
x = x.transpose(0, 4, 3, 1, 2)
x = torch.from_numpy(x)
y = torch.from_numpy(y)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class XDS2(torch.utils.data.Dataset):
    def __init__(self):
        self.len = x.shape[0]
        self.Test_x = torch.FloatTensor(x.float())
        self.Test_y = torch.FloatTensor(y.float())

    def __getitem__(self, index):
        return self.Test_x[index], self.Test_y[index]

    def __len__(self):
        return self.len
Xset = XDS2()
test_loader = torch.utils.data.DataLoader(dataset=Xset, batch_size=29, shuffle=False, num_workers=0)
net = torch.load('final_5-2.pth')
net.eval()
print(net)
count = 0
for i, (X, _) in enumerate(test_loader):
    # print(inputs.detach().cpu().numpy().shape)
    X = X.to(device)
    outputs = net(X)
    if count == 0:
        fea_x = outputs.detach().cpu().numpy()
        count = 1
    else:
        fea_x = np.concatenate((fea_x, outputs.detach().cpu().numpy()))
print(fea_x.shape)
fea_x = minmax_scale(fea_x, feature_range=(0, 1), axis=0, copy=True)
fea_x = fea_x.reshape(Y.shape[0],Y.shape[1],class_num)
fea_x[Y == 0] = 0
print(fea_x.shape)
num_gt, label_pl_idx, o_index = sample_gt3(fea_x, Y, train_gt, test_gt, SAMPLE_PERCENTAGE)
np.save('l_p.npy', label_pl_idx)
np.save('n_g.npy', num_gt)
pl_choice = sample_choice2(num_gt, Y, label_pl_idx, o_index, 3000)
np.save('final-9.npy', pl_choice)

