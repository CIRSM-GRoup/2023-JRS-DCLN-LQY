from torch import nn
import torch
import torch.nn.functional as F
from torch import optim
from datass import loadData, applyFA, splitTrainTestSet, padWithZeros, Crop_and_resize, createImageCubes, one_hot, four_rotation, DA, four_rotation_2
import numpy as np
import random
from scipy.stats import entropy
import sklearn
from Utirl import sample_gt, sample_gt3, softmax, Reshape, HybridSN1, ContrastiveLoss
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import classification_report

# Hyper Parameters
windowSize = 11
class_num = 9
K = 20
batch_size = 16
temperature = 0.5
dataset = 'UP'
SAMPLE_PERCENTAGE = 5
random_seed = 1

X, Y = loadData(dataset)
x = X.astype(np.float32)
y = Y.astype(np.float32)
x,fa = applyFA(x, numComponents=K)
x, y = createImageCubes(x, y, windowSize=windowSize, removeZeroLabels=False)
#np.save('y.npy',y)
# y = np.load('y.npy')
x = np.load('x_at_he.npy')

train_gt, test_gt = sample_gt(Y, SAMPLE_PERCENTAGE, mode='fixed')
indices = np.nonzero(Y.reshape(-1,1))
indices = indices[0]
Test_x = x[indices]
Test_y = y[indices] - 1

label_pl = np.load('final-9.npy')
label_pl = torch.from_numpy(label_pl) # 145*145*17
label_pl = label_pl.reshape(-1,1)
indices2 = np.nonzero(label_pl)
indices2 = indices2.numpy()[:,0]
Train_pl_x = x[indices2]
Train_pl_y = label_pl[indices2] - 1

Train_pl_x = Train_pl_x.reshape(-1, windowSize, windowSize, K, 1)
Train_pl_x = Train_pl_x.transpose(0, 4, 3, 1, 2)
Test_x = Test_x.reshape(-1, windowSize, windowSize, K, 1)
Test_x = Test_x.transpose(0, 4, 3, 1, 2)

# Data Augmentation
Train_pl_x_1,Train_pl_x_2 = four_rotation_2(Train_pl_x)
Train_pl_x = np.concatenate((Train_pl_x, Train_pl_x_1, Train_pl_x_2),axis=0)
Train_pl_y = np.concatenate((Train_pl_y, Train_pl_y, Train_pl_y),axis=0)


Test_x = torch.from_numpy(Test_x)
Test_y = torch.from_numpy(Test_y)
Train_pl_x = torch.from_numpy(Train_pl_x)
Train_pl_y = torch.from_numpy(Train_pl_y).squeeze(-1).squeeze(-1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class TrainDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Train_pl_x.shape[0]
        self.Train_x = torch.FloatTensor(Train_pl_x.float())
        self.Train_y = torch.FloatTensor(Train_pl_y.float())
    def __getitem__(self, index):
        return self.Train_x[index], self.Train_y[index]

    def __len__(self):
        return self.len
trainset = TrainDS()
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

class TestDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Test_x.shape[0]
        self.Test_x = torch.FloatTensor(Test_x.float())
        self.Test_y = torch.FloatTensor(Test_y.float())
    def __getitem__(self, index):
        return self.Test_x[index], self.Test_y[index]

    def __len__(self):
        return self.len
testset = TestDS()
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=32, shuffle=False, num_workers=0)

t = 0
to_acc = np.zeros((2,1))
acc_max = 0
while t < 2:
    net = HybridSN1().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    Loss_c = nn.CrossEntropyLoss().to(device)
    net.train()
    print(net)

    for epoch in range(50):
        total_loss = 0
        total_acc = 0
        for i, (X, Y) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            outputs = net(X)
            loss = Loss_c(outputs, Y.long())
            loss.backward(retain_graph=True)
            optimizer.step()
            pre = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            acc = np.sum(pre == Y.detach().cpu().numpy()) / 32
            total_loss += loss.item()
            total_acc += acc.item()

        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]  [acc: %.4f]'
               % (epoch + 1, total_loss / (len(train_loader)), loss.item(), total_acc / (len(train_loader))))
    print('Finished Training')
    net.eval()
    print(net)

    count = 0

    for i, (X, Y) in enumerate(test_loader):
        X = X.to(device)
        outputs = net(X)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        outputs = torch.from_numpy(outputs)
        if count == 0:
            y_pred_test = outputs
            count = 1
        else:
            y_pred_test = torch.cat((y_pred_test, outputs),dim = 0)
        print(y_pred_test.shape)

    acc = (np.sum(Test_y.detach().cpu().numpy() == y_pred_test.detach().cpu().numpy())) / len(y_pred_test)

    if acc > acc_max:
        torch.save(net, 'net.pth')
        acc_max = acc
    classification = classification_report(Test_y, y_pred_test, digits=4)
    print(classification, acc)
    to_acc[t] = acc
    t += 1
    print(to_acc)
    print(acc_max)