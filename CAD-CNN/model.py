from torch import nn
from torch.nn import Conv1d, MaxPool1d, Linear, LeakyReLU, Softmax, CrossEntropyLoss, Flatten, BatchNorm1d
from sklearn.metrics import accuracy_score, recall_score
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
import torch

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.norm = BatchNorm1d(num_features=514)
        self.conv1 = Conv1d(in_channels=1, out_channels=5, kernel_size=27, stride=1)
        self.pool = MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = Conv1d(in_channels=5, out_channels=10, kernel_size=15, stride=1)
        self.conv3= Conv1d(in_channels=10, out_channels=10, kernel_size=4, stride=1)
        self.conv4= Conv1d(in_channels=10, out_channels=10, kernel_size=3, stride=1)
        self.flatten = Flatten()
        self.l1 = Linear(in_features=270, out_features=30)
        self.l2 = Linear(in_features=30, out_features=10)
        self.l3 = Linear(in_features=10, out_features=2)
        self.loss = CrossEntropyLoss()
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self,x):
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x = F.leaky_relu_(self.conv1(x))
        x = self.pool(x)
        x = F.leaky_relu_(self.conv2(x))
        x = self.pool(x)
        x = F.leaky_relu_(self.conv3(x))
        x = self.pool(x)
        x = F.leaky_relu_(self.conv4(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.leaky_relu_(self.l1(x))
        x = F.leaky_relu_(self.l2(x))
        x = self.l3(x)
        y = F.softmax(x, dim=1)
        return y

    @staticmethod
    def eva(model, dataloader, use_gpu=False):
        model.eval()
        y_pres = np.array([])
        y_labels = np.array([])
        for data in dataloader:
            x, y = data
            if use_gpu:
                y = y.cpu().numpy()
                y_pre = model(x.cuda()).detach().cpu().numpy()
            else:
                y = y.numpy()
                y_pre = model(x).detach().numpy()
            y_pre = y_pre.argmax(1)
            y_pres = np.append(y_pres, y_pre)
            y_labels = np.append(y_labels, y)
        # print(np.sum(y_pres))
        # print(np.sum(y_labels))
        accuracy = accuracy_score(y_labels, y_pres)
        sensitivity = recall_score(y_labels, y_pres)
        return accuracy, sensitivity



if __name__ == '__main__':
    x = np.random.random((10, 1, 514))
    model = Net1()
    print(model(torch.tensor(x, dtype=torch.float)))
    # summary(model, (1, 514))
