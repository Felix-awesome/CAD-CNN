from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class MyDataset(Dataset):
    """
        DataSet
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x_idx = self.x[index]
        # 原文说对每个片段归一化，实际实验效果不如加归一化层好
        # mu = np.mean(x_idx, axis=1)
        # sigma = np.var(x_idx, axis=1)
        # x_idx = (x_idx - mu) / sigma
        # assert x_idx.shape == self.x[index].shape
        return torch.tensor(x_idx, dtype=torch.float), torch.tensor(self.y[index], dtype=torch.long)

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    source_dir = "./source_data"
    source_cad_dir = source_dir + "/CAD"
    source_normal_dir = source_dir + "/Normal"
    data_dir = "./data"
    CAD_dir = data_dir + "/CAD"
    Normal_dir = data_dir + "/Normal"
    # dataset = MyDataset()
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # for d in dataloader:
    #     print(d)
    #     break
