from glob import glob
import numpy as np
import os
from model import Net1
from torch.utils.data import DataLoader
import torch
import logging
from dataloader import MyDataset

model_dir = "./model"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

def deal_data(data, seconds):
    result = []
    sample_rate = 257
    for d in data:
        r = d.strip().split(' ')
        if len(r) == sample_rate * seconds:
            if "nan" not in r:
                result.append(r)
    return np.expand_dims(np.array(result, dtype=float), axis=1)


def get_data(cad_dir, normal_dir, seconds=2):
    normal_size = 80000
    if(seconds == 2):
        normal_size = 80000
    elif(seconds == 5):
        normal_size = 32000
    cad_data = []
    normal_data = []
    for file in glob(os.path.join(cad_dir, str(seconds) + 's', "*.txt")):
        with open(file, 'r') as f:
            cad_data.extend(f.read().strip().split('\n'))
    for file in glob(os.path.join(normal_dir, str(seconds) + 's', "*.txt")):
        with open(file, 'r') as f:
            normal_data.extend(f.read().strip().split('\n'))
    cad = deal_data(cad_data, seconds)
    # print(len(cad))
    y1 = np.ones((len(cad)))
    normal = deal_data(normal_data, seconds)[:normal_size]
    y0 = np.zeros((len(normal)))
    len_cad = len(cad)
    len_normal = len(normal)
    # 划分cad数据集
    train_cad = cad[:int(len_cad*0.9*0.7)]
    valid_cad = cad[int(len_cad*0.9*0.7):int(len_cad*0.9)]
    test_cad = cad[int(len_cad*0.9):]
    # 划分normal数据集
    train_normal = normal[:int(len_normal * 0.9 * 0.7)]
    valid_normal = normal[int(len_normal * 0.9 * 0.7):int(len_normal * 0.9)]
    test_normal = normal[int(len_normal * 0.9):]

    train_x = np.concatenate((train_cad, train_normal), axis=0)
    valid_x = np.concatenate((valid_cad, valid_normal), axis=0)
    test_x = np.concatenate((test_cad, test_normal), axis=0)

    # 生成标签
    train_y = np.concatenate((np.ones(len(train_cad)), np.zeros(len(train_normal))))
    valid_y = np.concatenate((np.ones(len(valid_cad)), np.zeros(len(valid_normal))))
    test_y = np.concatenate((np.ones(len(test_cad)), np.zeros(len(test_normal))))

    assert len(train_y) == len(train_x)
    assert len(valid_y) == len(valid_x)
    assert len(test_y) == len(test_x)
    logging.info("训练集大小：%d" %len(train_y))
    logging.info("验证集大小：%d" %len(valid_y))
    logging.info("测试集大小：%d" %len(test_y))
    return train_x, train_y, valid_x, valid_y, test_x, test_y
    # x = np.concatenate((cad, normal), axis=0)
    # y = np.concatenate((y1, y0), axis=0)
    # length = len(x)
    # idx = np.arange(length)
    # np.random.shuffle(idx)
    # #返回训练集，验证集，测试集
    # return x[idx[:int(length*0.9*0.7)]], y[idx[:int(length*0.9*0.7)]], \
    #        x[idx[int(length*0.9*0.7):int(length*0.9)]], y[idx[int(length*0.9*0.7):int(length*0.9)]], \
    #        x[idx[int(length*0.9):]], y[idx[int(length*0.9):]]

def train(train_dataloader,valid_dataloader, test_dataloader):
    epochs = 20
    if torch.cuda.is_available():
        model = Net1().cuda()
    else:
        model = Net1()

    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0001, weight_decay=0.2, momentum=0.7)
    step = 1
    sen = 0.8
    acc = 0.9
    for i in range(epochs):
        for (x, y) in train_dataloader:
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            y_pred = model(x) / 0.001
            loss = model.loss(y_pred, y)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 1000 == 0:
                logging.info("epoch：%d 迭代步数：%d, 损失 %f" % (i, step, loss.item()))
                logging.info("在验证集上验证：")
                acc1, sen1 =  model.eva(model, valid_dataloader)
                logging.info("accuracy:%f" % (acc1))
                logging.info("sensitivity:%f" % (sen1))
                if sen1 > sen or acc1 > acc:
                    sen = sen1
                    acc = acc1
                    logging.info("在测试集上测试：")
                    acc2, sen2 = model.eva(model, test_dataloader)
                    logging.info("accuracy:%f" % (acc2))
                    logging.info("sensitivity:%f" % (sen2))
                    file_path = os.path.join(model_dir, str(step) + '.pkl')
                    logging.info("迭代次数：%d, 保存模型" % step)
                    torch.save(model.state_dict(), file_path)
            step = step + 1
    logging.info("总迭代次数:%d" % step)
    logging.info("在测试集上测试：")
    logging.info("accuracy:%f, sensitivity:%f" % model.eva(model, test_dataloader))
    file_path = os.path.join(model_dir, str(step) + '.pkl')
    torch.save(model.state_dict(), file_path)


def test():
    pass

if __name__ == '__main__':
    log_file = 'train.log'
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    data_dir = "./data"
    CAD_dir = data_dir + "/CAD"
    Normal_dir = data_dir + "/Normal"

    train_x, train_y, valid_x, valid_y, test_x, test_y = get_data(cad_dir=CAD_dir, normal_dir=Normal_dir, seconds=2)

    train_dataset = MyDataset(train_x, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    valid_dataset = MyDataset(valid_x, valid_y)
    valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=True)

    test_dataset = MyDataset(test_x, test_y)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    train(train_dataloader, valid_dataloader, test_dataloader)


