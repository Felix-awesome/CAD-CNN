import wfdb
import numpy as np
from glob import glob
import os
import pywt
from scipy.interpolate import interp1d

source_dir = "./source_data"
source_cad_dir = source_dir + "/CAD"
source_normal_dir = source_dir + "/Normal"
data_dir = "./data"
CAD_dir = data_dir + "/CAD"
Normal_dir = data_dir + "/Normal"

def interp(data, size):
    """
    上采样
    :param data: 源数据
    :param size: 需要上采样的大小
    :return: 插值后的数据
    """
    x = np.arange(1, len(data)+1)
    xnew = np.linspace(1, len(data), len(data)+size)
    f = interp1d(x, data, kind='quadratic')
    data_int = f(xnew)
    assert len(data_int) == len(data) + size
    return data_int

def dwt(data):
    """
    小波变换
    :return:
    """
    w = pywt.Wavelet('db6')  # 选用Daubechies6小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    # print("maximum level is " + str(maxlev))
    threshold = 0.04  # Threshold for filtering

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, 'db6', level=maxlev)  # 将信号进行小波分解


    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波

    datarec = pywt.waverec(coeffs, 'db6')[:-1]  # 将信号进行小波重构
    # print(data)
    # print(datarec)
    assert len(data) == len(datarec)
    return datarec

# 读取编号为data的一条心电数据
def read_ecg_data(path, channel):
    '''
    读取心电信号文件
    sampfrom: 设置读取心电信号的起始位置，sampfrom=0表示从0开始读取，默认从0开始
    sampto：设置读取心电信号的结束位置，sampto = 1500表示从1500出结束，默认读到文件末尾
    channel_names：设置设置读取心电信号名字，必须是列表，channel_names=['MLII']表示读取MLII导联线
    channels：设置读取第几个心电信号，必须是列表，channels=[0, 3]表示读取第0和第3个信号，注意信号数不确定
    '''
    # 读取所有导联的信号
    record = wfdb.rdrecord(path, sampfrom=0,  channel_names=channel)
    # 仅仅读取“MLII”导联的信号
    # record = wfdb.rdrecord('../ecg_data/' + data, sampfrom=0, sampto=1500, channel_names=['MLII'])
    # 仅仅读取第0个信号（MLII）
    # record = wfdb.rdrecord('../ecg_data/' + data, sampfrom=0, sampto=1500, channels=[0])

    # 查看record类型
    # print(type(record))
    # 查看类中的方法和属性
    # print(dir(record))

    # 获得心电导联线信号，本文获得是MLII和V1信号数据
    # print(record.p_signal)
    # print(np.shape(record.p_signal))
    # 查看导联线信号长度，本文信号长度1500
    # print(record.sig_len)
    # 查看文件名
    # print(record.record_name)
    # 查看导联线条数，本文为导联线条数2
    # print(record.n_sig)
    # 查看信号名称（列表），本文导联线名称['MLII', 'V1']
    # print(record.sig_name)
    # 查看采样率
    # print(record.fs)

    '''
    读取注解文件
    sampfrom: 设置读取心电信号的起始位置，sampfrom=0表示从0开始读取，默认从0开始
    sampto：设置读取心电信号的结束位置，sampto=1500表示从1500出结束，默认读到文件末尾
    '''
    # annotation = wfdb.rdann('./I01', 'atr')
    # # 查看annotation类型
    # print(type(annotation))
    # # 查看类中的方法和属性
    # print(dir(annotation))
    #
    # # 标注每一个心拍的R波的尖锋位置的信号点，与心电信号对应
    # print(annotation.sample)
    # # 标注每一个心拍的类型N，L，R等等
    # print(annotation.symbol)
    # # 被标注的数量
    # print(annotation.ann_len)
    # # 被标注的文件名
    # print(annotation.record_name)
    # # 查看心拍的类型
    # print(wfdb.show_ann_labels())

    # 画出数据
    # draw_ecg(record.p_signal)
	# 返回一个numpy二维数组类型的心电信号，shape=(65000,1)
    return record.p_signal

def preprocess():
    # 处理Normal数据集对齐采样率到257，并进行离散小波变换去噪
    sample_rate = 250
    upsample_size = 7
    with open(os.path.join(source_normal_dir, "RECORDS")) as f:
        files = f.read().strip().split('\n')
    # print(files)
    for file in files:
        ecg = np.array(read_ecg_data(os.path.join(source_normal_dir, file), channel=["ECG"]))
        ecg = ecg.squeeze(axis=1)
        line = []
        for i, e in enumerate(ecg):
            line.append(e)
            if (i+1) % sample_rate == 0:
                newecg_line = dwt(interp(line, upsample_size))
                assert len(newecg_line) == sample_rate + upsample_size
                with open(os.path.join(Normal_dir, file+".txt"), 'a') as f:
                    f.write(' '.join([str(num) for num in newecg_line]))
                    f.write('\n')
                line.clear()

    # 对CAD数据集进行离散小波变换去噪
    for file in glob(source_cad_dir + "/*/*.dat"):
        ecg = np.array(read_ecg_data(file[:-4], channel=["II"]))
        ecg = ecg.squeeze(axis=1)
        line = []
        for i, e in enumerate(ecg):
            line.append(e)
            if (i+1) % (sample_rate+upsample_size) == 0:
                newecg_line = dwt(line)
                assert len(newecg_line) == sample_rate + upsample_size
                with open(os.path.join(CAD_dir, file.split('/')[-1][:-4]+".txt"), 'a') as f:
                    f.write(' '.join([str(num) for num in newecg_line]))
                    f.write('\n')
                line.clear()

def split_data(path, seconds):
    """
    把数据按秒分割
    :param seconds: 秒数
    :param path: 需要分割的数据路径
    :return:
    """
    p = os.path.join(path, str(seconds)+'s')
    if not os.path.exists(p):
        os.mkdir(p)
    for file in glob(path + "/*.txt"):
        with open(file, 'r') as f:
            data = f.read().strip().split('\n')
        name = file.split('/')[-1]
        with open(os.path.join(p, name), 'w') as f:
            line = []
            for i, d in enumerate(data):
                line.append(d)
                if (i+1) % seconds == 0:
                    f.write(' '.join(line))
                    f.write('\n')
                    line.clear()

if __name__ == "__main__":
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(CAD_dir):
        os.mkdir(CAD_dir)
    if not os.path.exists(Normal_dir):
        os.mkdir(Normal_dir)

    preprocess()
    split_data(CAD_dir, 2)
    split_data(CAD_dir, 5)
    split_data(Normal_dir, 2)
    split_data(Normal_dir, 5)

