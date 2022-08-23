import numpy as np
import struct
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import gzip
import os
import torchvision
import matplotlib.pyplot as plt
import random

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# 读取图片
def read_image(file_name):
    # 先用二进制方式把文件都读进来
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中
    offset = 0
    head = struct.unpack_from('>IIII', file_content, offset)  # 取前4个整数，返回一个元组
    offset += struct.calcsize('>IIII')
    imgNum = head[1]  # 图片数
    rows = head[2]  # 宽度
    cols = head[3]  # 高度

    images = np.empty((imgNum, 784))  # empty，是它所常见的数组内的所有元素均为空，没有实际意义，它是创建数组最快的方法
    image_size = rows * cols  # 单个图片的大小
    fmt = '>' + str(image_size) + 'B'  # 单个图片的format

    for i in range(imgNum):
        images[i] = np.array(struct.unpack_from(fmt, file_content, offset))
        offset += struct.calcsize(fmt)
    return images


# 读取标签
def read_label(file_name):
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中

    head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
    offset = struct.calcsize('>II')

    labelNum = head[1]  # label数
    bitsString = '>' + str(labelNum) + 'B'  # fmt格式：'>47040000B'
    label = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    return np.array(label)


def loadDataSet():
    train_x_filename = "train-images-idx3-ubyte"
    train_y_filename = "train-labels-idx1-ubyte"
    test_x_filename = "t10k-images-idx3-ubyte"
    test_y_filename = "t10k-labels-idx1-ubyte"
    train_data = read_image(train_x_filename)
    train_labels = read_label(train_y_filename)
    test_data = read_image(test_x_filename)
    test_labels = read_label(test_y_filename)

    return train_data, test_data, train_labels, test_labels


# 筛选小批量数据集
def Data_Iter(Data, Label, Batch_Size):
    index = list(range(len(Data)))
    random.shuffle(index)
    j = index[0:Batch_Size]
    return torch.tensor(Data[j], dtype=torch.float, device=device), torch.tensor(Label[j], device=device)


# 定义softmax分类函数
def SoftMax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdims=True)
    return X_exp / partition


# 定义样本标签
def Target_Function(num_outputs, batch_size, Label):
    Y = torch.zeros(batch_size, num_outputs, dtype=torch.float, device=device)
    for i in range(batch_size):
        Y[i, Label[i]] = 1
    return Y


# 定义交叉熵损失函数
def cross_entropy(Y, Y_hat, Batch_Size):
    return -(Y * Y_hat.log()).sum() / Batch_Size


def show(error, iteration):
    x = range(0, iteration, 1)
    plt.xlabel('Number of iterations')
    plt.ylabel('error rate')
    plt.plot(x, error, color='k')
    plt.show()
    # plt.imsave('./iteration.png')


def predict(test_data, test_labels, w, b, batch_size):
    x, y = Data_Iter(test_data, test_labels, batch_size)
    y_hat = SoftMax(torch.mm(x, w) + b).to(device)
    y_hat_max_position = y_hat.argmax(axis=1)
    error_count = 0
    len_y = len(y)
    for i in range(len_y):
        if y_hat_max_position[i] != y[i]:
            error_count += 1
    error_rate = error_count / len(y)
    return error_rate





# 读取数据
train_data, test_data, train_labels, test_labels = loadDataSet()
test_data = torch.tensor(test_data, dtype=torch.float, device=device)
test_labels = torch.tensor(test_labels, dtype=torch.float, device=device)
# 初始化训练参数
batch_size = 1000
num_inputs = 784
num_outputs = 10
lr = 0.0001
num_epochs = 3000
error = []
w = torch.tensor(np.random.normal(0, 0.01, ((num_inputs, num_outputs))), dtype=torch.float, requires_grad=True, device=device)
b = torch.tensor(np.zeros((batch_size, 1)), dtype=torch.float, requires_grad=True, device=device)
# 开始训练
for epoch in range(num_epochs):
    # lr = pow(10, -3) * pow(0.87, epoch) * 5
    data, label = Data_Iter(train_data, train_labels, batch_size)
        # train_loss, train_acc, n = 0.0, 0.0, 0
    # 归一化图像数据
    x = (data.reshape(batch_size, -1) / 255).to(device)
    # 计算样本估计值
    y_hat = SoftMax(torch.mm(x, w) + b).to(device)
    # 实际样本标签
    y = Target_Function(num_outputs, batch_size, label)
    # 计算损失函数
    loss = cross_entropy(y, y_hat, batch_size)
    # 求梯度
    loss.backward()
    # 更新训练参数
    w.data -= lr * w.grad
    b.data -= lr * b.grad
    # 梯度清零
    w.grad.data.zero_().to(device)
    b.grad.data.zero_().to(device)

    error_rate = predict(test_data, test_labels, w, b, batch_size)
    error.append(error_rate)
    print(epoch, error_rate)

show(error, num_epochs)





