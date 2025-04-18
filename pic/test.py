# # !/usr/bin/python3
# # -*- encoding: utf-8 -*-
#
import matplotlib.pyplot as plt
import numpy as np
import tushare as ts
import pandas as pd
import torch
from torch import nn
import datetime
import time

DAYS_FOR_TRAIN = 10


class LSTM_Regression(nn.Module):
    """
        使用LSTM进行回归
        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x


def create_dataset(data, days_for_train=5) -> (np.array, np.array):
    """
        根据给定的序列data，生成数据集
        数据集分为输入和输出，每一个输入的长度为days_for_train，每一个输出的长度为1。
        也就是说用days_for_train天的数据，对应下一天的数据。
        若给定序列的长度为d，将输出长度为(d-days_for_train+1)个输入/输出对
    """
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train):
        _x = data[i:(i + days_for_train)]
        dataset_x.append(_x)
        dataset_y.append(data[i + days_for_train])
    return (np.array(dataset_x), np.array(dataset_y))


if __name__ == '__main__':
    t0 = time.time()
    data_close = ts.get_k_data('000001', start='2019-01-01', index=True)['close']  # 取上证指数的收盘价
    data_close.to_csv('000001.csv', index=False)  # 将下载的数据转存为.csv格式保存
    data_close = pd.read_csv('000001.csv')  # 读取文件

    df_sh = ts.get_k_data('sh', start='2019-01-01', end=datetime.datetime.now().strftime('%Y-%m-%d'))
    print(df_sh.shape)

    data_close = data_close.astype('float32').values  # 转换数据类型
    plt.plot(data_close)
    plt.savefig('data.png', format='png', dpi=200)
    plt.close()

    # 将价格标准化到0~1
    max_value = np.max(data_close)
    min_value = np.min(data_close)
    data_close = (data_close - min_value) / (max_value - min_value)

    # dataset_x
    # 是形状为(样本数, 时间窗口大小)
    # 的二维数组，用于训练模型的输入
    # dataset_y
    # 是形状为(样本数, )
    # 的一维数组，用于训练模型的输出。
    dataset_x, dataset_y = create_dataset(data_close, DAYS_FOR_TRAIN)  # 分别是（1007,10,1）（1007,1）

    # 划分训练集和测试集，70%作为训练集
    train_size = int(len(dataset_x) * 0.7)

    train_x = dataset_x[:train_size]
    train_y = dataset_y[:train_size]

    # 将数据改变形状，RNN 读入的数据维度是 (seq_size, batch_size, feature_size)
    train_x = train_x.reshape(-1, 1, DAYS_FOR_TRAIN)
    train_y = train_y.reshape(-1, 1, 1)

    # 转为pytorch的tensor对象
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)

    model = LSTM_Regression(DAYS_FOR_TRAIN, 8, output_size=1, num_layers=2)  # 导入模型并设置模型的参数输入输出层、隐藏层等

    model_total = sum([param.nelement() for param in model.parameters()])  # 计算模型参数
    print("Number of model_total parameter: %.8fM" % (model_total / 1e6))

    train_loss = []
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    for i in range(200):
        out = model(train_x)
        loss = loss_function(out, train_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(loss.item())

        # 将训练过程的损失值写入文档保存，并在终端打印出来
        with open('log.txt', 'a+') as f:
            f.write('{} - {}\n'.format(i + 1, loss.item()))
        if (i + 1) % 1 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(i + 1, loss.item()))

    # 画loss曲线
    plt.figure()
    plt.plot(train_loss, 'b', label='loss')
    plt.title("Train_Loss_Curve")
    plt.ylabel('train_loss')
    plt.xlabel('epoch_num')
    plt.savefig('loss.png', format='png', dpi=200)
    plt.close()

    # torch.save(model.state_dict(), 'model_params.pkl')  # 可以保存模型的参数供未来使用
    t1 = time.time()
    T = t1 - t0
    print('The training time took %.2f' % (T / 60) + ' mins.')

    tt0 = time.asctime(time.localtime(t0))
    tt1 = time.asctime(time.localtime(t1))
    print('The starting time was ', tt0)
    print('The finishing time was ', tt1)

    # for test
    model = model.eval()  # 转换成评估模式
    # model.load_state_dict(torch.load('model_params.pkl'))  # 读取参数

    # 注意这里用的是全集 模型的输出长度会比原数据少DAYS_FOR_TRAIN 填充使长度相等再作图
    dataset_x = dataset_x.reshape(-1, 1, DAYS_FOR_TRAIN)  # (seq_size, batch_size, feature_size)
    dataset_x = torch.from_numpy(dataset_x)

    pred_test = model(dataset_x)  # 全量训练集
    # 的模型输出 (seq_size, batch_size, output_size)
    pred_test = pred_test.view(-1).data.numpy()
    pred_test = np.concatenate((np.zeros(DAYS_FOR_TRAIN), pred_test))  # 填充0 使长度相同
    assert len(pred_test) == len(data_close)

    plt.plot(pred_test, 'r', label='prediction')
    plt.plot(data_close, 'b', label='real')
    plt.plot((train_size, train_size), (0, 1), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出
    plt.legend(loc='best')
    plt.savefig('result.png', format='png', dpi=200)
    plt.close()

# import numpy as np
# import torch
# from torch import nn
# import matplotlib
# matplotlib.use("Qt5Agg")
# import matplotlib.pyplot as plt
#
# # 参数设置
# DAYS_FOR_TRAIN = 20
# HIDDEN_SIZE = 32
# NUM_LAYERS = 2
# EPOCHS = 100
# LR = 5e-3
#
#
# class LSTM_Regression(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size=3, num_layers=2):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out)
#         return out
#
#
# def manual_minmax_scale(data):
#     min_vals = np.min(data, axis=0)
#     max_vals = np.max(data, axis=0)
#     return (data - min_vals) / (max_vals - min_vals), min_vals, max_vals
#
#
# def manual_minmax_inverse(data, min_vals, max_vals):
#     return data * (max_vals - min_vals) + min_vals
#
#
# def create_dataset(data, n_past):
#     X, Y = [], []
#     for i in range(len(data) - n_past):
#         X.append(data[i:i + n_past])
#         Y.append(data[i + n_past])
#     return np.array(X), np.array(Y)
#
#
# def load_usbl_data(file_path):
#     data = np.loadtxt(file_path)
#     timestamps = data[:, 0]
#     coords = data[:, 1:]
#     return timestamps, coords
#
#
# if __name__ == '__main__':
#     # 1. 加载数据
#     file_path = "usbl_pos.txt"
#     timestamps, coords = load_usbl_data(file_path)
#
#     # 2. 手动归一化
#     coords_normalized, min_vals, max_vals = manual_minmax_scale(coords)
#
#     # 3. 创建数据集
#     X, Y = create_dataset(coords_normalized, DAYS_FOR_TRAIN)
#     train_size = int(len(X) * 0.8)
#     train_x, test_x = X[:train_size], X[train_size:]
#     train_y, test_y = Y[:train_size], Y[train_size:]
#
#     # 转为Tensor
#     train_x = torch.FloatTensor(train_x)
#     train_y = torch.FloatTensor(train_y)
#     test_x = torch.FloatTensor(test_x)
#     test_y = torch.FloatTensor(test_y)
#
#     # 4. 训练模型
#     model = LSTM_Regression(input_size=3, hidden_size=HIDDEN_SIZE)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#
#     for epoch in range(EPOCHS):
#         outputs = model(train_x)
#         loss = criterion(outputs[:, -1, :], train_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if (epoch + 1) % 10 == 0:
#             print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.6f}')
#
#     # 5. 测试与可视化
#     model.eval()
#     with torch.no_grad():
#         test_predict = model(test_x)[:, -1, :]
#
#     test_predict = manual_minmax_inverse(test_predict.numpy(), min_vals, max_vals)
#     test_y = manual_minmax_inverse(test_y.numpy(), min_vals, max_vals)
#
#     # 创建包含三个子图的画布
#     fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
#     axes = axes.flatten()
#     titles = ['X-axis Position', 'Y-axis Position', 'Z-axis Position']
#
#     for i in range(3):
#         axes[i].plot(timestamps[DAYS_FOR_TRAIN + train_size:],
#                      test_y[:, i],
#                      'b-', label='Ground Truth')
#         axes[i].plot(timestamps[DAYS_FOR_TRAIN + train_size:],
#                      test_predict[:, i],
#                      'r--', label='Prediction')
#         axes[i].axvline(x=timestamps[train_size + DAYS_FOR_TRAIN],
#                         color='g', linestyle='--', label='Train/Test Split')
#         axes[i].set_title(titles[i])
#         axes[i].set_ylabel('Position (m)')
#         axes[i].grid(True)
#         axes[i].legend()
#
#     axes[-1].set_xlabel('Time (s)')
#     plt.tight_layout()
#     plt.show()
#
#     # 计算并打印误差
#     mse = np.mean((test_predict - test_y) ** 2, axis=0)
#     mae = np.mean(np.abs(test_predict - test_y), axis=0)
#     print('\nPerformance Metrics:')
#     print(f'{"Axis":<5} | {"MSE":<10} | {"MAE":<10}')
#     print('-' * 30)
#     for i, axis in enumerate(['X', 'Y', 'Z']):
#         print(f'{axis:<5} | {mse[i]:<10.6f} | {mae[i]:<10.6f}')
