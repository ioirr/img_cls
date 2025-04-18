import os
import torch.nn as nn
import math
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from dataset import DataProcessor, MyDataSet, load_data_from_txt
from model_t import Transformer_nav as create_model
from torch.utils.data import random_split
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# 定义空列表
gt = []
pred = []
test_loss = 0.0
i = 0

def plot_trajectory_and_coordinates(pred, gt):

    """
    封装的功能函数，用于绘制三维轨迹对比图和 X、Y、Z 坐标随时间变化的曲线。

    参数:
        pred (numpy.ndarray): 预测值，形状为 [num_samples, num_features]。
        gt (numpy.ndarray): 真实值，形状为 [num_samples, num_features]。
    """
    # 检查输入形状是否正确
    assert pred.shape == gt.shape, "预测值和真实值的形状必须相同！"
    assert pred.shape[1] >= 3, "数据至少需要包含 3 列（XYZ 坐标）！"

    # 提取 XYZ 坐标（后三列）
    pred_xyz = pred[:, -3:]
    gt_xyz = gt[:, -3:]

    # 时间轴
    time_steps = np.arange(pred.shape[0])

    # ------------------------------
    # 1. 绘制三维轨迹图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制真实轨迹 (ground truth)
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label="Ground Truth", color="blue")
    # 绘制预测轨迹 (prediction)
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], label="Prediction", color="red", linestyle="--")

    # 设置标题和标签
    ax.set_title("3D Trajectory Comparison")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.legend()

    # 显示三维轨迹图
    plt.show()

    # ------------------------------
    # 2. 绘制 X、Y、Z 随时间变化的图像
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # X 轴随时间变化
    axes[0].plot(time_steps, gt_xyz[:, 0], label="Ground Truth", color="blue")
    axes[0].plot(time_steps, pred_xyz[:, 0], label="Prediction", color="red", linestyle="--")
    axes[0].set_title("X Coordinate Over Time")
    axes[0].legend()

    # Y 轴随时间变化
    axes[1].plot(time_steps, gt_xyz[:, 1], label="Ground Truth", color="blue")
    axes[1].plot(time_steps, pred_xyz[:, 1], label="Prediction", color="red", linestyle="--")
    axes[1].set_title("Y Coordinate Over Time")
    axes[1].legend()

    # Z 轴随时间变化
    axes[2].plot(time_steps, gt_xyz[:, 2], label="Ground Truth", color="blue")
    axes[2].plot(time_steps, pred_xyz[:, 2], label="Prediction", color="red", linestyle="--")
    axes[2].set_title("Z Coordinate Over Time")
    axes[2].set_xlabel("Time Steps")
    axes[2].legend()

    # 调整布局并显示图像
    plt.tight_layout()
    plt.show()


def main():

    global gt, pred, test_loss, i
    # use gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create model
    model = create_model("USBL").to(device)

    # load model weights
    weights_path = "./weights/Transformer_nav.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 加载数据
    real_pos_path = "./data_set/real_pos.txt"
    usbl_pos_path = "./data_set/usbl_pos_modified.txt"


    real_data = load_data_from_txt(real_pos_path)
    sensor_data = load_data_from_txt(usbl_pos_path)
    dataset = MyDataSet(sensor_data=sensor_data, real_data=real_data, sequence_length=10)
    test_dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)


    # 定义误差
    loss_function = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            test_inputs, test_targets = batch["input"], batch["target"]
            test_outputs = model(test_inputs.to(device), mask=None)
            last_test_outputs = test_outputs[:, -1:]
            loss = loss_function(last_test_outputs, test_targets.to(device))
            test_loss += loss.item()

            test_targets = test_targets.squeeze(dim=1)
            last_test_outputs = last_test_outputs.squeeze(dim=1)
            gt.append(test_targets.cpu().numpy())
            pred.append(last_test_outputs.cpu().numpy())
            i += 1

    pred = np.concatenate(pred, axis=0)
    gt = np.concatenate(gt, axis=0)
    print(pred.shape)
    print(gt.shape)
    avg_mse = test_loss / i
    print(i)
    print(avg_mse)




if __name__ == '__main__':
    main()
    plot_trajectory_and_coordinates(pred, gt)

