import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DataProcessor:
    def __init__(self):
        self.mean = None
        self.std = None

    def time_to_periodic_features(self, time_diff, max_time):
        sin_time = torch.sin(2 * torch.pi * time_diff / max_time)
        cos_time = torch.cos(2 * torch.pi * time_diff / max_time)
        return torch.stack([sin_time, cos_time], dim=-1)

    def normalize(self, data, mean=None, std=None):
        if mean is None or std is None:
            mean = torch.mean(data, dim=0)
            std = torch.std(data, dim=0)
            self.mean = mean
            self.std = std
        normalized_data = (data - mean) / (std + 1e-5)
        return normalized_data

    def process_data(self, sensor_data, real_data):
        # 提取时间戳
        sensor_time = sensor_data[:, 0]
        real_time = real_data[:, 0]

        # 计算时间差
        sensor_time_diff = sensor_time - sensor_time[0]
        real_time_diff = real_time - real_time[0]

        # 获取最大时间差
        max_time = max(sensor_time_diff[-1], real_time_diff[-1])

        # 转换时间戳为周期性特征
        sensor_time_features = self.time_to_periodic_features(sensor_time_diff, max_time)
        real_time_features = self.time_to_periodic_features(real_time_diff, max_time)

        # 合并时间特征与空间特征
        sensor_features = torch.cat([sensor_time_features, sensor_data[:, 1:]], dim=-1)
        real_features = torch.cat([real_time_features, real_data[:, 1:]], dim=-1)

        # 归一化处理
        sensor_features_normalized = self.normalize(sensor_features)
        real_features_normalized = self.normalize(real_features, self.mean, self.std)

        return sensor_features_normalized, real_features_normalized


class MyDataSet(Dataset):
    def __init__(self, sensor_data, real_data, sequence_length):
        self.sequence_length = sequence_length
        self.processor = DataProcessor()

        # 处理数据
        self.sensor_data, self.real_data = self.processor.process_data(sensor_data, real_data)

        # 构造序列
        self.sensor_sequences = self._create_sequences(self.sensor_data)
        self.real_sequences = self._create_sequences(self.real_data, offset=1)

        # 截断数据集，确保两者长度一致
        min_length = min(len(self.sensor_sequences), len(self.real_sequences))
        self.sensor_sequences = self.sensor_sequences[:min_length]
        self.real_sequences = self.real_sequences[:min_length]

    def _create_sequences(self, data, offset=0):
        """将数据划分为连续的子序列"""
        sequences = []
        for i in range(len(data) - self.sequence_length - offset + 1):
            sequences.append(data[i:i + self.sequence_length + offset])
        return torch.stack(sequences)

    def __len__(self):
        return len(self.sensor_sequences)

    def __getitem__(self, idx):
        input_sequence = self.sensor_sequences[idx]  # 输入序列
        target_sequence = self.real_sequences[idx][-1:]  # 目标值
        return {"input": input_sequence, "target": target_sequence}


def load_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            values = list(map(float, line.strip().split()))
            data.append(values)
    data = np.array(data)
    data = torch.tensor(data, dtype=torch.float32)
    return data


# 主程序
if __name__ == "__main__":
    real_pos_path = "./data_set/real_pos.txt"
    usbl_pos_path = "./data_set/usbl_pos_modified.txt"

    # 加载数据
    real_data = load_data_from_txt(real_pos_path)
    sensor_data = load_data_from_txt(usbl_pos_path)

    # 序列长度和批次大小
    sequence_length = 10
    batch_size = 16

    # 创建数据集和数据加载器
    dataset = MyDataSet(sensor_data=sensor_data, real_data=real_data, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # 测试数据集
    for step, batch in enumerate(dataloader):
        inputs, targets = batch["input"], batch["target"]
        print(f"Step: {step}")
        print("输入特征形状:", inputs.shape)  # (batch_size, sequence_length, feature_dim)
        print("目标值形状:", targets.shape)  # (batch_size, 1, feature_dim)