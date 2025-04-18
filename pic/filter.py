# import numpy as np
# from scipy.signal import savgol_filter
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("Qt5Agg")
#
#
# def read_usbl_data(path, noise_std=0.3, spike_prob=0.05):
#     """从文本文件读取USBL数据（格式：timestamp x y z）"""
#     data = np.loadtxt(file_path)
#     timestamps = data[:, 0]  # 时间戳（可选，用于绘图横轴）
#     true_coords = data[:, 1:]  # XYZ坐标
#     # true_coords[:, 0] /= 50  # X 坐标缩小 10 倍
#     # true_coords[:, 1] /= 50  # Y 坐标缩小 10 倍
#
#     if path == file_path:
#         noisy_coords = true_coords + np.random.normal(0, noise_std, true_coords.shape)
#         # 添加脉冲噪声（突发噪声）
#         for i in range(3):  # 对X/Y/Z分别处理
#             spikes = np.random.choice([0, 1],
#                                       size=true_coords.shape[0],
#                                       p=[1 - spike_prob, spike_prob])
#             noisy_coords[:, i] += spikes * np.random.uniform(-2, 2, true_coords.shape[0])
#         return timestamps, noisy_coords
#     elif path == file_path1:
#         return timestamps, true_coords
#
#
# def usbl_sg_filter(data, window_length=5, polyorder=2):
#     """Savitzky-Golay滤波（直接处理原始数据）"""
#     filtered = np.zeros_like(data)
#     for i in range(3):  # 分别处理X/Y/Z
#         filtered[:, i] = savgol_filter(
#             data[:, i],
#             window_length,
#             polyorder,
#             mode='nearest'
#         )
#     return filtered
#
#
# def plot_comparison(timestamps, gt_data, raw_data, filtered_data, dim=0, title='X-axis Position'):
#     """绘制原始数据与滤波结果对比"""
#     plt.figure(figsize=(12, 6))
#     plt.plot(timestamps, raw_data[:, dim], 'r.', alpha=0.5, label='Raw USBL')
#     plt.plot(timestamps, filtered_data[:, dim], 'b-', linewidth=2, label='SG Filtered')
#     plt.plot(timestamps, gt_data[:, dim], 'g-', linewidth=2, label='GT')
#     plt.title(f"{title} (Window={window_size})")
#     plt.xlabel('Time (s)')
#     plt.ylabel('Position (m)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
# def calculate_errors(f_data, g_data):
#     """
#     计算滤波后数据与真实数据的误差
#     返回:
#         dict: 各坐标轴的MSE、MAE、MaxAE
#     """
#     errors = {
#         'X': {}, 'Y': {}, 'Z': {}
#     }
#
#     for i, axis in enumerate(['X', 'Y', 'Z']):
#         # 提取当前轴的数据
#         filtered = f_data[:, i]
#         gt = g_data[:, i]
#
#         # 均方误差 (MSE)
#         errors[axis]['MSE'] = np.mean((filtered - gt) ** 2)
#
#         # # 平均绝对误差 (MAE)
#         # errors[axis]['MAE'] = np.mean(np.abs(filtered - gt))
#         #
#         # # 最大绝对误差 (MaxAE)
#         # errors[axis]['MaxAE'] = np.max(np.abs(filtered - gt))
#
#     return errors
# # 主程序
# if __name__ == "__main__":
#     # 1. 读取数据文件
#     file_path = "usbl_pos.txt"  # 修改为你的文件路径
#     file_path1 = "real_pos.txt"
#     timestamps, raw_data = read_usbl_data(file_path)
#     _, gt_data = read_usbl_data(file_path1)
#
#     # 2. 应用滤波器（调整窗口大小和多项式阶数）
#     window_size = 21  # 建议根据数据频率调整（奇数）
#     polyorder = 5  # 多项式阶数
#     filtered_data = usbl_sg_filter(raw_data, window_length=window_size, polyorder=polyorder)
#     errors = calculate_errors(filtered_data, gt_data)
#     for axis in ['X', 'Y', 'Z']:
#         print(f"{axis}-Axis Errors:")
#         print(f"  MSE: {errors[axis]['MSE']:.6f}")
#
#     # 3. 可视化结果
#     plot_comparison(timestamps, gt_data, raw_data, filtered_data, dim=0, title='X-axis')
#     plot_comparison(timestamps, gt_data, raw_data, filtered_data, dim=1, title='Y-axis')
#     plot_comparison(timestamps, gt_data, raw_data, filtered_data, dim=2, title='Z-axis')

# 定义输入和输出文件路径
# 文件路径
# 定义输入和输出文件路径
input_file = "usbl_pos.txt"
output_file = "usbl_pos_modified.txt"

# 打开输入文件并读取内容
with open(input_file, "r") as file:
    lines = file.readlines()

# 处理每一行数据
modified_lines = []
for line in lines:
    # 去除行末的换行符并按空格分割
    parts = line.strip().split()

    # 确保当前行至少有 4 列
    if len(parts) >= 4:
        try:
            # 提取第 4 列的值并转换为浮点数
            col4_value = float(parts[3])

            # 减去 400 并保留小数点后 6 位
            modified_col4 = f"{col4_value - 400:.6f}"

            # 替换第 4 列的值
            parts[3] = modified_col4

            # 将处理后的行重新组合为字符串
            modified_line = " ".join(parts)
            modified_lines.append(modified_line)
        except ValueError:
            # 如果第 4 列无法转换为浮点数，跳过该行并打印警告
            print(f"Warning: Unable to process line: {line.strip()}")

# 将处理后的数据写入输出文件
with open(output_file, "w") as file:
    file.write("\n".join(modified_lines))

print(f"处理完成！修改后的数据已保存到 {output_file}")