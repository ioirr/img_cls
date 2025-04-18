import os
import math
import argparse
import sys
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from dataset import DataProcessor, MyDataSet, load_data_from_txt
from model_t import Transformer_nav as create_model
from torch.utils.data import random_split


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    real_pos_path = "./data_set/real_pos.txt"
    usbl_pos_path = "./data_set/usbl_pos_modified.txt"

    # 加载数据
    real_data = load_data_from_txt(real_pos_path)
    sensor_data = load_data_from_txt(usbl_pos_path)

    # 序列长度和批次大小
    sequence_length = args.sequence_length
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # 创建数据集和数据加载器
    dataset = MyDataSet(sensor_data=sensor_data, real_data=real_data, sequence_length=sequence_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=args.shuffle, num_workers=nw)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    for batch in val_dataloader:
        print(type(batch))  # 应该是 dict 或其他可索引对象
        print(batch.keys())  # 应该包含 "input" 和 "target"
        break
    model = create_model("USBL").to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    loss_function = torch.nn.MSELoss()

    save_path = './weights/Transformer_nav.pth'

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_dataloader, file=sys.stdout)
        for step, batch in enumerate(train_bar):
            inputs, targets = batch["input"], batch["target"]
            optimizer.zero_grad()
            outputs = model(inputs.to(device), mask=None)
            last_step_outputs = outputs[:, -1:]
            loss = loss_function(last_step_outputs, targets.to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, args.epochs, running_loss/(step + 1))

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, file=sys.stdout)
            for step, val_batch in enumerate(val_bar):
                val_inputs, val_targets = val_batch["input"], val_batch["target"]
                val_outputs = model(val_inputs.to(device), mask=None)
                last_val_outputs = val_outputs[:, -1:]
                # 记录预测轨迹

                loss = loss_function(last_val_outputs, val_targets.to(device))
                val_loss += loss.item()
                val_bar.desc = "valid epoch[{}/{}] val_loss:{:.3f}".format(epoch + 1,
                                                                           args.epochs, val_loss/(step + 1))

        if epoch == 30:
            torch.save(model.state_dict(), save_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--sequence-length', type=float, default=10)


    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,
    #                     default="./data_set")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth',
    #                     help='initial weights path')
    # 是否冻结权重
    # parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)