import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoder(nn.Module):

    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        # 创建一个常量 PE 矩阵
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000**((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000**((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        # 使得单词嵌入表示相对大一些
        x = x * math.sqrt(self.d_model)
        # 增加位置常量到单词嵌入表示中
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]

        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # d_ff 默认设置为 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class NormLayer(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        # 层归一化包含两个可以学习的参数
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm



class Encoder(nn.Module):

    def __init__(self, d_model, N, heads, dropout, max_seq_len):
        super().__init__()
        self.N = N
        # self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, dropout) for _ in range(N)])
        self.norm = NormLayer(d_model)
        self.fc = nn.Linear(d_model, 5)
        self.input_proj = nn.Linear(5, d_model)  # 线性映射层

    def forward(self, src, mask):
        # x = self.embed(src)
        x = self.input_proj(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.fc(self.norm(x))
        return x



class EncoderLayer(nn.Module):

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


def Transformer_nav(aims: str):
    if aims == "USBL":
        model = Encoder(d_model=800,
                        N=6,
                        heads=8,
                        dropout=0.1,
                        max_seq_len=10)
        return model
    else:
        print("model doesn't relate")

if __name__ == "__main__":
    # 超参数
    vocab_size = 1496  # 词汇表大小
    d_model = 800  # 嵌入维度
    n_layers = 6  # 编码器层数
    heads = 8  # 注意力头数
    d_ff = 2048  # 前馈网络隐藏层维度
    max_seq_len = 10  # 最大序列长度
    dropout = 0.1  # Dropout 概率

    # 创建模型
    model = Encoder(d_model, n_layers, heads, dropout, max_seq_len)
    model1 = Transformer_nav("USBL")

    src = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],  # 第一个序列
        [10, 11, 12, 13, 14, 0, 0, 0, 0, 0]  # 第二个序列
    ])  # Shape: (batch_size=2, seq_len=10)

    output = model1(src, mask=None)
    print(output.shape)  # 输出形状: (batch_size, seq_len, d_model)