import torch
import torch.nn as nn
import torch.nn.functional as F



class TransformerEncoderLayerPreNorm(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        # d_model：输入和输出特征的维度（即词嵌入的维度）。
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)

        self.activation = nn.ReLU(inplace=True)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        # 归一化操作，有助于加速训练收敛、改善梯度传播，以及提升模型性能
        src = self.norm1(src)  # [K,B*N,C]
        src2, mask = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)  # 一种常用的正则化技术，用于在训练过程中随机地将神经网络的某些神经元输出设置为零（dropout）或按比例缩放，以减少过拟合（overfitting）的风险
        
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        
        return src