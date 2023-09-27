# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:53:01 2020

@author: PCCH
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903 
    图注意力层
    """
    def __init__(self, in_c, out_c, dropout):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c   # 节点表示向量的输入特征数
        self.out_c = out_c   # 节点表示向量的输出特征数
        self.dropout = dropout
        
        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_c, out_c)))
        nn.init.kaiming_uniform_(self.W)
        self.a = nn.Parameter(torch.zeros(size=(2*out_c, 1)))
        nn.init.kaiming_uniform_(self.a)   # 初始化
        
        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU() #当x<0,alpha*x
    
    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数 the input data, [B, N, C]
        adj: 图的邻接矩阵  [N, N] 非零即一，数据结构基本知识
        """
        B, N = inp.size(0), inp.size(1)
        h = torch.matmul(inp, self.W)   # [B,N,out_features] ,其中matmul保证维度不塌缩

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * self.out_c)
        # [B, N, N, 2 * out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        # [B,N, N, 1] => [B, N, N] 图注意力的相关系数（未归一化）
        
        zero_vec = -1e12 * torch.ones_like(e)    # 将没有连接的边置为负无穷,形状和e一致的1矩阵

        attention = torch.where(adj > 0, e, zero_vec)   # [B,N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=2)    # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)   # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [B,N, N].[N, out_features] => [B,N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示     
        return h_prime  
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT_model(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param out_c: int, number of output channels.
        :param K:
        """
        super(GAT_model, self).__init__()
        self.conv1 = GraphAttentionLayer(in_c, hid_c)
        self.conv2 = GraphAttentionLayer(hid_c, out_c)
        self.act = nn.ReLU()

    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]  # [N, N]
        flow_x = data["flow_x"].to(device)  # [B, N, H, D]
        B, N = flow_x.size(0), flow_x.size(1)
        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]
        output_1 = self.act(self.conv1(flow_x, graph_data))
        output_2 = self.act(self.conv2(output_1, graph_data))

        return output_2.unsqueeze(2)#[B,1,N,1]
