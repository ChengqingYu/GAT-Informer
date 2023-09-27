import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.fft as fft


class cross_att(nn.Module):
    def __init__(self, dim_input,num_head,dropout):
        super(cross_att, self).__init__()
        self.query = nn.Linear(dim_input, dim_input)
        self.key = nn.Linear(dim_input, dim_input)
        self.value = nn.Linear(dim_input, dim_input)
        self.laynorm = nn.LayerNorm([dim_input])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.linear1 = nn.Linear(num_head, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x,x2):
        q = self.dropout(self.query(x2))
        k = self.dropout(self.key(x))
        k = k.transpose(-2, -1)
        v = self.dropout(self.value(x))
        kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32)/self.num_head)
        result = 0.0
        for i in range(self.num_head):
            line = self.dropout(self.softmax(q@k/kd)) @ v
            line = line.unsqueeze(-1)
            if i < 1:
                result = line
            else:
                result = torch.cat([result,line],dim=-1)
        result = self.linear1(result)
        result = result.squeeze(-1) + x
        x = self.laynorm(result)
        return x