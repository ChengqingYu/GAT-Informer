import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import random
from metric.mask_metric import masked_mae,masked_mape,masked_rmse,masked_mse
from block.informer_arch import Informer, InformerStack
from block.GAT import GraphAttentionLayer
from block.cross import cross_att

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

seed = 3407
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def Inverse_normalization(x,max,min):
    return x * (max - min) + min


### PEMS-BAY、METR-LA、PeMS04、PeMS08、AQI
data_name = 'PeMS04'
data_file = "../../data/" + data_name + "/data12.npz"

raw_data = np.load(data_file,allow_pickle=True)
print(raw_data.files)

batch_size = 32
epoch = 51
IF_mask = 0
DATASET_INPUT_LEN = 12 # history length
DATASET_OUTPUT_LEN = 12 # future length
### 超参数
IF_STACK = False   # Whether to use InformerStack
num_layer = 2      # Number of layers of GAT
NUM_NODES = 307    # num nodes
enc_in = NUM_NODES
dec_in = NUM_NODES
c_out  = NUM_NODES
seq_len = DATASET_INPUT_LEN  # input sequence length
label_len = DATASET_INPUT_LEN // 2  # start token length used in decoder
out_len = DATASET_OUTPUT_LEN  # prediction sequence length
factor = 3  # probsparse attn factor
d_model = 32
n_heads = 4
if IF_STACK:
    e_layers = [4, 2, 1] # for InformerStack
else:
    e_layers = 2  # num of encoder layers
d_layers = 1  # num of decoder layers
d_ff = 32
dropout = 0.15
attn = 'prob'  # attention used in encoder, options:[prob, full]
embed = "timeF"  # [timeF, fixed, learned]
activation= "gelu"
output_attention = False
distil = True  # whether to use distilling in encoder, using this argument means not using distilling
mix = True  # use mix attention in generative decoder
num_time_features  = 2  # number of used time features [time_of_day, day_of_week, day_of_month, day_of_year]
time_of_day_size   = 288
day_of_week_size   =  7
day_of_month_size  = 31
day_of_year_size   = 366
IF_cross = True       ### Whether to use cross attention to merge GAT with Informer

###调整学习率和训练梯度
lr_rate = 0.002       ### learn rate
weight_decay = 0.0005 ### weight decay
max_norm = 0          ### Gradient pruning
max_num =  100        ### 和max_norm配套的东西
IF_mask_rate = True   ### PEMS08以及METR-LA是True
num_lr = 5            ###多少步验证集误差没有下降就调整学习率
gamme = 0.5
milestone = [1,4,10,15,30,50,70,90] ### milestone

###CPU和GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cpu")

### 获取训练集数据
if IF_mask == 0.25:
    train_x = raw_data["train_x_mask_25"]
elif IF_mask == 0.5:
    train_x = raw_data["train_x_mask_50"]
elif IF_mask == 0.75:
    train_x = raw_data["train_x_mask_75"]
else:
    train_x = raw_data["train_x_raw"]

train_y = raw_data["train_y"]

graph_data = torch.tensor(raw_data["graph"]).to(torch.float32)


###输入，输出长度，时间序列数量
input_len = train_x.shape[-1]
output_len = train_y.shape[-1]
num_id = train_x.shape[-2]

train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)

train_data = torch.cat([train_x,train_y],dim=2).to(torch.float32)
train_data = DataLoader(train_data,batch_size=batch_size,shuffle=False)
#print(train_data.shape)

### 获取验证集数据
if IF_mask == 0.25:
    valid_x = raw_data["vail_x_mask_25"]
elif IF_mask == 0.5:
    valid_x = raw_data["vail_x_mask_50"]
elif IF_mask == 0.75:
    valid_x = raw_data["vail_x_mask_75"]
else:
    valid_x = raw_data["vail_x_raw"]

valid_y = raw_data["vail_y"]
valid_x = torch.tensor(valid_x).to(torch.float32)
valid_y = torch.tensor(valid_y).to(torch.float32)
valid_data = torch.cat([valid_x,valid_y],dim=2).to(torch.float32)
valid_data = DataLoader(valid_data,batch_size=batch_size,shuffle=False)

### 测试集
if IF_mask == 0.25:
    test_x = raw_data["test_x_mask_25"]
elif IF_mask == 0.5:
    test_x = raw_data["test_x_mask_50"]
elif IF_mask == 0.75:
    test_x = raw_data["test_x_mask_75"]
else:
    test_x = raw_data["test_x_raw"]

test_y = raw_data["test_y"]
test_x = torch.tensor(test_x)
test_y = torch.tensor(test_y)
test_data = torch.cat([test_x, test_y],dim=2).to(torch.float32)
test_data = DataLoader(test_data,batch_size=batch_size,shuffle=False)

###
max_min = raw_data['max_min']
max_data, min_data = max_min[0],max_min[1]

if IF_mask == 0.25:
    mask_id = raw_data["mask_id_25"]
elif IF_mask == 0.5:
    mask_id = raw_data["mask_id_50"]
elif IF_mask == 0.75:
    mask_id = raw_data["mask_id_75"]
else:
    print("没有任何MASK")

#print(mask_id)
### 数据处理完了，开始建模

class GATINFORMER(nn.Module):
    def __init__(self, IF_STACK,num_layer, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 time_of_day_size, day_of_week_size, day_of_month_size,day_of_year_size,
                 factor, d_model, n_heads, e_layers, d_layers, d_ff,dropout, attn, embed, freq, activation,
                 output_attention, distil, mix, num_time_features,IF_cross):
        super(GATINFORMER, self).__init__()
        ### Parameter
        self.IF_STACK = IF_STACK
        self.num_layer = num_layer
        self.lay_norm = nn.LayerNorm([out_len])
        self.IF_cross = IF_cross

        ###GAT
        self.GAT1 = GraphAttentionLayer(seq_len, out_len, dropout)
        self.GAT2 = GraphAttentionLayer(out_len, out_len, dropout)

        ###Informer
        if self.IF_STACK:
            self.Informer =InformerStack(enc_in, dec_in, c_out, seq_len, label_len, out_len,
                time_of_day_size, day_of_week_size, day_of_month_size,day_of_year_size,
                factor, d_model, n_heads, e_layers, d_layers, d_ff,
                dropout=dropout, attn=attn, embed=embed, freq=freq, activation=activation,
                output_attention=output_attention, distil=distil, mix=mix, num_time_features=num_time_features)
        else:
            self.Informer = Informer(enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 time_of_day_size, day_of_week_size, day_of_month_size=day_of_month_size,
                 day_of_year_size=day_of_year_size,
                 factor=factor, d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_layers=d_layers, d_ff=d_ff,
                 dropout=dropout, attn=attn, embed=embed, freq=freq, activation=activation,
                 output_attention=output_attention, distil=distil, mix=mix, num_time_features=num_time_features)

        ### decoder
        self.cross = cross_att(out_len,n_heads,dropout)
        self.decoder = nn.Conv1d(in_channels=out_len,out_channels=out_len,kernel_size=1)

    def forward(self, x, y, graph_data, device):

        ###图数据处理
        graph_data = graph_data.to(device)
        graph_data = GATINFORMER.calculate_laplacian_with_self_loop(graph_data)

        ###基于GAT的空间建模
        for i in range(self.num_layer):
            if i == 0:
                prediction_GAT = F.gelu(self.GAT1(x,graph_data))
            else:
                prediction_GAT = F.gelu(self.GAT2(prediction_GAT, graph_data))

        ### 基于Informer的时间建模
        prediction_In = self.Informer(x, y)

        ###特征融合
        if self.IF_cross:
            # 方案1：采用cross_attention的思想
            x = self.cross(prediction_In, prediction_GAT).transpose(-2, -1)
        else:
            #方案2：直接相加并正则化
            x = prediction_GAT + prediction_In
            x = self.lay_norm(x).transpose(-2, -1)

        ###生成最终预测结果
        x = self.decoder(x).transpose(-2, -1)
        return x

    @staticmethod
    def calculate_laplacian_with_self_loop(matrix):
        row_sum = matrix.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_laplacian = (
            matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
        )
        return normalized_laplacian

###模型设置
my_net = GATINFORMER(IF_STACK,num_layer,enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 time_of_day_size, day_of_week_size, day_of_month_size=day_of_month_size,
                 day_of_year_size=day_of_year_size,
                 factor=factor, d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_layers=d_layers, d_ff=d_ff,
                 dropout=dropout, attn=attn, embed=embed, freq='h', activation=activation,
                 output_attention=output_attention, distil=distil, mix=mix, num_time_features=num_time_features,IF_cross=IF_cross)

my_net = my_net.to(device)
optimizer = optim.Adam(params=my_net.parameters(),lr=lr_rate,weight_decay=weight_decay)
# optimizer = optim.Adam(params=my_net.parameters(),lr=lr_rate)
num_vail = 0
min_vaild_loss = float("inf")

###开始训练
for i in range(epoch):
    num = 0
    loss_out = 0.0
    my_net.train()
    for data in train_data:
        my_net.zero_grad()
        ###训练集情况
        train_feature = data[:,:,:input_len].to(device)
        train_target = data[:,:,input_len:].to(device)
        train_pre = my_net(train_feature,train_target,graph_data, device)
        if IF_mask_rate:
            loss_data = masked_mae(train_pre,train_target,0.0)
        else:
            loss_data = masked_mae(train_pre, train_target)
        ###误差反向传播
        loss_data.backward()

        if max_norm > 0 and i < max_num:
            nn.utils.clip_grad_norm_(my_net.parameters(), max_norm = max_norm)
        else:
            pass

        num += 1
        optimizer.step()
        loss_out += loss_data
    loss_out = loss_out/num

    ###验证集情况
    num_va = 0
    loss_vaild = 0.0
    my_net.eval()
    with torch.no_grad():
        for data in test_data:
            ###验证集情况
            valid_x = data[:, :, :input_len].to(device)
            valid_y = data[:, :, input_len:].to(device)
            valid_pre = my_net(valid_x,valid_y,graph_data, device)
            if IF_mask_rate:
                loss_data = masked_mae(valid_pre, valid_y,0.0)
            else:
                loss_data = masked_mae(valid_pre, valid_y)

            num_va += 1
            loss_vaild += loss_data
        loss_vaild = loss_vaild / num_va

    """
    ###学习率调整策略
    if loss_vaild < min_vaild_loss:
        num_vail = 0
        min_vaild_loss = loss_vaild
    else:
        num_vail +=1
    ###满足条件时调整学习率
    if num_vail >= num_lr:
        num_vail = 0
        for params in optimizer.param_groups:
            # 遍历Optimizer中的每一组参数，将该组参数的学习率 * 0.5
            params['lr'] *= gamme
    """
    if (i + 1) in milestone:
        for params in optimizer.param_groups:
            # 遍历Optimizer中的每一组参数，将该组参数的学习率 * 0.5
            params['lr'] *= gamme

    print('训练集第{}个epoch的训练集Loss: {:02.4f},验证集Loss:{:02.4f}:'.format(i+1,loss_out,loss_vaild))


my_net.eval()
my_net = my_net.to(device2)
with torch.no_grad():
    all_pre = 0.0
    all_true = 0.0
    num = 0
    for data in test_data:
        test_feature = data[:,:,:input_len].to(device2)
        test_target = data[:,:,input_len:].to(device2)
        test_pre = my_net(test_feature,test_target,graph_data, device2)
        if num == 0:
            all_pre = test_pre
            all_true = test_target
        else:
            all_pre = torch.cat([all_pre, test_pre], dim=0)
            all_true = torch.cat([all_true, test_target], dim=0)
        num += 1

test_x = Inverse_normalization(test_x, max_data, min_data)
final_pred = Inverse_normalization(all_pre, max_data, min_data)
final_target = Inverse_normalization(all_true, max_data, min_data)


if IF_mask_rate:
###met系列
    mae,mape,rmse = masked_mae(final_pred, final_target,0.0),\
                    masked_mape(final_pred, final_target,0.0)*100,masked_rmse(final_pred, final_target,0.0)
    print('测试集完整的\nRMSE: {}, MAPE: {}, MAE: {}'.format(rmse,mape,mae))

    mae2,mape2,rmse2 = masked_mae(final_pred[:,:,2], final_target[:,:,2],0.0),\
                       masked_mape(final_pred[:,:,2], final_target[:,:,2],0.0)*100,masked_rmse(final_pred[:,:,2], final_target[:,:,2],0.0)
    print('测试集3步的\nRMSE: {}, MAPE: {}, MAE: {}'.format(rmse2,mape2,mae2))

    mae2,mape2,rmse2 = masked_mae(final_pred[:,:,5], final_target[:,:,5],0.0),\
                       masked_mape(final_pred[:,:,5], final_target[:,:,5],0.0)*100,masked_rmse(final_pred[:,:,5], final_target[:,:,5],0.0)
    print('测试集6步的\nRMSE: {}, MAPE: {}, MAE: {}'.format(rmse2,mape2,mae2))

    mae2,mape2,rmse2 = masked_mae(final_pred[:,:,-1], final_target[:,:,-1],0.0),\
                       masked_mape(final_pred[:,:,-1], final_target[:,:,-1],0.0)*100,masked_rmse(final_pred[:,:,-1], final_target[:,:,-1],0.0)
    print('测试集12步的\nRMSE: {}, MAPE: {}, MAE: {}'.format(rmse2,mape2,mae2))

else:
    ###pem系列
    mae,mape,rmse = masked_mae(final_pred, final_target),\
                    masked_mape(final_pred, final_target,0.0)*100,masked_rmse(final_pred, final_target)
    print('测试集完整的\nRMSE: {}, MAPE: {}, MAE: {}'.format(rmse,mape,mae))

    mae2,mape2,rmse2 = masked_mae(final_pred[:,:,2], final_target[:,:,2]),\
                       masked_mape(final_pred[:,:,2], final_target[:,:,2],0.0)*100,masked_rmse(final_pred[:,:,2], final_target[:,:,2])
    print('测试集3步的\nRMSE: {}, MAPE: {}, MAE: {}'.format(rmse2,mape2,mae2))

    mae2,mape2,rmse2 = masked_mae(final_pred[:,:,5], final_target[:,:,5]),\
                       masked_mape(final_pred[:,:,5], final_target[:,:,5],0.0)*100,masked_rmse(final_pred[:,:,5], final_target[:,:,5])
    print('测试集6步的\nRMSE: {}, MAPE: {}, MAE: {}'.format(rmse2,mape2,mae2))

    mae2,mape2,rmse2 = masked_mae(final_pred[:,:,-1], final_target[:,:,-1]),\
                       masked_mape(final_pred[:,:,-1], final_target[:,:,-1],0.0)*100,masked_rmse(final_pred[:,:,-1], final_target[:,:,-1])
    print('测试集12步的\nRMSE: {}, MAPE: {}, MAE: {}'.format(rmse2,mape2,mae2))

