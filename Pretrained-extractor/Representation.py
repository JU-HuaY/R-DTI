import torch
import torch.nn as nn
from spd_gnn import SPD_GNN
from MultiHeadAttention import MultiHeadAttention
import torch.nn.functional as F
from Infoloss import InfoNCELoss
from spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified # 4D

class EncoderLayer(nn.Module):
    def __init__(self, i_channel, o_channel, growth_rate, groups, pad2=7):
        super(EncoderLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=i_channel, out_channels=o_channel, kernel_size=(2 * pad2 + 1), stride=1,
                               groups=groups, padding=pad2,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(i_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=o_channel, out_channels=growth_rate, kernel_size=(2 * pad2 + 1), stride=1,
                               groups=groups, padding=pad2,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(o_channel)
        self.drop_rate = 0.1

    def forward(self, x):
        # xn = self.bn1(x)
        xn = self.relu(x)
        xn = self.conv1(xn)
        xn = self.bn2(xn)
        xn = self.relu(xn)
        xn = self.conv2(xn)

        return torch.cat([x, xn], 1)


class Decoder(nn.Module):
    def __init__(self, inc, dimension, outc, head, layers):
        super(Decoder, self).__init__()
        self.layers = layers
        self.act = nn.ReLU(inplace=True)
        self.linear_in = nn.Linear(inc, dimension, bias=False)
        self.LN_in = nn.LayerNorm(dimension)
        self.Attention = MultiHeadAttention(h=head, d_model=dimension)
        self.linear = nn.ModuleList([nn.Linear(dimension, dimension) for _ in range(layers)])
        self.LN_fea = nn.LayerNorm(dimension)
        self.layers = layers
        self.linear_out = nn.Linear(dimension, outc)


    def forward(self, x):
        x = self.act(self.LN_in(self.linear_in(x)))
        x = self.Attention(x, x, x)
        for i in range(self.layers):
            x = self.act(self.linear[i](x))
        x = self.linear_out(x)
        return x


class language_encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=15, pad2=7):
        super(language_encoder, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv1d(in_channels=inc, out_channels=inc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1,
                                 bias=False)
        self.dense_cnn = nn.ModuleList(
            [EncoderLayer(inc + growth_rate * i_la, inc + (growth_rate // 2) * i_la, growth_rate, groups, pad2) for i_la
             in
             range(layers)])
        self.conv_out = nn.Conv1d(in_channels=inc + growth_rate * layers, out_channels=outc, kernel_size=(pad1 * 2 + 1),
                                  stride=1,
                                  padding=pad1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.layers):
            x = self.dense_cnn[i](x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        return x

class Covariance2(nn.Module):

    def __init__(self, append_mean=False, epsilon=1e-3):
        super(Covariance2, self).__init__()
        self.append_mean = append_mean
        self.epsilon = epsilon  # 小的常数用于添加到协方差矩阵的对角线上

    def forward(self, input):

        mean = torch.mean(input, 1, keepdim=True)
        x = input - mean.expand(-1,  input.size(1), input.size(2))
        output = torch.matmul(x, x.transpose(1, 2)) / input.size(1)
        # 生成协方差矩阵的对角线正则化
        I = torch.eye(output.size(2)).expand_as(output).to(input.device)
        output += I * self.epsilon
        return output
class Covariance(nn.Module):

    def __init__(self, append_mean=False, epsilon=1e-3):
        super(Covariance, self).__init__()
        self.append_mean = append_mean
        self.epsilon = epsilon  # 小的常数用于添加到协方差矩阵的对角线上

    def forward(self, input):

        mean = torch.mean(input, 2, keepdim=True)
        x = input - mean.expand(-1, -1, input.size(2), input.size(3))
        output = torch.matmul(x, x.transpose(2, 3)) / input.size(2)

        # 生成协方差矩阵的对角线正则化
        I = torch.eye(output.size(2)).expand_as(output).to(input.device)
        output += I * self.epsilon

        if self.append_mean:
            mean_sq = torch.matmul(mean, mean.transpose(2, 3))
            output.add_(mean_sq)
            output = torch.cat((output, mean), 3)
            one = input.new(1, 1, 1, 1).fill_(1).expand(mean.size(0), mean.size(1), -1, -1)
            mean = torch.cat((mean, one), 1).transpose(2, 3)
            output = torch.cat((output, mean), 2)

        return output

def add_epsilon(output, epsilon):
    I = torch.eye(output.size(1)).expand_as(output).cpu()
    output += I * epsilon
    return output

import numpy as np

class SPDVectorize(nn.Module):

    def __init__(self, input_size):
        super(SPDVectorize, self).__init__()
        row_idx, col_idx = np.triu_indices(input_size)
        self.register_buffer('row_idx', torch.LongTensor(row_idx))
        self.register_buffer('col_idx', torch.LongTensor(col_idx))

    def forward(self, input):
        output = input[:, :, self.row_idx, self.col_idx]
        return output

class SPD(nn.Module):
    def __init__(self, size):
        super(SPD, self).__init__()
        self.rect = SPDRectified().cpu()

    def forward(self, h_co):
        h_co = h_co.cpu()
        h_vec_co_Riemannian = self.rect(h_co)
        B, L = h_vec_co_Riemannian.shape[0], h_vec_co_Riemannian.shape[1]
        h_hidden = h_vec_co_Riemannian#.view(B, L, 256)
        return h_hidden

def cosine_similarity_loss(x, x_hat):
    dot_product = torch.sum(x * x_hat, dim=2)
    norm_x = torch.norm(x, dim=2)
    norm_x_hat = torch.norm(x_hat, dim=2)
    cos_sim = dot_product / (norm_x * norm_x_hat + 1e-8)  # 添加极小值防止除以零
    loss = 1 - cos_sim  # 转换为距离形式，越接近1损失越小
    return torch.mean(loss)

def kl_divergence_logits(p_logits, q_logits):
    p_probs = nn.functional.softmax(p_logits, dim=-1)
    q_probs = nn.functional.softmax(q_logits, dim=-1)
    kl_loss = p_probs * (torch.log(p_probs + 1e-6) - torch.log(q_probs + 1e-6))  # 添加极小值避免 log(0)
    kl_loss = torch.mean(kl_loss)  # 对所有样本求平均
    return kl_loss

class Representation_model(nn.Module):
    def __init__(self, num_layers, hidden_dim, out_dim, edge_feat_dim, batch):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.covariance = Covariance()
        self.covariance2 = Covariance2()
        self.edge_feat_dim = edge_feat_dim
        self.represent = nn.Embedding(25, hidden_dim)
        self.SPD_GNN = SPD_GNN(num_layers, hidden_dim, edge_feat_dim)
        self.ST_trans = SPDTransform(hidden_dim, 16)
        self.ReEig = SPD(16)

        self.LM_encoder = language_encoder(1024, hidden_dim, hidden_dim, 3, groups=64, pad1=7, pad2=3)
        self.LM_trans = SPDTransform(hidden_dim, 16)
        self.LM_SPD = SPD(16)
        self.LM_decoder = Decoder(inc=out_dim, dimension=out_dim, outc=3, head=2, layers=2)
        self.batch = batch

    def forward(self, h_vec, x_pos, h_LM, prot_batchs, B, N):
        h_vec = self.represent(h_vec.view(B * N))
        x_pos = x_pos.view(B * N, 3)
        prot_batchs = prot_batchs.view(B * N)
        h_vec_co = self.covariance2(h_vec.unsqueeze(2))
        h_vec_co = self.ST_trans(h_vec_co.unsqueeze(1)).squeeze(1)
        h_vec_co, x_pos, outputs = self.SPD_GNN.forward(h_vec_co, x_pos, prot_batchs)
        h_vec_co = h_vec_co.view(B, N, 16, 16)
        h_vec_hidden_cpu = self.ReEig(h_vec_co)
        h_vec_hidden = h_vec_hidden_cpu.to("cuda")

        h_lm = self.LM_encoder.forward(h_LM.permute(0, 2, 1)).permute(0, 2, 1)
        h_lm_co = self.covariance(h_lm.unsqueeze(3))
        h_lm_co = self.LM_trans(h_lm_co)
        h_lm_hidden_cpu = self.LM_SPD(h_lm_co)
        h_lm_hidden = h_lm_hidden_cpu.to("cuda")
        h_lm_out_pos = self.LM_decoder(h_lm_hidden.view(B, N, 256))
        return h_lm_out_pos, h_vec_hidden.view(B, N, 256), h_lm_hidden.view(B, N, 256)

    def LM_generate(self, h_LM): # language model hide feature generation
        h_lm = self.LM_encoder.forward(h_LM.permute(0, 2, 1)).permute(0, 2, 1).detach()
        h_lm_co = self.covariance(h_lm.unsqueeze(3)).detach()
        h_lm_co = self.LM_trans(h_lm_co).detach()
        h_lm_hidden = self.LM_SPD(h_lm_co).detach()
        mask = torch.triu(torch.ones_like(h_lm_hidden[0][0]), diagonal=1).bool()
        h_hiddens = h_lm_hidden[:, :, mask]
        return h_hiddens

    def __call__(self, data, device, train=True):
        res_seqs, res_cooss, prot_features, prot_batchs, B, N = data[0], data[1], data[2], data[3], data[4], data[5]
        # B_data = int(B/2) # DataParallel
        h_lm_out_pos, h_vec_hidden, h_lm_hidden \
            = self.forward(res_seqs.to(device), res_cooss.to(device), prot_features.to(device), prot_batchs.to(device), int(B), int(N))
        SL1_loss = nn.SmoothL1Loss()
        contrastive_loss = InfoNCELoss(temperature=0.1, normalize=True)

        if train:
            loss1, scores = contrastive_loss(h_vec_hidden, h_lm_hidden)
            loss2 = SL1_loss(h_lm_out_pos, res_cooss.to(device))
            loss_all = loss2 + loss1 * 0.1

            return loss_all, loss1, loss2
        else:
            loss1, scores = contrastive_loss(h_vec_hidden, h_lm_hidden)
            loss2 = SL1_loss(h_lm_out_pos, res_cooss)
            loss_all = loss2 + loss1 * 0.1 #loss1 * 10 + loss2 * 0.01 + loss3 * 0.01
            return loss_all, loss1, loss2

