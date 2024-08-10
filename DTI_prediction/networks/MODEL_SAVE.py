import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from .Informer_block import AttentionLayer, ProbAttention
from .dt_spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified
from .rresnet import RResNet, ProjVecField
from .rresnet.manifolds import spd

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


class Encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=15, pad2=7):
        super(Encoder, self).__init__()
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


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.05):
        super(Decoder, self).__init__()
        self.atten0 = AttentionLayer(ProbAttention(None, 3, dropout), d_model,
                                     n_heads)  # SelfAttention(75, 5, 0.2)
        self.atten1 = AttentionLayer(ProbAttention(None, 3, dropout), d_model,
                                     n_heads)  # SelfAttention(75, 5, 0.2)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, xs, xd, xp):
        xp = self.drop(self.atten0(xp, xd, xd, None))
        xd = self.drop(self.atten1(xd, xp, xp, None))
        xs = xs + torch.cat((xd, xp), dim=1)
        return xs


class DT_LeNet(nn.Module):
    def __init__(self, hidden, classes, layers):
        super(DT_LeNet, self).__init__()
        self.CNNs = nn.ModuleList(
            [nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=7, padding=3) for _ in range(layers)])
        self.BN = nn.BatchNorm1d(hidden)  # nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])

        self.FC_combs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.FC_down = nn.Linear(hidden, 128)
        self.FC_out = nn.Linear(128, classes)
        self.layers = layers
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, dti_feature):
        dti_feature = dti_feature.permute(0, 2, 1)  # self.BN(dti_feature.permute(0, 2, 1))
        for i in range(self.layers):
            dti_feature = self.act(self.CNNs[i](dti_feature)) + dti_feature
        dti_feature = dti_feature.permute(0, 2, 1)
        dti_feature = torch.mean(dti_feature, dim=1)
        GRL_feature = dti_feature.clone()
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti_feature = self.FC_down(dti_feature)
        dti = self.FC_out(dti_feature)
        return dti, dti_feature, GRL_feature


class SPDVectorize(nn.Module):
    def __init__(self, input_size):
        super(SPDVectorize, self).__init__()
        row_idx, col_idx = np.triu_indices(input_size)
        self.register_buffer('row_idx', torch.LongTensor(row_idx))
        self.register_buffer('col_idx', torch.LongTensor(col_idx))

    def forward(self, input):
        output = input[:, self.row_idx, self.col_idx]
        return output

class SPD_LeNet(nn.Module):
    def __init__(self, hidden, classes, layers):
        super(SPD_LeNet, self).__init__()
        self.vec = SPDVectorize(hidden)
        self.FC_down = nn.Linear(int((hidden+1)*hidden/2), 1024)
        self.FC_combs = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(layers)])
        self.FC_out = nn.Linear(1024, classes)
        self.layers = layers
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, dti_spd):
        dti_vec = self.vec(dti_spd)  # self.BN(dti_feature.permute(0, 2, 1))
        dti_feature = self.FC_down(dti_vec)
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti = self.FC_out(dti_feature)
        return dti, dti_feature

class GRL(nn.Module):

    def __init__(self, max_iter):
        super(GRL, self).__init__()
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - 1)
        return -coeff * gradOutput


class Discriminator(nn.Module):
    def __init__(self, max_iter, source_number, device):
        super(Discriminator, self).__init__()
        self.fc1 = Parameter(torch.Tensor(256, 128))#nn.Linear(75, 128)
        self.fc2 = Parameter(torch.Tensor(75, 128))
        self.fc3 = Parameter(torch.Tensor(128, source_number))  # nn.Linear(128, source_number)
        self.drop_lm = nn.Dropout(0.0)
        self.drop_sty = nn.Dropout(0.0)
        self.drop = nn.Dropout(0.5)
        self.grl_layer = GRL(10000)
        self.grl_layer2 = GRL(100)
        self.source_number = source_number
        self.device = device

    def forward(self, feature1, feature2):
        if self.source_number > 2:
            adversarial_out1 = self.grl_layer(self.drop_lm(feature1.detach()))
            adversarial_out2 = self.grl_layer2(self.drop_sty(feature2))
            adversarial_out1 = torch.matmul(adversarial_out1,  nn.init.xavier_uniform_(self.fc1))
            adversarial_out2 = torch.matmul(adversarial_out2, nn.init.xavier_uniform_(self.fc2))
            adversarial_out = adversarial_out2 * torch.sigmoid(adversarial_out1)
            adversarial_out = torch.matmul(self.drop(torch.relu(adversarial_out)), nn.init.xavier_uniform_(self.fc3))
            # print(adversarial_out)
        else:
            adversarial_out = torch.zeros((feature1.shape[0], 1), device=self.device)
        return adversarial_out


class Contrast_Fusion(nn.Module):
    def __init__(self, dropout=0.05):
        super(Contrast_Fusion, self).__init__()
        self.so_L = nn.Sigmoid()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, LM_fea, Sty_fea):
        LM_att = self.so_L(LM_fea)
        fus_fea = LM_fea + Sty_fea * LM_att
        return fus_fea

class ContrastLoss(nn.Module):
    def __init__(self, source_number):
        super(ContrastLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.source_number = source_number
        pass

    def forward(self, anchor_fea, reassembly_fea, contrast_label):
        if self.source_number > 2:
            contrast_label = contrast_label.float()
            anchor_fea = anchor_fea.detach()
            loss = -(F.cosine_similarity(anchor_fea, reassembly_fea, dim=-1))
            loss = loss * contrast_label
        else:
            loss = 0 * contrast_label
        return loss.mean()

class Smooth_loss(nn.Module):

    def __init__(self, smoothing=0.1):
        super(Smooth_loss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        confidence = 1 - self.smoothing
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Covariance(nn.Module):

    def __init__(self, append_mean=False, epsilon=1e-5):
        super(Covariance, self).__init__()
        self.append_mean = append_mean
        self.epsilon = epsilon  # 小的常数用于添加到协方差矩阵的对角线上

    def forward(self, input):
        mean = torch.mean(input, 2, keepdim=True)
        x = input - mean.expand(-1, -1, input.size(2))
        output = torch.bmm(x, x.transpose(1, 2)) / input.size(1)

        if self.append_mean:
            mean_sq = torch.bmm(mean, mean.transpose(1, 2))
            output.add_(mean_sq)
            output = torch.cat((output, mean), 2)
            one = input.new(1, 1, 1).fill_(1).expand(mean.size(0), -1, -1)
            mean = torch.cat((mean, one), 1).transpose(1, 2)
            output = torch.cat((output, mean), 1)

        return output

class RResents(nn.Module):
    def __init__(self, num_blocks, in_dim, hidden_dim, out_dim, n_hidden):
        super(RResents, self).__init__()
        self.RResNet = RResNet(spd.SPD(metric="aff_inv"), [ProjVecField(spd.SPD(metric="aff_inv"), in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_hidden=n_hidden)
                                                       for _ in range(num_blocks)])
        self.epsilon = 1e-5

    def forward(self, input):
        # 生成协方差矩阵的对角线正则化
        I = torch.eye(input.size(1)).expand_as(input).to(input.device)
        input += I * self.epsilon
        output = self.RResNet(input)
        return output

class SPD(nn.Module):
    def __init__(self, size, epsilon):
        super(SPD, self).__init__()
        self.rect = SPDRectified().cpu()
        self.ST_tangent = SPDTangentSpace(size, vectorize=False).cpu()
        self.epsilon = epsilon

    def add_epsilon(self, output):
        I = torch.eye(output.size(1)).expand_as(output).cpu()
        output += I * self.epsilon
        return output

    def forward(self, h_co):
        h_co = h_co.cpu()
        h_co = self.add_epsilon(h_co)
        h_vec_co_Riemannian = self.rect(h_co)
        h_hidden = self.ST_tangent(h_vec_co_Riemannian)
        return h_hidden.to("cuda")

class Reduce_dimension(nn.Module):
    def __init__(self,):
        super(Reduce_dimension, self).__init__()
        self.trans = nn.Linear(2, 1, bias=False)

    def forward(self, input):
        B, C = input.shape[0], input.shape[-1]
        input_r = input.view(B, 600, 2, C).permute(0, 1, 3, 2)
        input_r = self.trans(input_r).squeeze(-1)
        return input_r

class SPD_DTI(nn.Module):
    def __init__(self, layer_gnn, device, hidden1=192, hidden2=75, n_layers=1, attn_heads=1,
                 dropout=0.0):
        super(SPD_DTI, self).__init__()
        '''Drug'''
        self.encoder_drug = Encoder(768, hidden1, 128, 3, groups=32, pad1=7, pad2=3)
        self.W_gnn = nn.ModuleList([nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2)])
        self.gnn_act = nn.GELU()
        self.W_LM = nn.Linear(hidden1, hidden2, bias=False)
        self.W_G = nn.Linear(hidden2, hidden2, bias=False)
        self.softmax_d = nn.Softmax(-1)

        '''Protein'''
        self.encoder_protein_LM = Encoder(1024, hidden1, 128, 3, groups=64, pad1=7, pad2=3)
        self.encoder_protein_SPD = Encoder(120, hidden2, 60, 3, groups=15, pad1=7, pad2=3)
        # self.Wp_LM = nn.Linear(hidden1, hidden2, bias=False)
        # self.Wp_SPD = nn.Linear(hidden2, hidden2, bias=False)
        self.R_D = Reduce_dimension()
        self.softmax_p = nn.Softmax(-1)
        # self.Informer_blocks = nn.ModuleList(
        #     [AttentionLayer(ProbAttention(None, 3, 0), hidden1 + hidden2, attn_heads),
        #      AttentionLayer(ProbAttention(None, 5, 0), hidden1 + hidden2, attn_heads)])

        '''DECISION'''
        self.map = nn.Linear(534, 267)
        self.covariance = Covariance()
        self.Down_Sample = SPDTransform(267, 64)
        self.DTI_SPD = SPD(64, 1e-6)
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.layer_gnn = layer_gnn
        self.hidden = hidden1
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.FC_out1 = SPD_LeNet(64, 2, 3)
        self.act = nn.ReLU()
        self.GELU = nn.GELU()
        # for p in self.parameters():
        #     p.requires_grad = False

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = self.gnn_act(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return xs

    def ReSize(self, feature, N):
        molecule_ST = torch.zeros((N, 100, 75), device=self.device)
        for i in range(N):
            C_L = feature[i].shape[0]
            if C_L >= 100:
                molecule_ST[i, :, :] = feature[i][0:100, :]
            else:
                molecule_ST[i, :C_L, :] = feature[i]
        return molecule_ST

    def attention_DLM(self, xs, x):
        """The attention mechanism is applied to the last layer of CNN."""
        xs_ = self.act(self.W_G(xs))
        h = self.act(self.W_LM(x))
        scale = xs_.size(-1) ** -0.5
        weights = self.softmax_d(torch.matmul(xs_, h.permute(0, 2, 1)) * scale)
        ys = torch.matmul(weights, h) + xs
        return ys

    def feature_joint(self, feature_a, feature_b):
        zeros_a = torch.zeros((feature_a.shape[0], feature_b.shape[1], feature_a.shape[2])).to(feature_a.device)
        zeros_b = torch.zeros((feature_b.shape[0], feature_a.shape[1], feature_b.shape[2])).to(feature_b.device)
        feature_a_expand = torch.cat((feature_a, zeros_a), 1)
        feature_b_expand = torch.cat((zeros_b, feature_b), 1)
        joint_feature = torch.cat((feature_a_expand, feature_b_expand), 2)
        joint_feature = self.GELU(self.map(joint_feature))
        return joint_feature

    def forward(self, inputs):

        molecule_smiles, molecule_atoms, molecule_adjs, proteins, protein_LM, protein_SPD, molecule_LM = inputs
        N = molecule_smiles.shape[0]

        """DTI 1D feature with pretrain language model"""
        proteins_LM = self.encoder_protein_LM(protein_LM.permute(0, 2, 1)).permute(0, 2, 1)
        proteins_SPD = self.encoder_protein_SPD(protein_SPD.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        proteins = torch.cat((proteins_LM, proteins_SPD), 2)
        # proteins = self.R_D(proteins)

        molecule_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        molecule_ST = self.gnn(molecule_atoms, molecule_adjs, self.layer_gnn)
        molecule_ST = self.ReSize(molecule_ST, N)
        molecule_ST = self.attention_DLM(molecule_ST, molecule_LM)
        molecules = torch.cat((molecule_LM, molecule_ST), 2)

        DT_Feature = self.feature_joint(proteins, molecules)
        DT_Feature_co = self.covariance(DT_Feature.permute(0, 2, 1))
        DT_Feature_co = self.Down_Sample(DT_Feature_co)
        DT_Feature_co = self.DTI_SPD(DT_Feature_co)
        # DT_SPD = self.DTI_SPD(DT_Feature_co)
        # print(DT_SPD.shape)
        DTI, dti_feature = self.FC_out1(DT_Feature_co)

        return DTI

    def predict(self, res):
        if res > 0.5:
            result = 1
        else:
            result = 0
        return result

    def __call__(self, data, epoch=1, train=True):
        inputs, correct_interaction, SID = data[:-2], data[-2], data[-1]
        correct_interaction = torch.LongTensor(correct_interaction.to('cpu').numpy()).cuda()
        protein_drug_interaction = self.forward(inputs)  # , dis_invariant
        if train:
            loss = F.cross_entropy(protein_drug_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction  # .to('cpu').data.numpy().reshape(-1)
            ys = F.softmax(protein_drug_interaction, 1)  # .to('cpu').data.numpy() C
            return correct_labels, ys

# 0.9579959718991505



import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from .Informer_block import AttentionLayer, ProbAttention
from .dt_spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified
from .rresnet import RResNet, ProjVecField
from .rresnet.manifolds import spd

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


class Encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=15, pad2=7):
        super(Encoder, self).__init__()
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

class SPDVectorize(nn.Module):
    def __init__(self, input_size):
        super(SPDVectorize, self).__init__()
        row_idx, col_idx = np.triu_indices(input_size)
        self.register_buffer('row_idx', torch.LongTensor(row_idx))
        self.register_buffer('col_idx', torch.LongTensor(col_idx))

    def forward(self, input):
        output = input[:, self.row_idx, self.col_idx]
        return output

class ContrastLoss(nn.Module):
    def __init__(self, source_number):
        super(ContrastLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.source_number = source_number
        pass

    def forward(self, anchor_fea, reassembly_fea, contrast_label):
        if self.source_number > 2:
            contrast_label = contrast_label.float()
            anchor_fea = anchor_fea.detach()
            loss = -(F.cosine_similarity(anchor_fea, reassembly_fea, dim=-1))
            loss = loss * contrast_label
        else:
            loss = 0 * contrast_label
        return loss.mean()

class Smooth_loss(nn.Module):

    def __init__(self, smoothing=0.1):
        super(Smooth_loss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        confidence = 1 - self.smoothing
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Covariance(nn.Module):

    def __init__(self, append_mean=False, epsilon=1e-5):
        super(Covariance, self).__init__()
        self.append_mean = append_mean
        self.epsilon = epsilon  # 小的常数用于添加到协方差矩阵的对角线上

    def forward(self, input):
        mean = torch.mean(input, 2, keepdim=True)
        x = input - mean.expand(-1, -1, input.size(2))
        output = torch.bmm(x, x.transpose(1, 2)) / input.size(1)

        if self.append_mean:
            mean_sq = torch.bmm(mean, mean.transpose(1, 2))
            output.add_(mean_sq)
            output = torch.cat((output, mean), 2)
            one = input.new(1, 1, 1).fill_(1).expand(mean.size(0), -1, -1)
            mean = torch.cat((mean, one), 1).transpose(1, 2)
            output = torch.cat((output, mean), 1)

        return output

class RResents(nn.Module):
    def __init__(self, num_blocks, in_dim, hidden_dim, out_dim, n_hidden):
        super(RResents, self).__init__()
        self.RResNet = RResNet(spd.SPD(metric="aff_inv"), [ProjVecField(spd.SPD(metric="aff_inv"), in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_hidden=n_hidden)
                                                       for _ in range(num_blocks)])
        self.epsilon = 1e-5

    def forward(self, input):
        # 生成协方差矩阵的对角线正则化
        I = torch.eye(input.size(1)).expand_as(input).to(input.device)
        input += I * self.epsilon
        output = self.RResNet(input)
        return output

class SPD(nn.Module):
    def __init__(self, size, epsilon):
        super(SPD, self).__init__()
        self.rect = SPDRectified().cpu()
        self.ST_tangent = SPDTangentSpace(size, vectorize=False).cpu()
        self.epsilon = epsilon

    def add_epsilon(self, output):
        I = torch.eye(output.size(1)).expand_as(output).cpu()
        output += I * self.epsilon
        return output

    def forward(self, h_co):
        h_co = h_co.cpu()
        h_co = self.add_epsilon(h_co)
        h_vec_co_Riemannian = self.rect(h_co)
        h_hidden = self.ST_tangent(h_vec_co_Riemannian)
        return h_hidden.to("cuda")

class Reduce_dimension(nn.Module):
    def __init__(self,):
        super(Reduce_dimension, self).__init__()
        self.trans = nn.Linear(2, 1, bias=False)

    def forward(self, input):
        B, C = input.shape[0], input.shape[-1]
        input_r = input.view(B, 600, 2, C).permute(0, 1, 3, 2)
        input_r = self.trans(input_r).squeeze(-1)
        return input_r

class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b

class Feature_Joint(nn.Module):
    def __init__(self, dim_a, dim_b, out_dim):
        super(Feature_Joint, self).__init__()
        self.map = nn.Linear(dim_a + dim_b, out_dim)
        self.GELU = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, feature_a, feature_b):
        zeros_a = torch.zeros((feature_a.shape[0], feature_a.shape[1], feature_b.shape[2])).to(feature_a.device)
        zeros_b = torch.zeros((feature_b.shape[0], feature_b.shape[1], feature_a.shape[2])).to(feature_b.device)
        feature_a_expand = torch.cat((feature_a, zeros_a), 2)
        feature_b_expand = torch.cat((zeros_b, feature_b), 2)
        joint_feature = torch.cat((feature_a_expand, feature_b_expand), 1)
        joint_feature = self.GELU(self.map(joint_feature))
        return self.drop(joint_feature)

class SPD_LeNet(nn.Module):
    def __init__(self, hidden, classes, layers):
        super(SPD_LeNet, self).__init__()
        self.vec = SPDVectorize(hidden)
        self.FC_down = nn.Linear(int((hidden+1)*hidden/2), 1024)
        self.FC_combs = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(layers)])
        self.FC_out = nn.Linear(1024, classes)
        self.layers = layers
        self.act = nn.ReLU()

    def forward(self, dti_spd):
        dti_vec = self.vec(dti_spd)  # self.BN(dti_feature.permute(0, 2, 1))
        dti_feature = self.FC_down(dti_vec)
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti = self.FC_out(dti_feature)
        return dti, dti_feature

class SPD_DTI(nn.Module):
    def __init__(self, layer_gnn, device, hidden1=192, hidden2=75, n_layers=3, dropout=0.0):
        super(SPD_DTI, self).__init__()
        '''Drug'''
        self.encoder_drug = Encoder(768, hidden1, 128, 3, groups=32, pad1=7, pad2=3)
        self.W_gnn = nn.ModuleList([nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2)])
        self.gnn_act = nn.GELU()
        self.W_LM = nn.Linear(hidden1, hidden2, bias=False)
        self.W_G = nn.Linear(hidden2, hidden2, bias=False)
        self.softmax_d = nn.Softmax(-1)
        # self.d_joint = Feature_Joint(hidden1, hidden2, hidden1 + hidden2)

        '''Protein'''
        self.encoder_protein_LM = Encoder(1024, hidden1, 128, 3, groups=64, pad1=7, pad2=3)
        self.encoder_protein_SPD = Encoder(120, hidden2, 60, 3, groups=15, pad1=7, pad2=3)
        self.R_D = Reduce_dimension()
        self.softmax_p = nn.Softmax(-1)

        '''DECISION'''
        self.dt_joint = Feature_Joint(hidden1+hidden2, hidden1+hidden2, hidden1+hidden2)
        self.covariance = Covariance()
        self.Down_Sample = SPDTransform(hidden1+hidden2, 64)
        self.DTI_SPD = SPD(64, 1e-6)
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.layer_gnn = layer_gnn
        self.classfier = SPD_LeNet(64, 2, n_layers)
        self.act = nn.ReLU()
        self.GELU = nn.GELU()
        # for p in self.parameters():
        #     p.requires_grad = False

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = self.gnn_act(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return xs

    def ReSize(self, feature, N):
        molecule_ST = torch.zeros((N, 100, 75), device=self.device)
        for i in range(N):
            C_L = feature[i].shape[0]
            if C_L >= 100:
                molecule_ST[i, :, :] = feature[i][0:100, :]
            else:
                molecule_ST[i, :C_L, :] = feature[i]
        return molecule_ST

    def attention_DLM(self, xs, x):
        """The attention mechanism is applied to the last layer of CNN."""
        xs_ = self.act(self.W_G(xs))
        h = self.act(self.W_LM(x))
        scale = xs_.size(-1) ** -0.5
        weights = self.softmax_d(torch.matmul(xs_, h.permute(0, 2, 1)) * scale)
        ys = torch.matmul(weights, h) + xs
        return ys

    def forward(self, inputs):
        """Data loading"""
        molecule_smiles, molecule_atoms, molecule_adjs, proteins, protein_LM, protein_SPD, molecule_LM = inputs
        N = molecule_smiles.shape[0]

        """Protein feature extractor"""
        proteins_LM = self.encoder_protein_LM(protein_LM.permute(0, 2, 1)).permute(0, 2, 1)
        proteins_SPD = self.encoder_protein_SPD(protein_SPD.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        proteins = torch.cat((proteins_LM, proteins_SPD), 2)

        """Drug feature extractor"""
        molecule_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        molecule_ST = self.gnn(molecule_atoms, molecule_adjs, self.layer_gnn)
        molecule_ST = self.ReSize(molecule_ST, N)
        molecule_ST = self.attention_DLM(molecule_ST, molecule_LM)
        molecules = torch.cat((molecule_LM, molecule_ST), 2)

        """SPD-based joint feature and classfier"""
        DT_Feature = self.dt_joint(proteins, molecules)
        DT_Feature_co = self.covariance(DT_Feature.permute(0, 2, 1))
        DT_Feature_co = self.Down_Sample(DT_Feature_co)
        DT_Feature_co = self.DTI_SPD(DT_Feature_co)
        DTI, dti_feature = self.classfier(DT_Feature_co)

        return DTI

    def predict(self, res):
        if res > 0.5:
            result = 1
        else:
            result = 0
        return result

    def __call__(self, data, epoch=1, train=True):
        inputs, correct_interaction, SID = data[:-2], data[-2], data[-1]
        correct_interaction = torch.LongTensor(correct_interaction.to('cpu').numpy()).cuda()
        protein_drug_interaction = self.forward(inputs)  # , dis_invariant
        if train:
            loss = F.cross_entropy(protein_drug_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction  # .to('cpu').data.numpy().reshape(-1)
            ys = F.softmax(protein_drug_interaction, 1)  # .to('cpu').data.numpy() C
            return correct_labels, ys

# 0.9588003006697172


import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from .Informer_block import AttentionLayer, ProbAttention
from .dt_spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified
from .rresnet import RResNet, ProjVecField
from .rresnet.manifolds import spd

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

    def forward(self, x):
        # xn = self.bn1(x)
        xn = self.relu(x)
        xn = self.conv1(xn)
        xn = self.bn2(xn)
        xn = self.relu(xn)
        xn = self.conv2(xn)
        return torch.cat([x, xn], 1)


class Encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=15, pad2=7):
        super(Encoder, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv1d(in_channels=inc, out_channels=inc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1,
                                 bias=False)
        self.dense_cnn = nn.ModuleList(
            [EncoderLayer(inc + growth_rate * i_la, inc + (growth_rate // 2) * i_la, growth_rate, groups, pad2) for i_la
             in range(layers)])
        self.conv_out = nn.Conv1d(in_channels=inc + growth_rate * layers, out_channels=outc, kernel_size=(pad1 * 2 + 1),
                                  stride=1, padding=pad1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.layers):
            x = self.dense_cnn[i](x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        return x

class SPDVectorize(nn.Module):
    def __init__(self, input_size):
        super(SPDVectorize, self).__init__()
        row_idx, col_idx = np.triu_indices(input_size)
        self.register_buffer('row_idx', torch.LongTensor(row_idx))
        self.register_buffer('col_idx', torch.LongTensor(col_idx))

    def forward(self, input):
        output = input[:, self.row_idx, self.col_idx]
        return output

class Atom_rep(nn.Module):
    def __init__(self, channels, device, atom_classes=16, atom_hidden=33):
        super(Atom_rep, self).__init__()
        self.embed_comg = nn.Embedding(atom_classes, atom_hidden)
        self.device = device
        self.channel = channels

    def forward(self, molecule_atoms, N):
        molecule_vec = torch.zeros((molecule_atoms.shape[0], molecule_atoms.shape[1], self.channel), device=self.device)
        for i in range(N):
            fea = torch.zeros((molecule_atoms.shape[1], self.channel), device=self.device)
            atom_fea = molecule_atoms[i][:, 0:16]
            p = torch.argmax(atom_fea, dim=1)
            com = self.embed_comg(p)
            oth1 = molecule_atoms[i][:, 44:75]
            tf = F.normalize(oth1, dim=1)
            fea[:, 0:33] = com
            fea[:, 33:64] = tf
            molecule_vec[i, :, :] = fea
        return molecule_vec

class cross_attention(nn.Module):
    def __init__(self, hidden1, hidden2, dropout):
        super(cross_attention, self).__init__()
        self.W_q = nn.Linear(hidden2, hidden2)
        self.W_k = nn.Linear(hidden1, hidden2)
        self.W_v = nn.Linear(hidden1, hidden2)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(-1)

    def forward(self, xs, x):
        q = self.W_q(xs)
        k = self.W_k(x)
        v = self.W_v(x)
        weight = torch.matmul(q, k.permute(0, 2, 1))
        scale = weight.size(-1) ** -0.5
        weights = self.softmax(self.dropout(weight * scale))
        ys = torch.matmul(weights, v) + xs
        return ys

class Smooth_loss(nn.Module):

    def __init__(self, smoothing=0.1):
        super(Smooth_loss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        confidence = 1 - self.smoothing
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Covariance(nn.Module):

    def __init__(self, append_mean=False, epsilon=1e-5):
        super(Covariance, self).__init__()
        self.append_mean = append_mean
        self.epsilon = epsilon  # 小的常数用于添加到协方差矩阵的对角线上

    def forward(self, input):
        mean = torch.mean(input, 2, keepdim=True)
        x = input - mean.expand(-1, -1, input.size(2))
        output = torch.bmm(x, x.transpose(1, 2)) / input.size(1)

        if self.append_mean:
            mean_sq = torch.bmm(mean, mean.transpose(1, 2))
            output.add_(mean_sq)
            output = torch.cat((output, mean), 2)
            one = input.new(1, 1, 1).fill_(1).expand(mean.size(0), -1, -1)
            mean = torch.cat((mean, one), 1).transpose(1, 2)
            output = torch.cat((output, mean), 1)

        return output

class RResents(nn.Module):
    def __init__(self, num_blocks, in_dim, hidden_dim, out_dim, n_hidden):
        super(RResents, self).__init__()
        self.RResNet = RResNet(spd.SPD(metric="aff_inv"), [ProjVecField(spd.SPD(metric="aff_inv"), in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_hidden=n_hidden)
                                                       for _ in range(num_blocks)])
        self.epsilon = 1e-5

    def forward(self, input):
        # 生成协方差矩阵的对角线正则化
        I = torch.eye(input.size(1)).expand_as(input).to(input.device)
        input += I * self.epsilon
        output = self.RResNet(input)
        return output

class SPD(nn.Module):
    def __init__(self, size, epsilon):
        super(SPD, self).__init__()
        self.rect = SPDRectified().cpu()
        self.ST_tangent = SPDTangentSpace(size, vectorize=False).cpu()
        self.epsilon = epsilon

    def add_epsilon(self, output):
        I = torch.eye(output.size(1)).expand_as(output).cpu()
        output += I * self.epsilon
        return output

    def forward(self, h_co):
        h_co = h_co.cpu()
        h_co = self.add_epsilon(h_co)
        h_vec_co_Riemannian = self.rect(h_co)
        h_hidden = self.ST_tangent(h_vec_co_Riemannian)
        return h_hidden.to("cuda")

class Reduce_dimension(nn.Module):
    def __init__(self,):
        super(Reduce_dimension, self).__init__()
        self.trans = nn.Linear(2, 1, bias=False)

    def forward(self, input):
        B, C = input.shape[0], input.shape[-1]
        input_r = input.view(B, 600, 2, C).permute(0, 1, 3, 2)
        input_r = self.trans(input_r).squeeze(-1)
        return input_r

class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b

class Feature_Joint(nn.Module):
    def __init__(self, dim_a, dim_b, out_dim):
        super(Feature_Joint, self).__init__()
        self.map = nn.Linear(dim_a + dim_b, out_dim)
        self.GELU = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, feature_a, feature_b):
        zeros_a = torch.zeros((feature_a.shape[0], feature_a.shape[1], feature_b.shape[2])).to(feature_a.device)
        zeros_b = torch.zeros((feature_b.shape[0], feature_b.shape[1], feature_a.shape[2])).to(feature_b.device)
        feature_a_expand = torch.cat((feature_a, zeros_a), 2)
        feature_b_expand = torch.cat((zeros_b, feature_b), 2)
        joint_feature = torch.cat((feature_a_expand, feature_b_expand), 1)
        joint_feature = self.GELU(self.map(joint_feature))
        return self.drop(joint_feature)

class SPD_LeNet(nn.Module):
    def __init__(self, hidden, classes, layers):
        super(SPD_LeNet, self).__init__()
        self.vec = SPDVectorize(hidden)
        self.FC_down = nn.Linear(int((hidden+1)*hidden/2), 1024)
        self.FC_combs = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(layers)])
        self.FC_out = nn.Linear(1024, classes)
        self.layers = layers
        self.act = nn.ReLU()

    def forward(self, dti_spd):
        dti_vec = self.vec(dti_spd)  # self.BN(dti_feature.permute(0, 2, 1))
        dti_feature = self.FC_down(dti_vec)
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti = self.FC_out(dti_feature)
        return dti, dti_feature

class SPD_DTI(nn.Module):
    def __init__(self, layer_gnn, device, hidden1=192, hidden2=64, n_layers=3, dropout=0.05):
        super(SPD_DTI, self).__init__()
        '''Drug'''
        self.encoder_drug = Encoder(768, hidden1, 64, 3, groups=32, pad1=7, pad2=3)
        self.atom_rep = Atom_rep(hidden2, device)
        self.W_gnn = nn.ModuleList([nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2)])
        self.gnn_act = nn.GELU()
        self.C = hidden2
        self.cross_attention = cross_attention(hidden1, hidden2, dropout)
        # self.d_joint = Feature_Joint(hidden1, hidden2, hidden1 + hidden2)

        '''Protein'''
        self.encoder_protein_LM = Encoder(1024, hidden1, 128, 3, groups=64, pad1=7, pad2=3)
        self.encoder_protein_SPD = Encoder(120, hidden2, 60, 3, groups=15, pad1=7, pad2=3)
        self.R_D = Reduce_dimension()
        self.softmax_p = nn.Softmax(-1)

        '''DECISION'''
        self.dt_joint = Feature_Joint(hidden1+hidden2, hidden1+hidden2, hidden1+hidden2)
        self.covariance = Covariance()
        self.Down_Sample = SPDTransform(hidden1+hidden2, 64)
        self.DTI_SPD = SPD(64, 1e-6)
        self.device = device
        self.layer_gnn = layer_gnn
        self.classfier = SPD_LeNet(64, 2, n_layers)
        self.act = nn.ReLU()
        self.GELU = nn.GELU()
        # for p in self.parameters():
        #     p.requires_grad = False

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = self.gnn_act(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return xs

    def ReSize(self, feature, N):
        molecule_ST = torch.zeros((N, 100, self.C), device=self.device)
        for i in range(N):
            C_L = feature[i].shape[0]
            if C_L >= 100:
                molecule_ST[i, :, :] = feature[i][0:100, :]
            else:
                molecule_ST[i, :C_L, :] = feature[i]
        return molecule_ST

    def forward(self, inputs):
        """Data loading"""
        molecule_smiles, molecule_atoms, molecule_adjs, proteins, protein_LM, protein_SPD, molecule_LM = inputs
        N = molecule_smiles.shape[0]

        """Protein feature extractor"""
        proteins_LM = self.encoder_protein_LM(protein_LM.permute(0, 2, 1)).permute(0, 2, 1)
        proteins_SPD = self.encoder_protein_SPD(protein_SPD.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        proteins = torch.cat((proteins_LM, proteins_SPD), 2)

        """Drug feature extractor"""
        molecule_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        molecule_vec = self.atom_rep(molecule_atoms, N)
        molecule_ST = self.gnn(molecule_vec, molecule_adjs, self.layer_gnn)
        molecule_ST = self.ReSize(molecule_ST, N)
        molecule_ST = self.cross_attention(molecule_ST, molecule_LM)
        molecules = torch.cat((molecule_LM, molecule_ST), 2)

        """SPD-based joint feature and classfier"""
        Joint_Feature = self.dt_joint(proteins, molecules)
        DTI_Feature = self.covariance(Joint_Feature.permute(0, 2, 1))
        DTI_Feature = self.Down_Sample(DTI_Feature)
        DTI_Feature = self.DTI_SPD(DTI_Feature)
        DTI, dti_feature = self.classfier(DTI_Feature)

        return DTI

    def predict(self, res):
        if res > 0.5:
            result = 1
        else:
            result = 0
        return result

    def __call__(self, data, epoch=1, train=True):
        inputs, correct_interaction, SID = data[:-2], data[-2], data[-1]
        correct_interaction = torch.LongTensor(correct_interaction.to('cpu').numpy()).cuda()
        protein_drug_interaction = self.forward(inputs)  # , dis_invariant
        if train:
            loss = F.cross_entropy(protein_drug_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction  # .to('cpu').data.numpy().reshape(-1)
            ys = F.softmax(protein_drug_interaction, 1)  # .to('cpu').data.numpy() C
            return correct_labels, ys

# 0.9627835353774188 0.9614679388444631

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from .Informer_block import AttentionLayer, ProbAttention
from .dt_spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified
from .rresnet import RResNet, ProjVecField
from .rresnet.manifolds import spd

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

    def forward(self, x):
        # xn = self.bn1(x)
        xn = self.relu(x)
        xn = self.conv1(xn)
        xn = self.bn2(xn)
        xn = self.relu(xn)
        xn = self.conv2(xn)
        return torch.cat([x, xn], 1)


class Encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=15, pad2=7):
        super(Encoder, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv1d(in_channels=inc, out_channels=inc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1,
                                 bias=False)
        self.dense_cnn = nn.ModuleList(
            [EncoderLayer(inc + growth_rate * i_la, inc + (growth_rate // 2) * i_la, growth_rate, groups, pad2) for i_la
             in range(layers)])
        self.conv_out = nn.Conv1d(in_channels=inc + growth_rate * layers, out_channels=outc, kernel_size=(pad1 * 2 + 1),
                                  stride=1, padding=pad1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.layers):
            x = self.dense_cnn[i](x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        return x

class SPDVectorize(nn.Module):
    def __init__(self, input_size):
        super(SPDVectorize, self).__init__()
        row_idx, col_idx = np.triu_indices(input_size)
        self.register_buffer('row_idx', torch.LongTensor(row_idx))
        self.register_buffer('col_idx', torch.LongTensor(col_idx))

    def forward(self, input):
        output = input[:, self.row_idx, self.col_idx]
        return output

class Atom_rep(nn.Module):
    def __init__(self, device, atom_classes=16, atom_hidden=44):
        super(Atom_rep, self).__init__()
        self.embed_comg = nn.Embedding(atom_classes, atom_hidden)
        self.device = device

    def forward(self, molecule_atoms, N):
        molecule_vec = torch.zeros((molecule_atoms.shape[0], molecule_atoms.shape[1], 75), device=self.device)
        for i in range(N):
            fea = torch.zeros((molecule_atoms.shape[1], 75), device=self.device)
            atom_fea = molecule_atoms[i][:, 0:16]
            p = torch.argmax(atom_fea, dim=1)
            com = self.embed_comg(p)
            oth1 = molecule_atoms[i][:, 44:75]
            tf = F.normalize(oth1, dim=1)
            fea[:, 0:44] = com
            fea[:, 44:75] = tf
            molecule_vec[i, :, :] = fea
        return molecule_vec

class cross_attention(nn.Module):
    def __init__(self, hidden1, hidden2, dropout):
        super(cross_attention, self).__init__()
        self.W_q = nn.Linear(hidden2, hidden2)
        self.W_k = nn.Linear(hidden1, hidden2)
        self.W_v = nn.Linear(hidden1, hidden2)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(-1)

    def forward(self, xs, x):
        q = self.W_q(xs)
        k = self.W_k(x)
        v = self.W_v(x)
        weight = torch.matmul(q, k.permute(0, 2, 1))
        scale = weight.size(-1) ** -0.5
        weights = self.softmax(self.dropout(weight * scale))
        ys = torch.matmul(weights, v) + xs
        return ys

class Smooth_loss(nn.Module):

    def __init__(self, smoothing=0.1):
        super(Smooth_loss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        confidence = 1 - self.smoothing
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Covariance(nn.Module):

    def __init__(self, append_mean=False, epsilon=1e-5):
        super(Covariance, self).__init__()
        self.append_mean = append_mean
        self.epsilon = epsilon  # 小的常数用于添加到协方差矩阵的对角线上

    def forward(self, input):
        mean = torch.mean(input, 2, keepdim=True)
        x = input - mean.expand(-1, -1, input.size(2))
        output = torch.bmm(x, x.transpose(1, 2)) / input.size(1)

        if self.append_mean:
            mean_sq = torch.bmm(mean, mean.transpose(1, 2))
            output.add_(mean_sq)
            output = torch.cat((output, mean), 2)
            one = input.new(1, 1, 1).fill_(1).expand(mean.size(0), -1, -1)
            mean = torch.cat((mean, one), 1).transpose(1, 2)
            output = torch.cat((output, mean), 1)

        return output

class RResents(nn.Module):
    def __init__(self, num_blocks, in_dim, hidden_dim, out_dim, n_hidden):
        super(RResents, self).__init__()
        self.RResNet = RResNet(spd.SPD(metric="aff_inv"), [ProjVecField(spd.SPD(metric="aff_inv"), in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_hidden=n_hidden)
                                                       for _ in range(num_blocks)])
        self.epsilon = 1e-5

    def forward(self, input):
        # 生成协方差矩阵的对角线正则化
        I = torch.eye(input.size(1)).expand_as(input).to(input.device)
        input += I * self.epsilon
        output = self.RResNet(input)
        return output

class SPD(nn.Module):
    def __init__(self, size, epsilon):
        super(SPD, self).__init__()
        self.rect = SPDRectified().cpu()
        self.ST_tangent = SPDTangentSpace(size, vectorize=False).cpu()
        self.epsilon = epsilon

    def add_epsilon(self, output):
        I = torch.eye(output.size(1)).expand_as(output).cpu()
        output += I * self.epsilon
        return output

    def forward(self, h_co):
        h_co = h_co.cpu()
        h_co = self.add_epsilon(h_co)
        h_vec_co_Riemannian = self.rect(h_co)
        h_hidden = self.ST_tangent(h_vec_co_Riemannian)
        return h_hidden.to("cuda")

class Reduce_dimension(nn.Module):
    def __init__(self,):
        super(Reduce_dimension, self).__init__()
        self.trans = nn.Linear(2, 1, bias=False)

    def forward(self, input):
        B, C = input.shape[0], input.shape[-1]
        input_r = input.view(B, 600, 2, C).permute(0, 1, 3, 2)
        input_r = self.trans(input_r).squeeze(-1)
        return input_r

class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b

class Feature_Joint(nn.Module):
    def __init__(self, dim_a, dim_b, out_dim):
        super(Feature_Joint, self).__init__()
        self.map = nn.Linear(dim_a + dim_b, out_dim)
        self.GELU = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, feature_a, feature_b):
        zeros_a = torch.zeros((feature_a.shape[0], feature_a.shape[1], feature_b.shape[2])).to(feature_a.device)
        zeros_b = torch.zeros((feature_b.shape[0], feature_b.shape[1], feature_a.shape[2])).to(feature_b.device)
        feature_a_expand = torch.cat((feature_a, zeros_a), 2)
        feature_b_expand = torch.cat((zeros_b, feature_b), 2)
        joint_feature = torch.cat((feature_a_expand, feature_b_expand), 1)
        joint_feature = self.GELU(self.map(joint_feature))
        return self.drop(joint_feature)

class SPD_LeNet(nn.Module):
    def __init__(self, hidden, classes, layers):
        super(SPD_LeNet, self).__init__()
        self.vec = SPDVectorize(hidden)
        self.FC_down = nn.Linear(int((hidden+1)*hidden/2), 1024)
        self.FC_combs = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(layers)])
        self.FC_out = nn.Linear(1024, classes)
        self.layers = layers
        self.act = nn.ReLU()

    def forward(self, dti_spd):
        dti_vec = self.vec(dti_spd)  # self.BN(dti_feature.permute(0, 2, 1))
        dti_feature = self.FC_down(dti_vec)
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti = self.FC_out(dti_feature)
        return dti, dti_feature

class SPD_DTI(nn.Module):
    def __init__(self, layer_gnn, device, hidden1=192, hidden2=75, n_layers=3, dropout=0.05):
        super(SPD_DTI, self).__init__()
        '''Drug'''
        self.encoder_drug = Encoder(768, hidden1, 64, 3, groups=32, pad1=7, pad2=3)
        self.atom_rep = Atom_rep(device)
        self.W_gnn = nn.ModuleList([nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2)])
        self.gnn_act = nn.GELU()
        self.cross_attention = cross_attention(hidden1, hidden2, dropout)
        # self.d_joint = Feature_Joint(hidden1, hidden2, hidden1 + hidden2)

        '''Protein'''
        self.encoder_protein_LM = Encoder(1024, hidden1, 128, 3, groups=32, pad1=7, pad2=3)
        self.encoder_protein_SPD = Encoder(120, hidden2, 60, 3, groups=15, pad1=7, pad2=3)
        self.R_D = Reduce_dimension()
        self.softmax_p = nn.Softmax(-1)

        '''DECISION'''
        self.dt_joint = Feature_Joint(hidden1+hidden2, hidden1+hidden2, hidden1+hidden2)
        self.covariance = Covariance()
        self.Down_Sample = SPDTransform(hidden1+hidden2, 64)
        self.DTI_SPD = SPD(64, 1e-6)
        self.device = device
        self.layer_gnn = layer_gnn
        self.classfier = SPD_LeNet(64, 2, n_layers)
        self.act = nn.ReLU()
        self.GELU = nn.GELU()
        # for p in self.parameters():
        #     p.requires_grad = False

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = self.gnn_act(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return xs

    def ReSize(self, feature, N):
        molecule_ST = torch.zeros((N, 100, 75), device=self.device)
        for i in range(N):
            C_L = feature[i].shape[0]
            if C_L >= 100:
                molecule_ST[i, :, :] = feature[i][0:100, :]
            else:
                molecule_ST[i, :C_L, :] = feature[i]
        return molecule_ST

    def forward(self, inputs):
        """Data loading"""
        molecule_smiles, molecule_atoms, molecule_adjs, proteins, protein_LM, protein_SPD, molecule_LM = inputs
        N = molecule_smiles.shape[0]

        """Protein feature extractor"""
        proteins_LM = self.encoder_protein_LM(protein_LM.permute(0, 2, 1)).permute(0, 2, 1)
        proteins_SPD = self.encoder_protein_SPD(protein_SPD.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        proteins = torch.cat((proteins_LM, proteins_SPD), 2)

        """Drug feature extractor"""
        molecule_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        molecule_vec = self.atom_rep(molecule_atoms, N)
        molecule_ST = self.gnn(molecule_vec, molecule_adjs, self.layer_gnn)
        molecule_ST = self.ReSize(molecule_ST, N)
        molecule_ST = self.cross_attention(molecule_ST, molecule_LM)
        molecules = torch.cat((molecule_LM, molecule_ST), 2)

        """SPD-based joint feature and classfier"""
        Joint_Feature = self.dt_joint(proteins, molecules)
        DTI_Feature = self.covariance(Joint_Feature.permute(0, 2, 1))
        DTI_Feature = self.Down_Sample(DTI_Feature)
        DTI_Feature = self.DTI_SPD(DTI_Feature)
        DTI, dti_feature = self.classfier(DTI_Feature)

        return DTI

    def predict(self, res):
        if res > 0.5:
            result = 1
        else:
            result = 0
        return result

    def __call__(self, data, epoch=1, train=True):
        inputs, correct_interaction, SID = data[:-2], data[-2], data[-1]
        correct_interaction = torch.LongTensor(correct_interaction.to('cpu').numpy()).cuda()
        protein_drug_interaction = self.forward(inputs)  # , dis_invariant
        if train:
            loss = F.cross_entropy(protein_drug_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction  # .to('cpu').data.numpy().reshape(-1)
            ys = F.softmax(protein_drug_interaction, 1)  # .to('cpu').data.numpy() C
            return correct_labels, ys

# 0.959725914470489	0.9601802483158144


import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from .Informer_block import AttentionLayer, ProbAttention
from .dt_spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified

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

    def forward(self, x):
        # xn = self.bn1(x)
        xn = self.relu(x)
        xn = self.conv1(xn)
        xn = self.bn2(xn)
        xn = self.relu(xn)
        xn = self.conv2(xn)
        return torch.cat([x, xn], 1)


class Encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=15, pad2=7):
        super(Encoder, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv1d(in_channels=inc, out_channels=inc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1,
                                 bias=False)
        self.dense_cnn = nn.ModuleList(
            [EncoderLayer(inc + growth_rate * i_la, inc + (growth_rate // 2) * i_la, growth_rate, groups, pad2) for i_la
             in range(layers)])
        self.conv_out = nn.Conv1d(in_channels=inc + growth_rate * layers, out_channels=outc, kernel_size=(pad1 * 2 + 1),
                                  stride=1, padding=pad1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.layers):
            x = self.dense_cnn[i](x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        return x

class SPDVectorize(nn.Module):
    def __init__(self, input_size):
        super(SPDVectorize, self).__init__()
        row_idx, col_idx = np.triu_indices(input_size)
        self.register_buffer('row_idx', torch.LongTensor(row_idx))
        self.register_buffer('col_idx', torch.LongTensor(col_idx))

    def forward(self, input):
        output = input[:, self.row_idx, self.col_idx]
        return output

class Atom_rep(nn.Module):
    def __init__(self, channels, device, atom_classes=16, atom_hidden=33):
        super(Atom_rep, self).__init__()
        self.embed_comg = nn.Embedding(atom_classes, atom_hidden)
        self.device = device
        self.channel = channels

    def forward(self, molecule_atoms, N):
        molecule_vec = torch.zeros((molecule_atoms.shape[0], molecule_atoms.shape[1], self.channel), device=self.device)
        for i in range(N):
            fea = torch.zeros((molecule_atoms.shape[1], self.channel), device=self.device)
            atom_fea = molecule_atoms[i][:, 0:16]
            p = torch.argmax(atom_fea, dim=1)
            com = self.embed_comg(p)
            oth1 = molecule_atoms[i][:, 44:75]
            tf = F.normalize(oth1, dim=1)
            fea[:, 0:33] = com
            fea[:, 33:64] = tf
            molecule_vec[i, :, :] = fea
        return molecule_vec

class cross_attention(nn.Module):
    def __init__(self, hidden1, hidden2, dropout):
        super(cross_attention, self).__init__()
        self.W_q = nn.Linear(hidden2, hidden2)
        self.W_k = nn.Linear(hidden1, hidden2)
        self.W_v = nn.Linear(hidden1, hidden2)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(-1)

    def forward(self, xs, x):
        q = self.W_q(xs)
        k = self.W_k(x)
        v = self.W_v(x)
        weight = torch.matmul(q, k.permute(0, 2, 1))
        scale = weight.size(-1) ** -0.5
        weights = self.softmax(self.dropout(weight * scale))
        ys = torch.matmul(weights, v) + xs
        return ys

class Smooth_loss(nn.Module):

    def __init__(self, smoothing=0.1):
        super(Smooth_loss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        confidence = 1 - self.smoothing
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Covariance(nn.Module):

    def __init__(self, append_mean=False, epsilon=1e-5):
        super(Covariance, self).__init__()
        self.append_mean = append_mean
        self.epsilon = epsilon  # 小的常数用于添加到协方差矩阵的对角线上

    def forward(self, input):
        mean = torch.mean(input, 2, keepdim=True)
        x = input - mean.expand(-1, -1, input.size(2))
        output = torch.bmm(x, x.transpose(1, 2)) / input.size(1)

        if self.append_mean:
            mean_sq = torch.bmm(mean, mean.transpose(1, 2))
            output.add_(mean_sq)
            output = torch.cat((output, mean), 2)
            one = input.new(1, 1, 1).fill_(1).expand(mean.size(0), -1, -1)
            mean = torch.cat((mean, one), 1).transpose(1, 2)
            output = torch.cat((output, mean), 1)

        return output

class RResents(nn.Module):
    def __init__(self, num_blocks, in_dim, hidden_dim, out_dim, n_hidden):
        super(RResents, self).__init__()
        self.RResNet = RResNet(spd.SPD(metric="aff_inv"), [ProjVecField(spd.SPD(metric="aff_inv"), in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_hidden=n_hidden)
                                                       for _ in range(num_blocks)])
        self.epsilon = 1e-5

    def forward(self, input):
        # 生成协方差矩阵的对角线正则化
        I = torch.eye(input.size(1)).expand_as(input).to(input.device)
        input += I * self.epsilon
        output = self.RResNet(input)
        return output

class SPD(nn.Module):
    def __init__(self, size, epsilon):
        super(SPD, self).__init__()
        self.rect = SPDRectified().cpu()
        self.ST_tangent = SPDTangentSpace(size, vectorize=False).cpu()
        self.epsilon = epsilon

    def add_epsilon(self, output):
        I = torch.eye(output.size(1)).expand_as(output).cpu()
        output += I * self.epsilon
        return output

    def forward(self, h_co):
        h_co = h_co.cpu()
        h_co = self.add_epsilon(h_co)
        h_vec_co_Riemannian = self.rect(h_co)
        h_hidden = self.ST_tangent(h_vec_co_Riemannian)
        return h_hidden.to("cuda")

class Reduce_dimension(nn.Module):
    def __init__(self,):
        super(Reduce_dimension, self).__init__()
        self.trans = nn.Linear(2, 1, bias=False)

    def forward(self, input):
        B, C = input.shape[0], input.shape[-1]
        input_r = input.view(B, 600, 2, C).permute(0, 1, 3, 2)
        input_r = self.trans(input_r).squeeze(-1)
        return input_r

class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b

class Feature_Joint(nn.Module):
    def __init__(self, dim_a, dim_b, out_dim):
        super(Feature_Joint, self).__init__()
        self.map = nn.Linear(dim_a + dim_b, out_dim)
        self.GELU = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, feature_a, feature_b):
        zeros_a = torch.zeros((feature_a.shape[0], feature_a.shape[1], feature_b.shape[2])).to(feature_a.device)
        zeros_b = torch.zeros((feature_b.shape[0], feature_b.shape[1], feature_a.shape[2])).to(feature_b.device)
        feature_a_expand = torch.cat((feature_a, zeros_a), 2)
        feature_b_expand = torch.cat((zeros_b, feature_b), 2)
        joint_feature = torch.cat((feature_a_expand, feature_b_expand), 1)
        joint_feature = self.GELU(self.map(joint_feature))
        return self.drop(joint_feature)

class SPD_LeNet(nn.Module):
    def __init__(self, hidden, classes, layers):
        super(SPD_LeNet, self).__init__()
        self.vec = SPDVectorize(hidden)
        self.FC_down = nn.Linear(int((hidden+1)*hidden/2), 512)
        self.FC_combs = nn.ModuleList([nn.Linear(512, 512) for _ in range(layers)])
        self.FC_out = nn.Linear(512, classes)
        self.layers = layers
        self.act = nn.ReLU()

    def forward(self, dti_spd):
        dti_vec = self.vec(dti_spd)  # self.BN(dti_feature.permute(0, 2, 1))
        dti_feature = self.FC_down(dti_vec)
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti = self.FC_out(dti_feature)
        return dti, dti_feature

class SPD_DTI(nn.Module):
    def __init__(self, layer_gnn, device, hidden1=192, hidden2=64, n_layers=3, dropout=0.05):
        super(SPD_DTI, self).__init__()
        '''Drug'''
        self.encoder_drug = Encoder(768, hidden1, 64, 3, groups=32, pad1=7, pad2=3)
        self.atom_rep = Atom_rep(hidden2, device)
        self.W_gnn = nn.ModuleList([nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2)])
        self.gnn_act = nn.GELU()
        self.C = hidden2
        self.cross_attention = cross_attention(hidden1, hidden2, dropout)
        # self.d_joint = Feature_Joint(hidden1, hidden2, hidden1 + hidden2)

        '''Protein'''
        self.encoder_protein_LM = Encoder(1024, hidden1, 128, 3, groups=64, pad1=7, pad2=3)
        self.encoder_protein_SPD = Encoder(120, hidden2, 30, 6, groups=15, pad1=7, pad2=3)
        self.R_D = Reduce_dimension()
        self.softmax_p = nn.Softmax(-1)

        '''DECISION'''
        self.dt_joint = Feature_Joint(hidden1+hidden2, hidden1+hidden2, hidden1+hidden2)
        self.covariance = Covariance()
        self.Down_Sample = SPDTransform(hidden1+hidden2, 32)
        self.DTI_SPD = SPD(32, 1e-6)
        self.device = device
        self.layer_gnn = layer_gnn
        self.classfier = SPD_LeNet(32, 2, n_layers)
        self.act = nn.ReLU()
        self.GELU = nn.GELU()
        # for p in self.parameters():
        #     p.requires_grad = False

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = self.gnn_act(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return xs

    def ReSize(self, feature, N):
        molecule_ST = torch.zeros((N, 100, self.C), device=self.device)
        for i in range(N):
            C_L = feature[i].shape[0]
            if C_L >= 100:
                molecule_ST[i, :, :] = feature[i][0:100, :]
            else:
                molecule_ST[i, :C_L, :] = feature[i]
        return molecule_ST

    def forward(self, inputs):
        """Data loading"""
        molecule_smiles, molecule_atoms, molecule_adjs, proteins, protein_LM, protein_SPD, molecule_LM = inputs
        N = molecule_smiles.shape[0]

        """Protein feature extractor"""
        proteins_LM = self.encoder_protein_LM(protein_LM.permute(0, 2, 1)).permute(0, 2, 1)
        proteins_SPD = self.encoder_protein_SPD(protein_SPD.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        proteins = torch.cat((proteins_LM, proteins_SPD), 2)

        """Drug feature extractor"""
        molecule_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        molecule_vec = self.atom_rep(molecule_atoms, N)
        molecule_ST = self.gnn(molecule_vec, molecule_adjs, self.layer_gnn)
        molecule_ST = self.ReSize(molecule_ST, N)
        molecule_ST = self.cross_attention(molecule_ST, molecule_LM)
        molecules = torch.cat((molecule_LM, molecule_ST), 2)

        """SPD-based joint feature and classfier"""
        Joint_Feature = self.dt_joint(proteins, molecules)
        DTI_Feature = self.covariance(Joint_Feature.permute(0, 2, 1))
        DTI_Feature = self.Down_Sample(DTI_Feature)
        DTI_Feature = self.DTI_SPD(DTI_Feature)
        DTI, dti_feature = self.classfier(DTI_Feature)

        return DTI

    def predict(self, res):
        if res > 0.5:
            result = 1
        else:
            result = 0
        return result

    def __call__(self, data, epoch=1, train=True):
        inputs, correct_interaction, SID = data[:-2], data[-2], data[-1]
        correct_interaction = torch.LongTensor(correct_interaction.to('cpu').numpy()).cuda()
        protein_drug_interaction = self.forward(inputs)  # , dis_invariant
        if train:
            loss = F.cross_entropy(protein_drug_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction  # .to('cpu').data.numpy().reshape(-1)
            ys = F.softmax(protein_drug_interaction, 1)  # .to('cpu').data.numpy() C
            return correct_labels, ys


# 0.9609231881001679	0.9627385947773598 wending


import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from .Informer_block import AttentionLayer, ProbAttention
from .dt_spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified

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

    def forward(self, x):
        # xn = self.bn1(x)
        xn = self.relu(x)
        xn = self.conv1(xn)
        xn = self.bn2(xn)
        xn = self.relu(xn)
        xn = self.conv2(xn)
        return torch.cat([x, xn], 1)


class Encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=15, pad2=7):
        super(Encoder, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv1d(in_channels=inc, out_channels=inc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1,
                                 bias=False)
        self.dense_cnn = nn.ModuleList(
            [EncoderLayer(inc + growth_rate * i_la, inc + (growth_rate // 2) * i_la, growth_rate, groups, pad2) for i_la
             in range(layers)])
        self.conv_out = nn.Conv1d(in_channels=inc + growth_rate * layers, out_channels=outc, kernel_size=(pad1 * 2 + 1),
                                  stride=1, padding=pad1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.layers):
            x = self.dense_cnn[i](x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        return x

class SPDVectorize(nn.Module):
    def __init__(self, input_size):
        super(SPDVectorize, self).__init__()
        row_idx, col_idx = np.triu_indices(input_size)
        self.register_buffer('row_idx', torch.LongTensor(row_idx))
        self.register_buffer('col_idx', torch.LongTensor(col_idx))

    def forward(self, input):
        output = input[:, self.row_idx, self.col_idx]
        return output

class Atom_rep(nn.Module):
    def __init__(self, channels, device, atom_classes=16, atom_hidden=33):
        super(Atom_rep, self).__init__()
        self.embed_comg = nn.Embedding(atom_classes, atom_hidden)
        self.device = device
        self.channel = channels

    def forward(self, molecule_atoms, N):
        molecule_vec = torch.zeros((molecule_atoms.shape[0], molecule_atoms.shape[1], self.channel), device=self.device)
        for i in range(N):
            fea = torch.zeros((molecule_atoms.shape[1], self.channel), device=self.device)
            atom_fea = molecule_atoms[i][:, 0:16]
            p = torch.argmax(atom_fea, dim=1)
            com = self.embed_comg(p)
            oth1 = molecule_atoms[i][:, 44:75]
            tf = F.normalize(oth1, dim=1)
            fea[:, 0:33] = com
            fea[:, 33:64] = tf
            molecule_vec[i, :, :] = fea
        return molecule_vec

class cross_attention(nn.Module):
    def __init__(self, hidden1, hidden2, dropout):
        super(cross_attention, self).__init__()
        self.W_q = nn.Linear(hidden2, hidden2)
        self.W_k = nn.Linear(hidden1, hidden2)
        self.W_v = nn.Linear(hidden1, hidden2)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(-1)

    def forward(self, xs, x):
        q = self.W_q(xs)
        k = self.W_k(x)
        v = self.W_v(x)
        weight = torch.matmul(q, k.permute(0, 2, 1))
        scale = weight.size(-1) ** -0.5
        weights = self.softmax(self.dropout(weight * scale))
        ys = torch.matmul(weights, v) + xs
        return ys

class Smooth_loss(nn.Module):

    def __init__(self, smoothing=0.1):
        super(Smooth_loss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        confidence = 1 - self.smoothing
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Covariance(nn.Module):

    def __init__(self, append_mean=False, epsilon=1e-5):
        super(Covariance, self).__init__()
        self.append_mean = append_mean
        self.epsilon = epsilon  # 小的常数用于添加到协方差矩阵的对角线上

    def forward(self, input):
        mean = torch.mean(input, 2, keepdim=True)
        x = input - mean.expand(-1, -1, input.size(2))
        output = torch.bmm(x, x.transpose(1, 2)) / input.size(1)

        if self.append_mean:
            mean_sq = torch.bmm(mean, mean.transpose(1, 2))
            output.add_(mean_sq)
            output = torch.cat((output, mean), 2)
            one = input.new(1, 1, 1).fill_(1).expand(mean.size(0), -1, -1)
            mean = torch.cat((mean, one), 1).transpose(1, 2)
            output = torch.cat((output, mean), 1)

        return output

class RResents(nn.Module):
    def __init__(self, num_blocks, in_dim, hidden_dim, out_dim, n_hidden):
        super(RResents, self).__init__()
        self.RResNet = RResNet(spd.SPD(metric="aff_inv"), [ProjVecField(spd.SPD(metric="aff_inv"), in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_hidden=n_hidden)
                                                       for _ in range(num_blocks)])
        self.epsilon = 1e-5

    def forward(self, input):
        # 生成协方差矩阵的对角线正则化
        I = torch.eye(input.size(1)).expand_as(input).to(input.device)
        input += I * self.epsilon
        output = self.RResNet(input)
        return output

class SPD(nn.Module):
    def __init__(self, size, epsilon):
        super(SPD, self).__init__()
        self.rect = SPDRectified().cpu()
        self.ST_tangent = SPDTangentSpace(size, vectorize=False).cpu()
        self.epsilon = epsilon

    def add_epsilon(self, output):
        I = torch.eye(output.size(1)).expand_as(output).cpu()
        output += I * self.epsilon
        return output

    def forward(self, h_co):
        h_co = h_co.cpu()
        h_co = self.add_epsilon(h_co)
        h_vec_co_Riemannian = self.rect(h_co)
        h_hidden = self.ST_tangent(h_vec_co_Riemannian)
        return h_hidden.to("cuda")

class Reduce_dimension(nn.Module):
    def __init__(self,):
        super(Reduce_dimension, self).__init__()
        self.trans = nn.Linear(2, 1, bias=False)

    def forward(self, input):
        B, C = input.shape[0], input.shape[-1]
        input_r = input.view(B, 600, 2, C).permute(0, 1, 3, 2)
        input_r = self.trans(input_r).squeeze(-1)
        return input_r

class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b

class Feature_Joint(nn.Module):
    def __init__(self, dim_a, dim_b, out_dim):
        super(Feature_Joint, self).__init__()
        self.map = nn.Linear(dim_a + dim_b, out_dim)
        self.GELU = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, feature_a, feature_b):
        zeros_a = torch.zeros((feature_a.shape[0], feature_a.shape[1], feature_b.shape[2])).to(feature_a.device)
        zeros_b = torch.zeros((feature_b.shape[0], feature_b.shape[1], feature_a.shape[2])).to(feature_b.device)
        feature_a_expand = torch.cat((feature_a, zeros_a), 2)
        feature_b_expand = torch.cat((zeros_b, feature_b), 2)
        joint_feature = torch.cat((feature_a_expand, feature_b_expand), 1)
        joint_feature = self.GELU(self.map(joint_feature))
        return self.drop(joint_feature)

class SPD_LeNet(nn.Module):
    def __init__(self, hidden, classes, layers):
        super(SPD_LeNet, self).__init__()
        self.vec = SPDVectorize(hidden)
        self.FC_down = nn.Linear(int((hidden+1)*hidden/2), 512)
        self.FC_combs = nn.ModuleList([nn.Linear(512, 512) for _ in range(layers)])
        self.FC_out = nn.Linear(512, classes)
        self.layers = layers
        self.act = nn.ReLU()

    def forward(self, dti_spd):
        dti_vec = self.vec(dti_spd)  # self.BN(dti_feature.permute(0, 2, 1))
        dti_feature = self.FC_down(dti_vec)
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti = self.FC_out(dti_feature)
        return dti, dti_feature

class SPD_DTI(nn.Module):
    def __init__(self, layer_gnn, device, hidden1=192, hidden2=64, n_layers=3, dropout=0.05):
        super(SPD_DTI, self).__init__()
        '''Drug'''
        self.encoder_drug = Encoder(768, hidden1, 64, 3, groups=32, pad1=7, pad2=3)
        self.atom_rep = Atom_rep(hidden2, device)
        self.W_gnn = nn.ModuleList([nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2)])
        self.gnn_act = nn.GELU()
        self.C = hidden2
        self.cross_attention = cross_attention(hidden1, hidden2, dropout)
        # self.d_joint = Feature_Joint(hidden1, hidden2, hidden1 + hidden2)

        '''Protein'''
        self.encoder_protein_LM = Encoder(1024, hidden1, 128, 3, groups=64, pad1=7, pad2=3)
        self.encoder_protein_SPD = Encoder(120, hidden2, 60, 5, groups=15, pad1=3, pad2=3)
        self.R_D = Reduce_dimension()
        self.softmax_p = nn.Softmax(-1)

        '''DECISION'''
        self.dt_joint = Feature_Joint(hidden1+hidden2, hidden1+hidden2, hidden1+hidden2)
        self.covariance = Covariance()
        self.Down_Sample = SPDTransform(hidden1+hidden2, 32)
        self.DTI_SPD = SPD(32, 1e-6)
        self.device = device
        self.layer_gnn = layer_gnn
        self.classfier = SPD_LeNet(32, 2, n_layers)
        self.act = nn.ReLU()
        self.GELU = nn.GELU()
        # for p in self.parameters():
        #     p.requires_grad = False

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = self.gnn_act(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return xs

    def ReSize(self, feature, N):
        molecule_ST = torch.zeros((N, 100, self.C), device=self.device)
        for i in range(N):
            C_L = feature[i].shape[0]
            if C_L >= 100:
                molecule_ST[i, :, :] = feature[i][0:100, :]
            else:
                molecule_ST[i, :C_L, :] = feature[i]
        return molecule_ST

    def forward(self, inputs):
        """Data loading"""
        molecule_smiles, molecule_atoms, molecule_adjs, proteins, protein_LM, protein_SPD, molecule_LM = inputs
        N = molecule_smiles.shape[0]

        """Protein feature extractor"""
        proteins_LM = self.encoder_protein_LM(protein_LM.permute(0, 2, 1)).permute(0, 2, 1)
        proteins_SPD = self.encoder_protein_SPD(protein_SPD.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        proteins = torch.cat((proteins_LM, proteins_SPD), 2)

        """Drug feature extractor"""
        molecule_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        molecule_vec = self.atom_rep(molecule_atoms, N)
        molecule_ST = self.gnn(molecule_vec, molecule_adjs, self.layer_gnn)
        molecule_ST = self.ReSize(molecule_ST, N)
        molecule_ST = self.cross_attention(molecule_ST, molecule_LM)
        molecules = torch.cat((molecule_LM, molecule_ST), 2)

        """SPD-based joint feature and classfier"""
        Joint_Feature = self.dt_joint(proteins, molecules)
        DTI_Feature = self.covariance(Joint_Feature.permute(0, 2, 1))
        DTI_Feature = self.Down_Sample(DTI_Feature)
        DTI_Feature = self.DTI_SPD(DTI_Feature)
        DTI, dti_feature = self.classfier(DTI_Feature)

        return DTI

    def predict(self, res):
        if res > 0.5:
            result = 1
        else:
            result = 0
        return result

    def __call__(self, data, epoch=1, train=True):
        inputs, correct_interaction, SID = data[:-2], data[-2], data[-1]
        correct_interaction = torch.LongTensor(correct_interaction.to('cpu').numpy()).cuda()
        protein_drug_interaction = self.forward(inputs)  # , dis_invariant
        if train:
            loss = F.cross_entropy(protein_drug_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction  # .to('cpu').data.numpy().reshape(-1)
            ys = F.softmax(protein_drug_interaction, 1)  # .to('cpu').data.numpy() C
            return correct_labels, ys

# 0.9635189007608873	0.9635574890097556

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from .Informer_block import AttentionLayer, ProbAttention
from .dt_spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified

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

    def forward(self, x):
        # xn = self.bn1(x)
        xn = self.relu(x)
        xn = self.conv1(xn)
        xn = self.bn2(xn)
        xn = self.relu(xn)
        xn = self.conv2(xn)
        return torch.cat([x, xn], 1)


class Encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=15, pad2=7):
        super(Encoder, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv1d(in_channels=inc, out_channels=inc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1,
                                 bias=False)
        self.dense_cnn = nn.ModuleList(
            [EncoderLayer(inc + growth_rate * i_la, inc + (growth_rate // 2) * i_la, growth_rate, groups, pad2) for i_la
             in range(layers)])
        self.conv_out = nn.Conv1d(in_channels=inc + growth_rate * layers, out_channels=outc, kernel_size=(pad1 * 2 + 1),
                                  stride=1, padding=pad1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.layers):
            x = self.dense_cnn[i](x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        return x

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.QK = nn.Linear(in_features, out_features).to("cuda")
        self.V = nn.Linear(in_features, out_features).to("cuda")
        self.drop = nn.Dropout(0.1)
        self.act = nn.GELU()

    def forward(self, inp, adj):
        h_qk = self.QK(inp)
        h_v = self.act(self.V(inp))
        a_input = torch.matmul(h_qk, h_qk.permute(0,2,1))
        scale = a_input.size(-1) ** -0.5
        attention_adj = torch.sigmoid(a_input * scale) * adj
        h_prime = torch.matmul(attention_adj, h_v)
        return h_prime

class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, gnn_layer):
        super(GAT, self).__init__()
        self.attentions = [GATLayer(n_feat, n_hid) for _ in
                           range(gnn_layer)]
        self.gnn_layer = gnn_layer

    def forward(self, x, adj):
        for i in range(self.gnn_layer):
            x = self.attentions[i](x, adj) + x
        return x

class SPDVectorize(nn.Module):
    def __init__(self, input_size):
        super(SPDVectorize, self).__init__()
        row_idx, col_idx = np.triu_indices(input_size)
        self.register_buffer('row_idx', torch.LongTensor(row_idx))
        self.register_buffer('col_idx', torch.LongTensor(col_idx))

    def forward(self, input):
        output = input[:, self.row_idx, self.col_idx]
        return output

class Atom_rep(nn.Module):
    def __init__(self, channels, device, atom_classes=16, atom_hidden=33):
        super(Atom_rep, self).__init__()
        self.embed_comg = nn.Embedding(atom_classes, atom_hidden)
        self.device = device
        self.channel = channels

    def forward(self, molecule_atoms, N):
        molecule_vec = torch.zeros((molecule_atoms.shape[0], molecule_atoms.shape[1], self.channel), device=self.device)
        for i in range(N):
            fea = torch.zeros((molecule_atoms.shape[1], self.channel), device=self.device)
            atom_fea = molecule_atoms[i][:, 0:16]
            p = torch.argmax(atom_fea, dim=1)
            com = self.embed_comg(p)
            oth1 = molecule_atoms[i][:, 44:75]
            tf = F.normalize(oth1, dim=1)
            fea[:, 0:33] = com
            fea[:, 33:64] = tf
            molecule_vec[i, :, :] = fea
        return molecule_vec

class cross_attention(nn.Module):
    def __init__(self, hidden1, hidden2, dropout):
        super(cross_attention, self).__init__()
        self.W_q = nn.Linear(hidden2, hidden2)
        self.W_k = nn.Linear(hidden1, hidden2)
        self.W_v = nn.Linear(hidden1, hidden2)
        self.drop = nn.Dropout(p=dropout)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(1)

    def forward(self, xs, x):
        q = self.W_q(xs)
        k = self.W_k(x)
        v = self.W_v(x)
        weight = torch.matmul(q, k.permute(0, 2, 1))
        scale = weight.size(-1) ** -0.5
        weights = self.softmax(self.drop(weight * scale))
        ys = torch.matmul(weights, v) + xs
        return ys

class Smooth_loss(nn.Module):

    def __init__(self, smoothing=0.1):
        super(Smooth_loss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        confidence = 1 - self.smoothing
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Covariance(nn.Module):

    def __init__(self, append_mean=False, epsilon=1e-5):
        super(Covariance, self).__init__()
        self.append_mean = append_mean
        self.epsilon = epsilon  # 小的常数用于添加到协方差矩阵的对角线上

    def forward(self, input):
        mean = torch.mean(input, 2, keepdim=True)
        x = input - mean.expand(-1, -1, input.size(2))
        output = torch.bmm(x, x.transpose(1, 2)) / input.size(1)

        if self.append_mean:
            mean_sq = torch.bmm(mean, mean.transpose(1, 2))
            output.add_(mean_sq)
            output = torch.cat((output, mean), 2)
            one = input.new(1, 1, 1).fill_(1).expand(mean.size(0), -1, -1)
            mean = torch.cat((mean, one), 1).transpose(1, 2)
            output = torch.cat((output, mean), 1)

        return output

class SPD(nn.Module):
    def __init__(self, size, epsilon):
        super(SPD, self).__init__()
        self.rect = SPDRectified().cpu()
        self.ST_tangent = SPDTangentSpace(size, vectorize=False).cpu()
        self.epsilon = epsilon

    def add_epsilon(self, output):
        I = torch.eye(output.size(1)).expand_as(output).cpu()
        output += I * self.epsilon
        return output

    def forward(self, h_co):
        h_co = h_co.cpu()
        h_co = self.add_epsilon(h_co)
        h_vec_co_Riemannian = self.rect(h_co)
        h_hidden = self.ST_tangent(h_vec_co_Riemannian)
        return h_hidden.to("cuda")

class Reduce_dimension(nn.Module):
    def __init__(self,):
        super(Reduce_dimension, self).__init__()
        self.trans = nn.Linear(2, 1, bias=False)

    def forward(self, input):
        B, C = input.shape[0], input.shape[-1]
        input_r = input.view(B, 600, 2, C).permute(0, 1, 3, 2)
        input_r = self.trans(input_r).squeeze(-1)
        return input_r

class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b

class Feature_Joint(nn.Module):
    def __init__(self, dim_a, dim_b, out_dim):
        super(Feature_Joint, self).__init__()
        self.map = nn.Linear(dim_a + dim_b, out_dim)
        self.GELU = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, feature_a, feature_b):
        zeros_a = torch.zeros((feature_a.shape[0], feature_a.shape[1], feature_b.shape[2])).to(feature_a.device)
        zeros_b = torch.zeros((feature_b.shape[0], feature_b.shape[1], feature_a.shape[2])).to(feature_b.device)
        feature_a_expand = torch.cat((feature_a, zeros_a), 2)
        feature_b_expand = torch.cat((zeros_b, feature_b), 2)
        joint_feature = torch.cat((feature_a_expand, feature_b_expand), 1)
        joint_feature = self.GELU(self.map(joint_feature))
        return self.drop(joint_feature)

class SPD_LeNet(nn.Module):
    def __init__(self, hidden, classes, layers):
        super(SPD_LeNet, self).__init__()
        self.vec = SPDVectorize(hidden)
        self.FC_down = nn.Linear(int((hidden+1)*hidden/2), 512)
        self.FC_combs = nn.ModuleList([nn.Linear(512, 512) for _ in range(layers)])
        self.FC_out = nn.Linear(512, classes)
        self.layers = layers
        self.act = nn.ReLU()

    def forward(self, dti_spd):
        dti_vec = self.vec(dti_spd)  # self.BN(dti_feature.permute(0, 2, 1))
        dti_feature = self.FC_down(dti_vec)
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti = self.FC_out(dti_feature)
        return dti, dti_feature

class SPD_DTI(nn.Module):
    def __init__(self, layer_gnn, device, hidden1=192, hidden2=64, n_layers=3, dropout=0.05):
        super(SPD_DTI, self).__init__()
        '''Drug'''
        self.encoder_drug = Encoder(768, hidden1, 64, 3, groups=32, pad1=7, pad2=3)
        self.atom_rep = Atom_rep(hidden2, device)
        # self.W_gnn = nn.ModuleList([nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2)])
        self.GAT = GAT(hidden2, hidden2, n_layers).to("cuda")
        self.gnn_act = nn.GELU()
        self.C = hidden2
        self.cross_attention = cross_attention(hidden1, hidden2, dropout)
        # self.d_joint = Feature_Joint(hidden1, hidden2, hidden1 + hidden2)

        '''Protein'''
        self.encoder_protein_LM = Encoder(1024, hidden1, 128, 3, groups=64, pad1=7, pad2=3)
        self.encoder_protein_SPD = Encoder(120, hidden2, 60, 5, groups=15, pad1=3, pad2=3)
        self.R_D = Reduce_dimension()
        self.softmax_p = nn.Softmax(-1)

        '''DECISION'''
        self.dt_joint = Feature_Joint(hidden1+hidden2, hidden1+hidden2, hidden1+hidden2)
        self.covariance = Covariance()
        self.Down_Sample = SPDTransform(hidden1+hidden2, 32)
        self.DTI_SPD = SPD(32, 1e-6)
        self.device = device
        self.layer_gnn = layer_gnn
        self.classfier = SPD_LeNet(32, 2, n_layers)
        self.act = nn.ReLU()
        self.GELU = nn.GELU()
        # for p in self.parameters():
        #     p.requires_grad = False

    # def gnn(self, xs, A, layer):
    #     for i in range(layer):
    #         hs = self.gnn_act(self.W_gnn[i](xs))
    #         xs = xs + torch.matmul(A, hs)
    #     return xs

    def ReSize(self, feature, N):
        molecule_ST = torch.zeros((N, 100, self.C), device=self.device)
        for i in range(N):
            C_L = feature[i].shape[0]
            if C_L >= 100:
                molecule_ST[i, :, :] = feature[i][0:100, :]
            else:
                molecule_ST[i, :C_L, :] = feature[i]
        return molecule_ST

    def forward(self, inputs):
        """Data loading"""
        molecule_smiles, molecule_atoms, molecule_adjs, proteins, protein_LM, protein_SPD, molecule_LM = inputs
        N = molecule_smiles.shape[0]

        """Protein feature extractor"""
        proteins_LM = self.encoder_protein_LM(protein_LM.permute(0, 2, 1)).permute(0, 2, 1)
        proteins_SPD = self.encoder_protein_SPD(protein_SPD.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        proteins = torch.cat((proteins_LM, proteins_SPD), 2)

        """Drug feature extractor"""
        molecule_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        molecule_vec = self.atom_rep(molecule_atoms, N)
        molecule_ST = self.GAT(molecule_vec, molecule_adjs)
        molecule_ST = self.ReSize(molecule_ST, N)
        molecule_ST = self.cross_attention(molecule_ST, molecule_LM)
        molecules = torch.cat((molecule_LM, molecule_ST), 2)

        """SPD-based joint feature and classfier"""
        Joint_Feature = self.dt_joint(proteins, molecules)
        DTI_Feature = self.covariance(Joint_Feature.permute(0, 2, 1))
        DTI_Feature = self.Down_Sample(DTI_Feature)
        DTI_Feature = self.DTI_SPD(DTI_Feature)
        DTI, dti_feature = self.classfier(DTI_Feature)
        return DTI

    def __call__(self, data, epoch=1, train=True):
        inputs, correct_interaction, SID = data[:-2], data[-2], data[-1]
        correct_interaction = torch.LongTensor(correct_interaction.to('cpu').numpy()).cuda()
        protein_drug_interaction = self.forward(inputs)  # , dis_invariant
        if train:
            loss = F.cross_entropy(protein_drug_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction  # .to('cpu').data.numpy().reshape(-1)
            ys = F.softmax(protein_drug_interaction, 1)  # .to('cpu').data.numpy() C
            return correct_labels, ys

# 96.1

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from .Informer_block import AttentionLayer, ProbAttention
from .dt_spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified

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

    def forward(self, x):
        # xn = self.bn1(x)
        xn = self.relu(x)
        xn = self.conv1(xn)
        xn = self.bn2(xn)
        xn = self.relu(xn)
        xn = self.conv2(xn)
        return torch.cat([x, xn], 1)


class Encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=15, pad2=7):
        super(Encoder, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv1d(in_channels=inc, out_channels=inc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1,
                                 bias=False)
        self.dense_cnn = nn.ModuleList(
            [EncoderLayer(inc + growth_rate * i_la, inc + (growth_rate // 2) * i_la, growth_rate, groups, pad2) for i_la
             in range(layers)])
        self.conv_out = nn.Conv1d(in_channels=inc + growth_rate * layers, out_channels=outc, kernel_size=(pad1 * 2 + 1),
                                  stride=1, padding=pad1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.layers):
            x = self.dense_cnn[i](x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        return x

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.QK = nn.Linear(in_features, out_features).to("cuda")
        self.V = nn.Linear(in_features, out_features).to("cuda")
        self.drop = nn.Dropout(0.1)
        self.act = nn.GELU()

    def forward(self, inp, adj):
        h_qk = self.QK(inp)
        h_v = self.act(self.V(inp))
        a_input = torch.matmul(h_qk, h_qk.permute(0,2,1))
        scale = a_input.size(-1) ** -0.5
        attention_adj = torch.sigmoid(a_input * scale) * adj
        h_prime = torch.matmul(attention_adj, h_v)
        return h_prime

class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, gnn_layer):
        super(GAT, self).__init__()
        self.attentions = [GATLayer(n_feat, n_hid) for _ in
                           range(gnn_layer)]
        self.gnn_layer = gnn_layer

    def forward(self, x, adj):
        for i in range(self.gnn_layer):
            x = self.attentions[i](x, adj) + x
        return x

class SPDVectorize(nn.Module):
    def __init__(self, input_size):
        super(SPDVectorize, self).__init__()
        row_idx, col_idx = np.triu_indices(input_size)
        self.register_buffer('row_idx', torch.LongTensor(row_idx))
        self.register_buffer('col_idx', torch.LongTensor(col_idx))

    def forward(self, input):
        output = input[:, self.row_idx, self.col_idx]
        return output

class Atom_rep(nn.Module):
    def __init__(self, channels, device, atom_classes=16, atom_hidden=33):
        super(Atom_rep, self).__init__()
        self.embed_comg = nn.Embedding(atom_classes, atom_hidden)
        self.device = device
        self.channel = channels

    def forward(self, molecule_atoms, N):
        molecule_vec = torch.zeros((molecule_atoms.shape[0], molecule_atoms.shape[1], self.channel), device=self.device)
        for i in range(N):
            fea = torch.zeros((molecule_atoms.shape[1], self.channel), device=self.device)
            atom_fea = molecule_atoms[i][:, 0:16]
            p = torch.argmax(atom_fea, dim=1)
            com = self.embed_comg(p)
            oth1 = molecule_atoms[i][:, 44:75]
            tf = F.normalize(oth1, dim=1)
            fea[:, 0:33] = com
            fea[:, 33:64] = tf
            molecule_vec[i, :, :] = fea
        return molecule_vec

class cross_attention(nn.Module):
    def __init__(self, hidden1, hidden2, dropout):
        super(cross_attention, self).__init__()
        self.W_q = nn.Linear(hidden2, hidden2)
        self.W_k = nn.Linear(hidden1, hidden2)
        self.W_v = nn.Linear(hidden1, hidden2)
        self.drop = nn.Dropout(p=dropout)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(1)

    def forward(self, xs, x):
        q = self.W_q(xs)
        k = self.W_k(x)
        v = self.W_v(x)
        weight = torch.matmul(q, k.permute(0, 2, 1))
        scale = weight.size(-1) ** -0.5
        weights = self.softmax(self.drop(weight * scale))
        ys = torch.matmul(weights, v) + xs
        return ys

class Smooth_loss(nn.Module):

    def __init__(self, smoothing=0.1):
        super(Smooth_loss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        confidence = 1 - self.smoothing
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Covariance(nn.Module):

    def __init__(self, append_mean=False, epsilon=1e-5):
        super(Covariance, self).__init__()
        self.append_mean = append_mean
        self.epsilon = epsilon  # 小的常数用于添加到协方差矩阵的对角线上

    def forward(self, input):
        mean = torch.mean(input, 2, keepdim=True)
        x = input - mean.expand(-1, -1, input.size(2))
        output = torch.bmm(x, x.transpose(1, 2)) / input.size(1)

        if self.append_mean:
            mean_sq = torch.bmm(mean, mean.transpose(1, 2))
            output.add_(mean_sq)
            output = torch.cat((output, mean), 2)
            one = input.new(1, 1, 1).fill_(1).expand(mean.size(0), -1, -1)
            mean = torch.cat((mean, one), 1).transpose(1, 2)
            output = torch.cat((output, mean), 1)

        return output

class SPD(nn.Module):
    def __init__(self, size, epsilon):
        super(SPD, self).__init__()
        self.rect = SPDRectified().cpu()
        self.ST_tangent = SPDTangentSpace(size, vectorize=False).cpu()
        self.epsilon = epsilon

    def add_epsilon(self, output):
        I = torch.eye(output.size(1)).expand_as(output).cpu()
        output += I * self.epsilon
        return output

    def forward(self, h_co):
        h_co = h_co.cpu()
        h_co = self.add_epsilon(h_co)
        h_vec_co_Riemannian = self.rect(h_co)
        h_hidden = self.ST_tangent(h_vec_co_Riemannian)
        return h_hidden.to("cuda")

class Reduce_dimension(nn.Module):
    def __init__(self,):
        super(Reduce_dimension, self).__init__()
        self.trans = nn.Linear(2, 1, bias=False)

    def forward(self, input):
        B, C = input.shape[0], input.shape[-1]
        input_r = input.view(B, 600, 2, C).permute(0, 1, 3, 2)
        input_r = self.trans(input_r).squeeze(-1)
        return input_r

class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b

class Feature_Joint(nn.Module):
    def __init__(self, dim_a, dim_b, out_dim):
        super(Feature_Joint, self).__init__()
        self.map = nn.Linear(dim_a + dim_b, out_dim)
        self.GELU = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, feature_a, feature_b):
        zeros_a = torch.zeros((feature_a.shape[0], feature_a.shape[1], feature_b.shape[2])).to(feature_a.device)
        zeros_b = torch.zeros((feature_b.shape[0], feature_b.shape[1], feature_a.shape[2])).to(feature_b.device)
        feature_a_expand = torch.cat((feature_a, zeros_a), 2)
        feature_b_expand = torch.cat((zeros_b, feature_b), 2)
        joint_feature = torch.cat((feature_a_expand, feature_b_expand), 1)
        joint_feature = self.GELU(self.map(joint_feature))
        return self.drop(joint_feature)

class Ancillary_LeNet(nn.Module):
    def __init__(self, hidden, classes, layers):
        super(Ancillary_LeNet, self).__init__()
        self.CNNs = nn.ModuleList(
            [nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=7, padding=3) for _ in range(layers)])
        self.BN = nn.BatchNorm1d(hidden)  # nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])

        self.FC_combs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.FC_down = nn.Linear(hidden, 128)
        self.FC_out = nn.Linear(128, classes)
        self.layers = layers
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, dti_feature):
        dti_feature = dti_feature.permute(0, 2, 1)  # self.BN(dti_feature.permute(0, 2, 1))
        for i in range(self.layers):
            dti_feature = self.act(self.CNNs[i](dti_feature)) + dti_feature
        dti_feature = dti_feature.permute(0, 2, 1)
        dti_feature = torch.mean(dti_feature, dim=1)
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti_feature = self.FC_down(dti_feature)
        dti = self.FC_out(dti_feature)
        return dti

class SPD_LeNet(nn.Module):
    def __init__(self, hidden, classes, layers):
        super(SPD_LeNet, self).__init__()
        self.vec = SPDVectorize(hidden)
        self.FC_down = nn.Linear(int((hidden+1)*hidden/2), 512)
        self.FC_combs = nn.ModuleList([nn.Linear(512, 512) for _ in range(layers)])
        self.FC_out = nn.Linear(512, classes)
        self.layers = layers
        self.act = nn.ReLU()

    def forward(self, dti_spd):
        dti_vec = self.vec(dti_spd)  # self.BN(dti_feature.permute(0, 2, 1))
        dti_feature = self.FC_down(dti_vec)
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti = self.FC_out(dti_feature)
        return dti, dti_feature

class SPD_DTI(nn.Module):
    def __init__(self, layer_gnn, device, hidden1=192, hidden2=64, n_layers=3, dropout=0.05):
        super(SPD_DTI, self).__init__()
        '''Drug'''
        self.encoder_drug = Encoder(768, hidden1, 64, 3, groups=32, pad1=7, pad2=3)
        self.atom_rep = Atom_rep(hidden2, device)
        # self.W_gnn = nn.ModuleList([nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2)])
        self.GAT = GAT(hidden2, hidden2, n_layers).to("cuda")
        self.gnn_act = nn.GELU()
        self.C = hidden2
        self.cross_attention = cross_attention(hidden1, hidden2, dropout)
        # self.d_joint = Feature_Joint(hidden1, hidden2, hidden1 + hidden2)

        '''Protein'''
        self.encoder_protein_LM = Encoder(1024, hidden1, 128, 3, groups=64, pad1=7, pad2=3)
        self.encoder_protein_SPD = Encoder(120, hidden2, 60, 5, groups=15, pad1=3, pad2=3)
        self.R_D = Reduce_dimension()

        '''DECISION'''
        self.dt_joint = Feature_Joint(hidden1+hidden2, hidden1+hidden2, hidden1+hidden2)
        self.covariance = Covariance()
        self.Down_Sample = SPDTransform(hidden1+hidden2, 32)
        self.DTI_SPD = SPD(32, 1e-6)
        self.device = device
        self.layer_gnn = layer_gnn
        self.classfier = SPD_LeNet(32, 2, n_layers)
        self.a_c = Ancillary_LeNet(hidden1+hidden2, 2, n_layers)
        self.act = nn.ReLU()
        self.GELU = nn.GELU()
        # for p in self.parameters():
        #     p.requires_grad = False

    def ReSize(self, feature, N):
        molecule_ST = torch.zeros((N, 100, self.C), device=self.device)
        for i in range(N):
            C_L = feature[i].shape[0]
            if C_L >= 100:
                molecule_ST[i, :, :] = feature[i][0:100, :]
            else:
                molecule_ST[i, :C_L, :] = feature[i]
        return molecule_ST

    def forward(self, inputs):
        """Data loading"""
        molecule_smiles, molecule_atoms, molecule_adjs, proteins, protein_LM, protein_SPD, molecule_LM = inputs
        N = molecule_smiles.shape[0]

        """Protein feature extractor"""
        proteins_LM = self.encoder_protein_LM(protein_LM.permute(0, 2, 1)).permute(0, 2, 1)
        proteins_SPD = self.encoder_protein_SPD(protein_SPD.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        proteins = torch.cat((proteins_LM, proteins_SPD), 2)

        """Drug feature extractor"""
        molecule_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        molecule_vec = self.atom_rep(molecule_atoms, N)
        molecule_ST = self.GAT(molecule_vec, molecule_adjs)
        molecule_ST = self.ReSize(molecule_ST, N)
        molecule_ST = self.cross_attention(molecule_ST, molecule_LM)
        molecules = torch.cat((molecule_LM, molecule_ST), 2)

        """SPD-based joint feature and classfier"""
        Joint_Feature = self.dt_joint(proteins, molecules)
        ancillary = self.a_c(Joint_Feature)
        DTI_Feature = self.covariance(Joint_Feature.permute(0, 2, 1))
        DTI_Feature = self.Down_Sample(DTI_Feature)
        DTI_Feature = self.DTI_SPD(DTI_Feature)
        DTI, dti_feature = self.classfier(DTI_Feature)
        return DTI, ancillary

    def __call__(self, data, epoch=1, train=True):
        inputs, correct_interaction, SID = data[:-2], data[-2], data[-1]
        correct_interaction = torch.LongTensor(correct_interaction.to('cpu').numpy()).cuda()
        protein_drug_interaction, ancillary = self.forward(inputs)  # , dis_invariant
        t = int(epoch / 2)
        alpha = 0.4 * (15 / (t + 15))
        if train:
            loss = F.cross_entropy(protein_drug_interaction, correct_interaction)
            loss2 = F.cross_entropy(ancillary, correct_interaction)
            return loss * (0.7 - alpha) + loss2 * (0.3 + alpha)
        else:
            correct_labels = correct_interaction  # .to('cpu').data.numpy().reshape(-1)
            ys = F.softmax(protein_drug_interaction, 1)  # .to('cpu').data.numpy() C
            return correct_labels, ys

# 0.9636400526571405	0.9649562856935733

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from .Informer_block import AttentionLayer, ProbAttention
from .dt_spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified

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

    def forward(self, x):
        # xn = self.bn1(x)
        xn = self.relu(x)
        xn = self.conv1(xn)
        xn = self.bn2(xn)
        xn = self.relu(xn)
        xn = self.conv2(xn)
        return torch.cat([x, xn], 1)


class Encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=15, pad2=7):
        super(Encoder, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv1d(in_channels=inc, out_channels=inc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1,
                                 bias=False)
        self.dense_cnn = nn.ModuleList(
            [EncoderLayer(inc + growth_rate * i_la, inc + (growth_rate // 2) * i_la, growth_rate, groups, pad2) for i_la
             in range(layers)])
        self.conv_out = nn.Conv1d(in_channels=inc + growth_rate * layers, out_channels=outc, kernel_size=(pad1 * 2 + 1),
                                  stride=1, padding=pad1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.layers):
            x = self.dense_cnn[i](x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        return x

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.QK = nn.Linear(in_features, out_features).to("cuda")
        self.V = nn.Linear(in_features, out_features).to("cuda")
        self.drop = nn.Dropout(0.1)
        self.act = nn.GELU()

    def forward(self, inp, adj):
        h_qk = self.QK(inp)
        h_v = self.act(self.V(inp))
        a_input = torch.matmul(h_qk, h_qk.permute(0,2,1))
        scale = h_qk.size(-1) ** -0.5
        attention_adj = torch.sigmoid(a_input * scale) * adj
        h_prime = torch.matmul(attention_adj, h_v)
        return h_prime

class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, gnn_layer):
        super(GAT, self).__init__()
        self.attentions = [GATLayer(n_feat, n_hid) for _ in
                           range(gnn_layer)]
        self.gnn_layer = gnn_layer

    def forward(self, x, adj):
        for i in range(self.gnn_layer):
            x = self.attentions[i](x, adj) + x
        return x

class SPDVectorize(nn.Module):
    def __init__(self, input_size):
        super(SPDVectorize, self).__init__()
        row_idx, col_idx = np.triu_indices(input_size)
        self.register_buffer('row_idx', torch.LongTensor(row_idx))
        self.register_buffer('col_idx', torch.LongTensor(col_idx))

    def forward(self, input):
        output = input[:, self.row_idx, self.col_idx]
        return output

class Atom_rep(nn.Module):
    def __init__(self, channels, device, atom_classes=16, atom_hidden=33):
        super(Atom_rep, self).__init__()
        self.embed_comg = nn.Embedding(atom_classes, atom_hidden)
        self.device = device
        self.channel = channels

    def forward(self, molecule_atoms, N):
        molecule_vec = torch.zeros((molecule_atoms.shape[0], molecule_atoms.shape[1], self.channel), device=self.device)
        for i in range(N):
            fea = torch.zeros((molecule_atoms.shape[1], self.channel), device=self.device)
            atom_fea = molecule_atoms[i][:, 0:16]
            p = torch.argmax(atom_fea, dim=1)
            com = self.embed_comg(p)
            oth1 = molecule_atoms[i][:, 44:75]
            tf = F.normalize(oth1, dim=1)
            fea[:, 0:33] = com
            fea[:, 33:64] = tf
            molecule_vec[i, :, :] = fea
        return molecule_vec

class cross_attention(nn.Module):
    def __init__(self, hidden1, hidden2, dropout):
        super(cross_attention, self).__init__()
        self.W_q = nn.Linear(hidden1, hidden2)
        self.W_k = nn.Linear(hidden2, hidden2)
        self.W_v = nn.Linear(hidden2, hidden2)
        self.drop = nn.Dropout(p=0.1)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(-1)

    def forward(self, xs, x):
        q = self.W_q(x)
        k = self.W_k(xs)
        v = self.W_v(xs)
        weight = torch.matmul(q, k.permute(0, 2, 1))
        scale = weight.size(-1) ** -0.5
        weights = self.softmax(weight * scale)
        ys = torch.matmul(self.drop(weights), v) + xs
        return ys

class Smooth_loss(nn.Module):

    def __init__(self, smoothing=0.1):
        super(Smooth_loss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        confidence = 1 - self.smoothing
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Covariance(nn.Module):

    def __init__(self, append_mean=False, epsilon=1e-5):
        super(Covariance, self).__init__()
        self.append_mean = append_mean
        self.epsilon = epsilon  # 小的常数用于添加到协方差矩阵的对角线上

    def forward(self, input):
        mean = torch.mean(input, 2, keepdim=True)
        x = input - mean.expand(-1, -1, input.size(2))
        output = torch.bmm(x, x.transpose(1, 2)) / input.size(1)

        if self.append_mean:
            mean_sq = torch.bmm(mean, mean.transpose(1, 2))
            output.add_(mean_sq)
            output = torch.cat((output, mean), 2)
            one = input.new(1, 1, 1).fill_(1).expand(mean.size(0), -1, -1)
            mean = torch.cat((mean, one), 1).transpose(1, 2)
            output = torch.cat((output, mean), 1)

        return output

class SPD(nn.Module):
    def __init__(self, size, epsilon):
        super(SPD, self).__init__()
        self.rect = SPDRectified().cpu()
        self.ST_tangent = SPDTangentSpace(size, vectorize=False).cpu()
        self.epsilon = epsilon

    def add_epsilon(self, output):
        I = torch.eye(output.size(1)).expand_as(output).cpu()
        output += I * self.epsilon
        return output

    def forward(self, h_co):
        h_co = h_co.cpu()
        h_co = self.add_epsilon(h_co)
        h_vec_co_Riemannian = self.rect(h_co)
        h_hidden = self.ST_tangent(h_vec_co_Riemannian)
        return h_hidden.to("cuda")

class Reduce_dimension(nn.Module):
    def __init__(self,):
        super(Reduce_dimension, self).__init__()
        self.trans = nn.Linear(2, 1, bias=False)

    def forward(self, input):
        B, C = input.shape[0], input.shape[-1]
        input_r = input.view(B, 600, 2, C).permute(0, 1, 3, 2)
        input_r = self.trans(input_r).squeeze(-1)
        return input_r

class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b

class Feature_Joint(nn.Module):
    def __init__(self, dim_a, dim_b, out_dim):
        super(Feature_Joint, self).__init__()
        self.map = nn.Linear(dim_a + dim_b, out_dim)
        self.GELU = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, feature_a, feature_b):
        zeros_a = torch.zeros((feature_a.shape[0], feature_a.shape[1], feature_b.shape[2])).to(feature_a.device)
        zeros_b = torch.zeros((feature_b.shape[0], feature_b.shape[1], feature_a.shape[2])).to(feature_b.device)
        feature_a_expand = torch.cat((feature_a, zeros_a), 2)
        feature_b_expand = torch.cat((zeros_b, feature_b), 2)
        joint_feature = torch.cat((feature_a_expand, feature_b_expand), 1)
        joint_feature = self.GELU(self.map(joint_feature))
        return self.drop(joint_feature)

class Ancillary_LeNet(nn.Module):
    def __init__(self, hidden, classes, layers):
        super(Ancillary_LeNet, self).__init__()
        self.CNNs = nn.ModuleList(
            [nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=7, padding=3) for _ in range(layers)])
        self.BN = nn.BatchNorm1d(hidden)  # nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])

        self.FC_combs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.FC_down = nn.Linear(hidden, 128)
        self.FC_out = nn.Linear(128, classes)
        self.layers = layers
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, dti_feature):
        dti_feature = dti_feature.permute(0, 2, 1)  # self.BN(dti_feature.permute(0, 2, 1))
        for i in range(self.layers):
            dti_feature = self.act(self.CNNs[i](dti_feature)) + dti_feature
        dti_feature = dti_feature.permute(0, 2, 1)
        dti_feature = torch.mean(dti_feature, dim=1)
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti_feature = self.FC_down(dti_feature)
        dti = self.FC_out(dti_feature)
        return dti

class SPD_LeNet(nn.Module):
    def __init__(self, hidden, classes, layers):
        super(SPD_LeNet, self).__init__()
        self.vec = SPDVectorize(hidden)
        self.FC_down = nn.Linear(int((hidden+1)*hidden/2), 512)
        self.FC_combs = nn.ModuleList([nn.Linear(512, 512) for _ in range(layers)])
        self.FC_out = nn.Linear(512, classes)
        self.layers = layers
        self.act = nn.ReLU()

    def forward(self, dti_spd):
        dti_vec = self.vec(dti_spd)  # self.BN(dti_feature.permute(0, 2, 1))
        dti_feature = self.FC_down(dti_vec)
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti = self.FC_out(dti_feature)
        return dti, dti_feature

class SPD_DTI(nn.Module):
    def __init__(self, layer_gnn, device, hidden1=192, hidden2=64, n_layers=3, dropout=0.05):
        super(SPD_DTI, self).__init__()
        '''Drug'''
        self.encoder_drug = Encoder(768, hidden1, 64, 3, groups=32, pad1=7, pad2=3)
        self.atom_rep = Atom_rep(hidden2, device)
        # self.W_gnn = nn.ModuleList([nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2)])
        self.GAT = GAT(hidden2, hidden2, n_layers).to("cuda")
        self.gnn_act = nn.GELU()
        self.C = hidden2
        self.cross_attention = cross_attention(hidden1, hidden2, dropout)
        # self.d_joint = Feature_Joint(hidden1, hidden2, hidden1 + hidden2)

        '''Protein'''
        self.encoder_protein_LM = Encoder(1024, hidden1, 128, 3, groups=64, pad1=7, pad2=3)
        self.encoder_protein_SPD = Encoder(120, hidden2, 60, 5, groups=15, pad1=3, pad2=3)
        self.R_D = Reduce_dimension()

        '''DECISION'''
        self.dt_joint = Feature_Joint(hidden1+hidden2, hidden1+hidden2, hidden1+hidden2)
        self.covariance = Covariance()
        self.Down_Sample = SPDTransform(hidden1+hidden2, 32)
        self.DTI_SPD = SPD(32, 1e-6)
        self.device = device
        self.layer_gnn = layer_gnn
        self.classfier = SPD_LeNet(32, 2, n_layers)
        self.a_c = Ancillary_LeNet(hidden1+hidden2, 2, n_layers)
        self.act = nn.ReLU()
        self.GELU = nn.GELU()
        # for p in self.parameters():
        #     p.requires_grad = False

    def ReSize(self, feature, N):
        molecule_ST = torch.zeros((N, 100, self.C), device=self.device)
        for i in range(N):
            C_L = feature[i].shape[0]
            if C_L >= 100:
                molecule_ST[i, :, :] = feature[i][0:100, :]
            else:
                molecule_ST[i, :C_L, :] = feature[i]
        return molecule_ST

    def forward(self, inputs):
        """Data loading"""
        molecule_smiles, molecule_atoms, molecule_adjs, proteins, protein_LM, protein_SPD, molecule_LM = inputs
        N = molecule_smiles.shape[0]

        """Protein feature extractor"""
        proteins_LM = self.encoder_protein_LM(protein_LM.permute(0, 2, 1)).permute(0, 2, 1)
        proteins_SPD = self.encoder_protein_SPD(protein_SPD.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        proteins = torch.cat((proteins_LM, proteins_SPD), 2)

        """Drug feature extractor"""
        molecule_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        molecule_vec = self.atom_rep(molecule_atoms, N)
        molecule_ST = self.GAT(molecule_vec, molecule_adjs)
        molecule_ST = self.ReSize(molecule_ST, N)
        molecule_ST = self.cross_attention(molecule_ST, molecule_LM)
        molecules = torch.cat((molecule_LM, molecule_ST), 2)

        """SPD-based joint feature and classfier"""
        Joint_Feature = self.dt_joint(proteins, molecules)
        ancillary = self.a_c(Joint_Feature)
        DTI_Feature = self.covariance(Joint_Feature.permute(0, 2, 1))
        DTI_Feature = self.Down_Sample(DTI_Feature)
        DTI_Feature = self.DTI_SPD(DTI_Feature)
        DTI, dti_feature = self.classfier(DTI_Feature)
        return DTI, ancillary

    def __call__(self, data, epoch=1, train=True):
        inputs, correct_interaction, SID = data[:-2], data[-2], data[-1]
        correct_interaction = torch.LongTensor(correct_interaction.to('cpu').numpy()).cuda()
        protein_drug_interaction, ancillary = self.forward(inputs)  # , dis_invariant
        t = int(epoch / 2)
        alpha = 0.4 * (15 / (t + 15))
        if train:
            loss = F.cross_entropy(protein_drug_interaction, correct_interaction)
            loss2 = F.cross_entropy(ancillary, correct_interaction)
            return loss * (0.7 - alpha) + loss2 * (0.3 + alpha)
        else:
            correct_labels = correct_interaction  # .to('cpu').data.numpy().reshape(-1)
            ys = F.softmax(protein_drug_interaction, 1)  # .to('cpu').data.numpy() C
            return correct_labels, ys

# 372.1996183922747	0.9632177517616285	0.963261211132985

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from .Informer_block import AttentionLayer, ProbAttention
from .dt_spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified
from torch.autograd import Variable

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

    def forward(self, x):
        # xn = self.bn1(x)
        xn = self.relu(x)
        xn = self.conv1(xn)
        xn = self.bn2(xn)
        xn = self.relu(xn)
        xn = self.conv2(xn)
        return torch.cat([x, xn], 1)


class Encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=15, pad2=7):
        super(Encoder, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv1d(in_channels=inc, out_channels=inc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1,
                                 bias=False)
        self.dense_cnn = nn.ModuleList(
            [EncoderLayer(inc + growth_rate * i_la, inc + (growth_rate // 2) * i_la, growth_rate, groups, pad2) for i_la
             in range(layers)])
        self.conv_out = nn.Conv1d(in_channels=inc + growth_rate * layers, out_channels=outc, kernel_size=(pad1 * 2 + 1),
                                  stride=1, padding=pad1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.layers):
            x = self.dense_cnn[i](x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        return x

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.QK = nn.Linear(in_features, out_features).to("cuda")
        self.V = nn.Linear(in_features, out_features).to("cuda")
        self.drop = nn.Dropout(0.1)
        self.act = nn.GELU()

    def forward(self, inp, adj):
        h_qk = self.QK(inp)
        h_v = self.act(self.V(inp))
        a_input = torch.matmul(h_qk, h_qk.permute(0,2,1))
        scale = h_qk.size(-1) ** -0.5
        attention_adj = torch.sigmoid(a_input * scale) * adj
        h_prime = torch.matmul(attention_adj, h_v)
        return h_prime

class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, gnn_layer):
        super(GAT, self).__init__()
        self.attentions = [GATLayer(n_feat, n_hid) for _ in
                           range(gnn_layer)]
        self.gnn_layer = gnn_layer

    def forward(self, x, adj):
        for i in range(self.gnn_layer):
            x = self.attentions[i](x, adj) + x
        return x

class SPDVectorize(nn.Module):
    def __init__(self, input_size):
        super(SPDVectorize, self).__init__()
        row_idx, col_idx = np.triu_indices(input_size)
        self.register_buffer('row_idx', torch.LongTensor(row_idx))
        self.register_buffer('col_idx', torch.LongTensor(col_idx))

    def forward(self, input):
        output = input[:, self.row_idx, self.col_idx]
        return output

class Atom_rep(nn.Module):
    def __init__(self, channels, device, atom_classes=16, atom_hidden=33):
        super(Atom_rep, self).__init__()
        self.embed_comg = nn.Embedding(atom_classes, atom_hidden)
        self.device = device
        self.channel = channels

    def forward(self, molecule_atoms, N):
        molecule_vec = torch.zeros((molecule_atoms.shape[0], molecule_atoms.shape[1], self.channel), device=self.device)
        for i in range(N):
            fea = torch.zeros((molecule_atoms.shape[1], self.channel), device=self.device)
            atom_fea = molecule_atoms[i][:, 0:16]
            p = torch.argmax(atom_fea, dim=1)
            com = self.embed_comg(p)
            oth1 = molecule_atoms[i][:, 44:75]
            tf = F.normalize(oth1, dim=1)
            fea[:, 0:33] = com
            fea[:, 33:64] = tf
            molecule_vec[i, :, :] = fea
        return molecule_vec

class cross_attention(nn.Module):
    def __init__(self, hidden1, hidden2, dropout):
        super(cross_attention, self).__init__()
        self.W_q = nn.Linear(hidden1, hidden2)
        self.W_k = nn.Linear(hidden2, hidden2)
        self.W_v = nn.Linear(hidden2, hidden2)
        self.drop = nn.Dropout(p=0.1)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(-1)

    def forward(self, xs, x):
        q = self.W_q(x)
        k = self.W_k(xs)
        v = self.W_v(xs)
        weight = torch.matmul(q, k.permute(0, 2, 1))
        scale = weight.size(-1) ** -0.5
        weights = self.softmax(weight * scale)
        ys = torch.matmul(self.drop(weights), v) + xs
        return ys

class Smooth_loss(nn.Module):

    def __init__(self, smoothing=0.1):
        super(Smooth_loss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        confidence = 1 - self.smoothing
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Covariance(nn.Module):

    def __init__(self, append_mean=False, epsilon=1e-5):
        super(Covariance, self).__init__()
        self.append_mean = append_mean
        self.epsilon = epsilon  # 小的常数用于添加到协方差矩阵的对角线上

    def forward(self, input):
        mean = torch.mean(input, 2, keepdim=True)
        x = input - mean.expand(-1, -1, input.size(2))
        output = torch.bmm(x, x.transpose(1, 2)) / input.size(1)

        if self.append_mean:
            mean_sq = torch.bmm(mean, mean.transpose(1, 2))
            output.add_(mean_sq)
            output = torch.cat((output, mean), 2)
            one = input.new(1, 1, 1).fill_(1).expand(mean.size(0), -1, -1)
            mean = torch.cat((mean, one), 1).transpose(1, 2)
            output = torch.cat((output, mean), 1)

        return output

class SPD(nn.Module):
    def __init__(self, size, epsilon):
        super(SPD, self).__init__()
        self.rect = SPDRectified().cpu()
        self.ST_tangent = SPDTangentSpace(size, vectorize=False).cpu()
        self.epsilon = epsilon

    def add_epsilon(self, output):
        I = torch.eye(output.size(1)).expand_as(output).cpu()
        output += I * self.epsilon
        return output

    def forward(self, h_co):
        h_co = h_co.cpu()
        h_co = self.add_epsilon(h_co)
        h_vec_co_Riemannian = self.rect(h_co)
        h_hidden = self.ST_tangent(h_vec_co_Riemannian)
        return h_hidden.to("cuda")

class Reduce_dimension(nn.Module):
    def __init__(self,):
        super(Reduce_dimension, self).__init__()
        self.trans = nn.Linear(2, 1, bias=False)

    def forward(self, input):
        B, C = input.shape[0], input.shape[-1]
        input_r = input.view(B, 600, 2, C).permute(0, 1, 3, 2)
        input_r = self.trans(input_r).squeeze(-1)
        return input_r

class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b

class Feature_Joint(nn.Module):
    def __init__(self, dim_a, dim_b, out_dim):
        super(Feature_Joint, self).__init__()
        self.map = nn.Linear(dim_a + dim_b, out_dim)
        self.GELU = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, feature_a, feature_b):
        zeros_a = torch.zeros((feature_a.shape[0], feature_a.shape[1], feature_b.shape[2])).to(feature_a.device)
        zeros_b = torch.zeros((feature_b.shape[0], feature_b.shape[1], feature_a.shape[2])).to(feature_b.device)
        feature_a_expand = torch.cat((feature_a, zeros_a), 2)
        feature_b_expand = torch.cat((zeros_b, feature_b), 2)
        joint_feature = torch.cat((feature_a_expand, feature_b_expand), 1)
        joint_feature = self.GELU(self.map(joint_feature))
        return self.drop(joint_feature)

class Ancillary_LeNet(nn.Module):
    def __init__(self, hidden, classes, layers):
        super(Ancillary_LeNet, self).__init__()
        self.CNNs = nn.ModuleList(
            [nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=7, padding=3) for _ in range(layers)])
        self.BN = nn.BatchNorm1d(hidden)  # nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])

        self.FC_combs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.FC_down = nn.Linear(hidden, 128)
        self.FC_out = nn.Linear(128, classes)
        self.layers = layers
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, dti_feature):
        dti_feature = dti_feature.permute(0, 2, 1)  # self.BN(dti_feature.permute(0, 2, 1))
        for i in range(self.layers):
            dti_feature = self.act(self.CNNs[i](dti_feature)) + dti_feature
        dti_feature = dti_feature.permute(0, 2, 1)
        dti_feature = torch.mean(dti_feature, dim=1)
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti_feature = self.FC_down(dti_feature)
        dti = self.FC_out(dti_feature)
        return dti

class SPD_LeNet(nn.Module):
    def __init__(self, hid1, hid2, classes, layers):
        super(SPD_LeNet, self).__init__()
        self.vec1 = SPDVectorize(hid1)
        self.vec2 = SPDVectorize(hid2)
        self.FC_down1 = nn.Linear(int((hid1+1)*hid1/2), 512)
        self.FC_down2 = nn.Linear(int((hid2 + 1) * hid2 / 2), 64)
        self.FC_combs = nn.ModuleList([nn.Linear(576, 576) for _ in range(layers)])
        self.FC_out = nn.Linear(576, classes)
        self.layers = layers
        self.act = nn.ReLU()

    def forward(self, dti_spd1, dti_spd2):
        dti_vec1 = self.vec1(dti_spd1)  # self.BN(dti_feature.permute(0, 2, 1))
        dti_vec2 = self.vec2(dti_spd2)
        dti_feature1 = self.FC_down1(dti_vec1)
        dti_feature2 = self.FC_down2(dti_vec2)
        dti_feature = torch.cat((dti_feature1, dti_feature2), 1)
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti = self.FC_out(dti_feature)
        return dti, dti_feature

class LSCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LSCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class SPD_DTI(nn.Module):
    def __init__(self, layer_gnn, device, hidden1=192, hidden2=64, n_layers=3, dropout=0.05):
        super(SPD_DTI, self).__init__()
        '''Drug'''
        self.encoder_drug = Encoder(768, hidden1, 64, 3, groups=32, pad1=7, pad2=3)
        self.atom_rep = Atom_rep(hidden2, device)
        # self.W_gnn = nn.ModuleList([nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2), nn.Linear(hidden2, hidden2)])
        self.GAT = GAT(hidden2, hidden2, n_layers).to("cuda")
        self.gnn_act = nn.GELU()
        self.C = hidden2
        self.cross_attention = cross_attention(hidden1, hidden2, dropout)
        # self.d_joint = Feature_Joint(hidden1, hidden2, hidden1 + hidden2)

        '''Protein'''
        self.encoder_protein_LM = Encoder(1024, hidden1, 128, 3, groups=64, pad1=7, pad2=3)
        self.encoder_protein_SPD = Encoder(120, hidden2, 60, 5, groups=15, pad1=3, pad2=3)
        self.R_D = Reduce_dimension()

        '''DECISION'''
        self.dt_joint = Feature_Joint(hidden1+hidden2, hidden1+hidden2, hidden1+hidden2)
        self.covariance = Covariance()
        self.DS1 = SPDTransform(hidden1+hidden2, 32)
        self.DS2 = SPDTransform(1300, 24)
        self.DTI_SPD = SPD(32, 1e-6)
        self.DTI_SPD2 = SPD(24, 1e-6)
        self.device = device
        self.layer_gnn = layer_gnn
        self.classfier = SPD_LeNet(32, 24, 2, n_layers)
        self.a_c = Ancillary_LeNet(hidden1+hidden2, 2, n_layers)
        self.act = nn.ReLU()
        self.GELU = nn.GELU()
        # for p in self.parameters():
        #     p.requires_grad = False

    def ReSize(self, feature, N):
        molecule_ST = torch.zeros((N, 100, self.C), device=self.device)
        for i in range(N):
            C_L = feature[i].shape[0]
            if C_L >= 100:
                molecule_ST[i, :, :] = feature[i][0:100, :]
            else:
                molecule_ST[i, :C_L, :] = feature[i]
        return molecule_ST

    def forward(self, inputs):
        """Data loading"""
        molecule_smiles, molecule_atoms, molecule_adjs, proteins, protein_LM, protein_SPD, molecule_LM = inputs
        N = molecule_smiles.shape[0]

        """Protein feature extractor"""
        proteins_LM = self.encoder_protein_LM(protein_LM.permute(0, 2, 1)).permute(0, 2, 1)
        proteins_SPD = self.encoder_protein_SPD(protein_SPD.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        proteins = torch.cat((proteins_LM, proteins_SPD), 2)

        """Drug feature extractor"""
        molecule_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        molecule_vec = self.atom_rep(molecule_atoms, N)
        molecule_ST = self.GAT(molecule_vec, molecule_adjs)
        molecule_ST = self.ReSize(molecule_ST, N)
        molecule_ST = self.cross_attention(molecule_ST, molecule_LM)
        molecules = torch.cat((molecule_LM, molecule_ST), 2)

        """SPD-based joint feature and classfier"""
        Joint_Feature = self.dt_joint(proteins, molecules)
        ancillary = self.a_c(Joint_Feature)
        DTI_Feature1 = self.DS1(self.covariance(Joint_Feature.permute(0, 2, 1)))
        DTI_Feature2 = self.DS2(self.covariance(Joint_Feature))
        DTI_Feature1 = self.DTI_SPD(DTI_Feature1)
        DTI_Feature2 = self.DTI_SPD2(DTI_Feature2)
        DTI, dti_feature = self.classfier(DTI_Feature1, DTI_Feature2)
        return DTI, ancillary

    def __call__(self, data, epoch=1, train=True):
        inputs, correct_interaction, SID = data[:-2], data[-2], data[-1]
        correct_interaction = torch.LongTensor(correct_interaction.to('cpu').numpy()).cuda()
        protein_drug_interaction, ancillary = self.forward(inputs)  # , dis_invariant
        t = int(epoch / 2)
        alpha = 0.5 * (10 / (t + 10))
        if train:
            loss = F.cross_entropy(protein_drug_interaction, correct_interaction)
            loss2 = F.cross_entropy(ancillary, correct_interaction)
            return loss * (0.8 - alpha) + loss2 * (0.2 + alpha)
        else:
            correct_labels = correct_interaction  # .to('cpu').data.numpy().reshape(-1)
            ys = F.softmax(protein_drug_interaction, 1)  # .to('cpu').data.numpy() C
            return correct_labels, ys

#0.9641450963037755	0.9634428884833846
