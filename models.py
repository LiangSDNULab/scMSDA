import torch.nn.functional as F
from losses import *
import copy
import math
import torch
import torch.nn as nn
import torch.nn.init as init

L2norm = nn.functional.normalize
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(Encoder, self).__init__()

        encoder_dim = [input_dim]
        encoder_dim.extend(embedding_dim)
        self._dim = len(encoder_dim) - 1
        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Softmax(dim=1))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(Decoder, self).__init__()

        decoder_dim = [i for i in reversed(embedding_dim)]
        decoder_dim.append(input_dim)
        self._dim = len(decoder_dim) - 1

        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(x)

class scMSDA(nn.Module):
    def __init__(self, view, input_dim, embedding_dim, cluster_dim, n_clusters,device):
        super(scMSDA, self).__init__()
        self.view=2
        self.encoders = []
        self.decoders = []
        self.temperature_l=0.5#聚类损失模块温度参数
        self.view = view
        self.cluster_dim = cluster_dim

        for v in range(self.view):
            self.encoders.append(Encoder(input_dim[v], embedding_dim).to(device))
            self.decoders.append(Decoder(input_dim[v], embedding_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.n_clusters = n_clusters

        self.cluster = nn.Sequential(
            nn.Linear(embedding_dim[-1], self.cluster_dim),
            nn.BatchNorm1d(self.cluster_dim),
            nn.ReLU(),
            nn.Linear(self.cluster_dim, n_clusters),
            nn.Softmax(dim=1)
        )

        self.device=device
        self._dec_mean = nn.Sequential(nn.Linear(input_dim[0], input_dim[0]), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(input_dim[0], input_dim[0]), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(input_dim[0], input_dim[0]), nn.Sigmoid())
        self.zinb_loss = ZINBLoss().to(device)


        self.commonfeature_contrastive_module = nn.Sequential(
            nn.Linear(embedding_dim[2] * 2, embedding_dim[2]),
        )


        self.mk = torch.nn.Linear(embedding_dim[2] *2, 64, bias=False)  #
        self.mv = torch.nn.Linear(64, embedding_dim[2] * 2, bias=False)  #
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)


    def forward(self, xs):
        xrs = []
        zs = []
        for i in range(self.num_views):
            z = self.encoders[i](xs[i])
            xr = self.decoders[i](z)
            xrs.append(xr)
            zs.append(z)

        return xrs, zs
    def fusion(self, zs):
        """
        融合多个视图的特征。

        参数:
            zs (list): 包含多个视图特征的列表，例如 [z_i, z_j]。

        返回:
            z_fusion (torch.Tensor): 融合后的特征。
        """
        # 拼接多个视图的特征
        z_fusion = torch.cat(zs, dim=1)
        #z_fusion = zs[0]+zs[1]
        # 计算注意力权重
        attn = self.mk(z_fusion)
        attn = self.softmax(attn)
        # 融合特征
        out = self.mv(attn)
        out = self.dropout(out)
        # 特征归一化
        z_fusion = torch_normalize(self.commonfeature_contrastive_module(out + z_fusion), dim=1)
        return z_fusion

    def cluster_alignment(self, common_z, zs):
        """
        聚类和对齐多个视图的特征。

        参数:
            common_z (torch.Tensor): 融合后的特征。
            zs (list): 包含多个视图特征的列表，例如 [z_i, z_j]。

        返回:
            cl (torch.Tensor): 融合特征的聚类结果。
            cl_i (torch.Tensor): 第一个视图特征的聚类结果。
            cl_j (torch.Tensor): 第二个视图特征的聚类结果。
        """
        # 对融合后的特征进行聚类
        cl = self.cluster(common_z)

        # 对每个视图的特征进行聚类
        cl_i = self.cluster(F.normalize(zs[0]))
        cl_j = self.cluster(F.normalize(zs[1]))

        return cl, cl_i, cl_j


    def compute_centers(self, x, psedo_labels):
        n_samples = x.size(0)
        if len(psedo_labels.size()) > 1:
            weight = psedo_labels.T
        else:
            weight = torch.zeros(self.n_clusters, n_samples).to(x)  # L, N
            weight[psedo_labels, torch.arange(n_samples)] = 1
        weight = F.normalize(weight, p=1, dim=1)  # l1 normalization
        centers = torch.mm(weight, x)
        centers = F.normalize(centers, dim=1)

        return centers
    #参数self.temperature_l=0.5
    def compute_cluster_loss(self, q_centers, k_centers, psedo_labels):
        d_q = q_centers.mm(q_centers.T) / self.temperature_l
        d_k = (q_centers * k_centers).sum(dim=1) / self.temperature_l
        d_q = d_q.float()
        d_q[torch.arange(self.n_clusters), torch.arange(self.n_clusters)] = d_k



        zero_classes = torch.arange(self.n_clusters, device=q_centers.device)[torch.sum(F.one_hot(torch.unique(psedo_labels),
                                                                                 self.n_clusters), dim=0) == 0]
        mask = torch.zeros((self.n_clusters, self.n_clusters), dtype=torch.bool, device=d_q.device)
        mask[:, zero_classes] = 1
        d_q.masked_fill_(mask, -10)
        pos = d_q.diag(0)
        mask = torch.ones((self.n_clusters, self.n_clusters))
        mask = mask.fill_diagonal_(0).bool()
        neg = d_q[mask].reshape(-1, self.n_clusters - 1)
        loss = - pos + torch.logsumexp(torch.cat([pos.reshape(self.n_clusters, 1), neg], dim=1), dim=1)
        loss[zero_classes] = 0.
        loss = loss.sum() / (self.n_clusters - len(zero_classes))

        return loss
    def forward(self, xs):
        xrs = []
        zs = []
        cs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            c = self.cluster(F.normalize(z))
            #q = self.label_contrastive_module(z)
            zs.append(z)
            xrs.append(xr)
            cs.append(c)
        return xrs, zs, cs
    def ZINB_Loss(self,input,h):
        mean = self._dec_mean(h)
        disp = self._dec_disp(h)
        pi = self._dec_pi(h)
        zinb_loss = self.zinb_loss(input, mean, disp, pi)
        return zinb_loss

    def target_distribution(self, q):
        p = q ** 2 / q.sum(0)
        return (p.t() / p.sum(1)).t()
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ClusteringLayer(nn.Module):

    def __init__(self, n_clusters, latent_dim, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.randn(n_clusters, latent_dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.clusters, a=math.sqrt(5))

    def forward(self, inputs):
        q = 1.0 / (1.0 + (torch.sum(torch.square(torch.unsqueeze(inputs, dim=1) - self.clusters), dim=2) / self.alpha))
        q = torch.pow(q, (self.alpha + 1.0) / 2.0)
        q = torch.transpose(torch.transpose(q, dim0=0, dim1=1) / torch.sum(q, dim=1), dim0=0, dim1=1)
        return q
def torch_normalize(x, dim=1):
    return F.normalize(x, p=2, dim=dim)