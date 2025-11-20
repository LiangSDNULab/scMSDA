import sys
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import *
def cluster_contrastive_loss(c_i, c_j, n_clusters, temperature=0.5):
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    similarity_f = torch.nn.CosineSimilarity(dim=2)

    # entropy
    p_i = c_i.sum(0).view(-1)
    p_i /= p_i.sum()
    ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
    p_j = c_j.sum(0).view(-1)   
    p_j /= p_j.sum()
    ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
    ne_loss = ne_i + ne_j

    N = 2 * n_clusters                  
    mask = torch.ones((N, N))           
    mask = mask.fill_diagonal_(0)       
    for i in range(n_clusters):
        mask[i, n_clusters + i] = 0
        mask[n_clusters + i, i] = 0
    mask = mask.bool()           

    c_i, c_j = c_i.t(), c_j.t()                       
    c = torch.cat((c_i, c_j), dim=0)    

    # print((c.unsqueeze(1).shape, c.unsqueeze(0).shape))
    sim = similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / temperature    
    sim_i_j = torch.diag(sim, n_clusters)                               
    sim_j_i = torch.diag(sim, -n_clusters)                              

    positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) 
    negative_clusters = sim[mask].reshape(N, -1)                           

    labels = torch.zeros(N).to(positive_clusters.device).long()            
    logits = torch.cat((positive_clusters, negative_clusters), dim=1)      
    loss = criterion(logits, labels) / N

    return loss + ne_loss
def OT_Loss(zs,z_fusion,q_centers,args):
    cos_i = consine_similarity(zs[0], q_centers)
    cos_j = consine_similarity(zs[1], q_centers)

    cos_fusion = consine_similarity(z_fusion, q_centers)
    T = sinkhorn(cos_fusion.detach(), epsilon=0.05, sinkhorn_iterations=100).to(args.device)

    T = T / T.sum(dim=1, keepdim=True)

    with torch.no_grad():
        wd_i, e_i = w_dist(cos_i.to(args.device), T.to(args.device), q_centers.shape[0],
                           args)  # 确保所有计算都在同一设备上
        wd_j, e_j = w_dist(cos_j.to(args.device), T.to(args.device), q_centers.shape[0], args)

    # OT损失
    OT_loss = ((-wd_i - e_i) + (-wd_j - e_j)).to(args.device)
    return OT_loss
def Dgan_loss(zs,z_fusion,args):#args.neighbors=10
    _, P = TPL(z_fusion.t(), args.neighbors, args)
    Dgan_loss_1 = Dgan(args, zs[0], z_fusion, P).to(args.device)
    Dgan_loss_2 = Dgan(args, zs[1], z_fusion, P).to(args.device)
    Dgan_loss = Dgan_loss_1 + Dgan_loss_2
    return Dgan_loss
def Recon_loss(x,xrs):
    rc_loss_i = F.mse_loss(xrs[0], x[0])
    rc_loss_j = F.mse_loss(xrs[1], x[1])
    rc_loss = rc_loss_i + rc_loss_j
    return rc_loss
def calcul_var(data, labels):
    labels = torch.squeeze(labels)
    # 获取不同聚类簇的索引
    clusters = torch.unique(labels)

    var_sum = 0.
    # 计算每个聚类簇的中心和方差
    for cluster in clusters:
        cluster_data = data[labels == cluster]
        cluster_center = torch.mean(cluster_data, dim=0)
        distances = torch.norm(cluster_data - cluster_center, dim=1)
        variance = torch.var(distances)
        var_sum += variance
        # print(f'Cluster {cluster.item() + 1} Center: {cluster_center}, Variance: {variance.item()}')

    return var_sum


def compute_cluster_loss(self, q_centers, k_centers, psedo_labels):
    d_q = q_centers.mm(q_centers.T) / self.temperature_l
    d_k = (q_centers * k_centers).sum(dim=1) / self.temperature_l
    d_q = d_q.float()
    d_q[torch.arange(self.num_clusters), torch.arange(self.num_clusters)] = d_k

    # q -> k
    # d_q = q_centers.mm(k_centers.T) / temperature

    zero_classes = torch.arange(self.num_clusters).cuda()[torch.sum(F.one_hot(torch.unique(psedo_labels),
                                                                             self.num_clusters), dim=0) == 0]
    mask = torch.zeros((self.num_clusters, self.num_clusters), dtype=torch.bool, device=d_q.device)
    mask[:, zero_classes] = 1
    d_q.masked_fill_(mask, -10)
    pos = d_q.diag(0)
    mask = torch.ones((self.num_clusters, self.num_clusters))
    mask = mask.fill_diagonal_(0).bool()
    neg = d_q[mask].reshape(-1, self.num_clusters - 1)
    loss = - pos + torch.logsumexp(torch.cat([pos.reshape(self.num_clusters, 1), neg], dim=1), dim=1)
    loss[zero_classes] = 0.
    loss = loss.sum() / (self.num_clusters - len(zero_classes))

    return loss


def contrastive_loss(z, temperature=0.5):
    """
    To get contrastive loss

    Parameters
    ----------
    z: sample features
    temperature: float optional(default 0.5)
        super parameter

    Returns 
    -------
    loss: contrastive loss
    """
    batch_size = z.size(0) // 2

    similarity_martix = torch.matmul(z, z.T) / temperature
    mask = torch.eye(batch_size * 2, dtype=torch.bool).to(z.device)
    logits = similarity_martix.masked_select(~mask).view(batch_size * 2, -1)
    targets = torch.arange(batch_size).repeat(2).T.flatten().to(z.device)

    loss = torch.nn.CrossEntropyLoss()(logits, targets.long())

    return loss



def similarity_measure(x1, x2):
    # 计算相似度度量，可以使用欧氏距离、余弦相似度等方法
    # 这里使用余弦相似度作为示例
    cos_sim = F.cosine_similarity(x1, x2, dim=-1)
    return cos_sim

def loss_function(x1, x2, y1, y2, margin=0.1):
    # x1, x2: 样本向量表示
    # y1, y2: 伪标签，指示样本的相似性

    p_i = x1.sum(0).view(-1)
    p_i /= p_i.sum()
    ne_i = (p_i * torch.log(p_i)).sum()
    p_j = x2.sum(0).view(-1)
    p_j /= p_j.sum()
    ne_j = (p_j * torch.log(p_j)).sum()
    entropy = ne_i + ne_j

    # 计算相似度度量
    sim = similarity_measure(x1, x2)

    # 相同标签样本的损失函数
    loss_same = F.mse_loss(sim[y1 == y2], torch.ones_like(sim[y1 == y2]))

    # 不同标签样本的损失函数
    loss_diff = F.relu(margin - sim[y1 != y2]).mean()

    # 总体损失函数
    loss = loss_same - loss_diff

    # return loss + entropy
    return loss

#ZINB损失=================================================================================================================
class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        # scale_factor = scale_factor[:, None]
        # mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result
class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)
#===============================================================================================

#对比损失，区分正负样本对参考论文scSimGCL===============================================================================================
def scSimGCL_loss(z, z_aug, adj, tau, hidden_norm=True):
    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z, z, hidden_norm))
    inter_view_sim = f(sim(z, z_aug, hidden_norm))

    positive = inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)

    loss = positive / (intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())

    adj_count = torch.sum(adj, 1) * 2 + 1
    loss = torch.log(loss) / adj_count

    return -torch.mean(loss, 0)

def sim(z1, z2, hidden_norm):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.T)


def Dgan(args,h_i, h_j, S):
    dim = h_i.shape[0]
    criterion = nn.CrossEntropyLoss(reduction="sum")
    S_1 = S.to(args.device).repeat(2, 2)

    # S_1[S_1 > 0.9] = 1
    # S_1[S_1 < 0.9] = 0
    #all_one = torch.ones(args.batch_size*2, args.batch_size*2).to(args.device)
    all_one = torch.ones_like(S_1)
    S_2 = all_one - S_1#样本距离越近，权重越小
    S_2 = S_2.to(h_i.device)
    # S_2 = all_one

    N = 2 * dim
    h = torch.cat((h_i, h_j), dim=0)

    sim = torch.matmul(h, h.T) / args.temperature_f
    sim1 = torch.multiply(sim, S_2)

    # sim_i_j = torch.diag(sim, args.batch_size)
    # sim_j_i = torch.diag(sim, -args.batch_size)
    sim_i_j = torch.diag(sim, h_i.shape[0])
    sim_j_i = torch.diag(sim, -h_i.shape[0])

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(N)

    negative_samples = sim1[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    loss = criterion(logits, labels)
    loss /= N
    return loss
def mask_correlated_samples( N):
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(N//2):
        mask[i, N//2 + i] = 0
        mask[N//2 + i, i] = 0
    mask = mask.bool()
    return mask

# self.v=1
def soft_assign(z,center,args):
    q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - center, 2), 2) / args.v)
    q = q.pow((args.v + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return q
def target_distribution(q):
    p = q ** 2 / q.sum(0)
    return (p.t() / p.sum(1)).t()
def KL_loss(z,center,args):

    def kld(target, pred):
        return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=-1))

    q = soft_assign(z, center, args)
    p = target_distribution(q)

    kldloss = kld(p, q)
    return kldloss