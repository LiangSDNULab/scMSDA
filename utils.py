import math
import torch
import random
import numpy as np
import faiss
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ranksums
from numpy.random import randint
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.use('Agg')

import scanpy as sc
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def next_batch(X1, X2,batch_size):
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        #batch_Y = Y[start_idx: end_idx, ...]
        yield (batch_x1, batch_x2, (i + 1))
#使用潜在嵌入表示z_fusion和模型得到的标签计算秩和检验
def calculate_cluster_closeness(z_fusion, labels):
    """
    计算每个簇的内部和外部接近中心性，并进行 Wilcoxon 秩和检验。

    参数:
        z_fusion (np.ndarray): 潜在嵌入表示，形状为 (n_samples, n_features)。
        labels (np.ndarray): 样本的簇标签，形状为 (n_samples,)。

    返回:
        results (dict): 每个簇的 Wilcoxon 秩和检验结果。
    """
    # 计算距离矩阵
    distances = squareform(pdist(z_fusion, metric='euclidean'))  # 形状为 (n_samples, n_samples)

    # 获取唯一的簇标签
    unique_labels = np.unique(labels)
    results = {}

    # 遍历每个簇
    for cluster_label in unique_labels:
        # 获取当前簇的样本索引
        cluster_indices = np.where(labels == cluster_label)[0]
        non_cluster_indices = np.where(labels != cluster_label)[0]

        # 提取内部接近中心性（簇内样本之间的距离）
        intra_distances = distances[np.ix_(cluster_indices, cluster_indices)]
        intra_distances = intra_distances[np.triu_indices_from(intra_distances, k=1)]  # 取上三角，避免重复

        # 提取外部接近中心性（簇内样本与簇外样本之间的距离）
        inter_distances = distances[np.ix_(cluster_indices, non_cluster_indices)]
        inter_distances = inter_distances.flatten()

        # 进行 Wilcoxon 秩和检验
        ranksum_stat, ranksum_p = ranksums(intra_distances, inter_distances)

        # 保存结果
        results[cluster_label] = {
            'intra_distances_mean': np.mean(intra_distances),
            'inter_distances_mean': np.mean(inter_distances),
            'ranksum_stat': ranksum_stat,
            'ranksum_p': ranksum_p
        }

    return results
def normalize(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x
def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state

def pairwise_distance(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis

#亲和矩阵===============================================================================
def build_affinity_matrix(X, k):
    # 将数据集转换为Tensor对象
    X = X.clone().detach()  # torch.tensor(X).float()
    X = X.cpu().numpy()

    # 初始化IndexFlatL2对象
    index = faiss.IndexFlatL2(X.shape[1])
    # 将数据集加入到索引中
    index.add(X)
    # 利用索引查找每个向量的k个最近邻点
    _, ind = index.search(X, k + 1)

    # 计算每个向量与其k个最近邻点之间的距离
    dist = np.array([np.linalg.norm(X[i] - X[ind[i][1:]], axis=1) for i in range(X.shape[0])])
    dist = torch.tensor(dist)
    # dist = torch.norm(X[:, None, :] - X[ind[:, 1:]], dim=2)
    # 将距离转换为亲和值
    aff = torch.exp(-dist ** 2 / 2)
    # 构造亲和矩阵
    W = torch.zeros(X.shape[0], X.shape[0])

    for i in range(X.shape[0]):
        W[i, ind[i][1:]] = aff[i]
        W[ind[i][1:], i] = aff[i]
    adj = np.array(W)
    normalization = 'NormAdj'
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    # adj = adj.astype("float")# torch.float(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    ind = torch.from_numpy(ind)
    return adj  # , ind
def fetch_normalization(type):
   switcher = {
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
       'NormAdj': normalized_adjacency,  # A' = (D)^-1/2 * ( A) * (D)^-1/2
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func
def normalized_adjacency(adj):
   adj = adj
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def normalized_adjacency(adj):
    adj = adj
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        device=torch.device('cuda')
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    # print(f'running k-means on {device}..')
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    dis_min = float('inf')
    initial_state_best = None
    for i in range(20):
        initial_state = initialize(X, num_clusters)
        dis = pairwise_distance_function(X, initial_state).sum()
        if dis < dis_min:
            dis_min = dis
            initial_state_best = initial_state

    initial_state = initial_state_best
    iteration = 0
    while True:
        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        if iteration > 500:
            break
        if center_shift ** 2 < tol:
            break

    return choice_cluster.cpu(), initial_state, dis
    # return choice_cluster.cpu(), initial_state
def pairwise_distance(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis
def pairwise_cosine(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis

#TPL==========================================================================
def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result
def TPL(X, num_neighbors,args, links=0):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return:
    """
    size = X.shape[1]#X的维度是128,256
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10**-10#每一列都是相同的

    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    sorted_distances = None
    torch.cuda.empty_cache()
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links is not 0:
        links = torch.Tensor(links).cuda()
        weights += torch.eye(size).cuda()
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.to(args.device)
    weights = weights.cuda()
    return weights, raw_weights

#用来保存性能指标到csv文件
def write_results(file_path, args, best_nmi, best_nmi_epoch, best_ari, best_ari_epoch,
                  best_acc, best_acc_epoch, best_acc_score, best_nmi_score, best_ari_score,
                  ):
    with open(file_path, "a+") as file:
        file.write(f"Dataset: {args.dataset}\n")  # 写入数据集名称
        file.write(f"model_version={args.version}_{args.lambda0}*recon_loss+{args.lambda1}*cl_loss+{args.lambda2}*CWCL_Loss+{args.lambda3}*OT_loss\n")
        file.write(f"seed={args.seed}_p={args.p}_neighbors={args.neighbors}\n")
        file.write(f"run={args.run}\n")
        file.write(f"Highest NMI: {best_nmi:.6f} in epoch {best_nmi_epoch}\n")
        file.write(f"Highest ARI: {best_ari:.6f} in epoch {best_ari_epoch}\n")
        file.write(f"Highest ACC: {best_acc:.6f} in epoch {best_acc_epoch}\n")
        file.write(f"Final Score at Best ACC: {str(best_acc_score)}\n")
        file.write(f"Final Score at Best NMI: {str(best_nmi_score)}\n")
        file.write(f"Final Score at Best ARI: {str(best_ari_score)}\n")
        file.write("-" * 50 + "\n")

#消融实验部分
#试图扩增技术
def add_random_perturbation(expression_matrix, noise_type="gaussian", noise_scale=0.1, dropout_rate=0.1):
    """
    对表达矩阵进行随机扰动。

    参数:
        expression_matrix (np.ndarray): 输入的表达矩阵，形状为 (n_cells, n_genes)。
        noise_type (str): 噪声类型，可选 "gaussian"（高斯噪声）或 "uniform"（均匀噪声）或 "dropout"（随机丢弃）。
        noise_scale (float): 噪声的强度。对于高斯噪声，表示标准差；对于均匀噪声，表示范围 [-noise_scale, noise_scale]。
        dropout_rate (float): 随机丢弃的比例（仅在 noise_type="dropout" 时使用）。

    返回:
        np.ndarray: 扰动后的表达矩阵。
    """
    perturbed_matrix = expression_matrix  # 创建副本以避免修改原始矩阵

    if noise_type == "gaussian":
        # 添加高斯噪声
        noise = np.random.normal(loc=0, scale=noise_scale, size=expression_matrix.shape)
        perturbed_matrix += noise

    elif noise_type == "uniform":
        # 添加均匀噪声
        noise = np.random.uniform(low=-noise_scale, high=noise_scale, size=expression_matrix.shape)
        perturbed_matrix += noise

    elif noise_type == "dropout":
        # 随机丢弃部分值（设为 0）
        mask = np.random.rand(*expression_matrix.shape) < dropout_rate
        perturbed_matrix[mask] = 0

    else:
        raise ValueError(f"未知的噪声类型: {noise_type}")

    # 确保扰动后的值非负（适用于表达矩阵）
    perturbed_matrix = np.maximum(perturbed_matrix, 0)

    return perturbed_matrix
