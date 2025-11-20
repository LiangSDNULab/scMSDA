import math
import torch
import random
import numpy as np
import faiss
import psutil
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ranksums
from numpy.random import randint
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import matplotlib
import os
import torch
import torch.nn as nn
import h5py
import os, random
import numpy as np
import scipy.io as sio
from scipy import sparse
#from utils import normalize
import h5py
import scanpy as sc
import pandas as pd
import torch
import muon as mu
from utils import *
import scipy as sp
from scipy.io import loadmat
from torch.utils.data import Dataset
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import numpy as np
from time import time
import anndata as ad
#from memory_profiler import memory_usage

def load_data(data_name):
    X_list = []
    Y = None

    if data_name in ['synthetic3d']:
        mat = sio.loadmat('./data/synthetic3d.mat')
        X = mat['X']
        X_list.append(normalize(X[0][0].astype('float32')))
        X_list.append(normalize(X[1][0].astype('float32')))
        X_list.append(normalize(X[2][0].astype('float32')))
        Y = np.squeeze(mat['Y']).astype('int')

    return X_list, Y
matplotlib.use('Agg')

def save_results_small(args,path, dataset, embedding, predict_label):
    """
    保存模型嵌入和指标到 HDF5 文件

    参数:
        path (str): 保存路径
        dataset (str): 数据集名称
        best_Z (dict): 包含最佳指标的字典，需包含 'acc', 'ari', 'nmi', 'epoch'
        pred (torch.Tensor | np.ndarray): 模型嵌入
        Z_y_pred (np.ndarray | list): 聚类预测标签
        save_args (int): 是否保存（默认 1 表示保存）
    """


    # 确保目录存在
    if not os.path.exists(path):
        os.makedirs(path)

    # 构造文件路径（保留 acc 四位小数）
    h5_path = os.path.join(path, f"{dataset}.h5")

    # 如果是 torch.Tensor，转为 numpy
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()

    # 保存到 HDF5
    with h5py.File(h5_path, 'w') as h5f:
        h5f.create_dataset("embedding", data=embedding)
        h5f.create_dataset("predict_label", data=predict_label)


    print(f"结果已保存到 {h5_path}")
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
                  ari_silhouette, nmi_silhouette, ranksum_ari, ranksum_nmi):
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
        file.write(f"Best ARI Silhouette Score: {ari_silhouette}\n")
        file.write(f"Best NMI Silhouette Score: {nmi_silhouette}\n")

        file.write("Cluster_ranksum_ari:\n")
        for cluster_label, result in ranksum_ari.items():
            file.write(f"Cluster {cluster_label}: "
                       f"Intra-cluster mean distance: {result['intra_distances_mean']}, "
                       f"Inter-cluster mean distance: {result['inter_distances_mean']}, "
                       f"Wilcoxon Ranksum Statistic: {result['ranksum_stat']}, "
                       f"Wilcoxon p-value: {result['ranksum_p']}\n")

        file.write("-" * 50 + "\n")  # 添加分隔线便于阅读

        file.write("Cluster_ranksum_nmi:\n")
        for cluster_label, result in ranksum_nmi.items():
            file.write(f"Cluster {cluster_label}: "
                       f"Intra-cluster mean distance: {result['intra_distances_mean']}, "
                       f"Inter-cluster mean distance: {result['inter_distances_mean']}, "
                       f"Wilcoxon Ranksum Statistic: {result['ranksum_stat']}, "
                       f"Wilcoxon p-value: {result['ranksum_p']}\n")
        file.write("-" * 50 + "\n")
def write_results_new(file_path, args, best_nmi, best_nmi_epoch, best_ari, best_ari_epoch,
                  best_acc, best_acc_epoch, best_acc_score, best_nmi_score, best_ari_score,
                 ):
    with open(file_path, "a+") as file:
        file.write(f"Dataset: {args.dataset}\n")  # 写入数据集名称
        file.write(f"model_version={args.version}_{args.lambda0}*recon_loss+{args.lambda1}*cl_loss+{args.lambda2}*CWCL_Loss+{args.lambda3}*OT_loss\n")
        file.write(f"seed={args.seed}_p={args.p}_neighbors={args.neighbors}\n")
        file.write(f"run={args.run}\n")
        #file.write(f"noise_level={args.noise_level}\n")
        file.write(f"Highest NMI: {best_nmi:.6f} in epoch {best_nmi_epoch}\n")
        file.write(f"Highest ARI: {best_ari:.6f} in epoch {best_ari_epoch}\n")
        file.write(f"Highest ACC: {best_acc:.6f} in epoch {best_acc_epoch}\n")
        file.write(f"Final Score at Best ACC: {str(best_acc_score)}\n")
        file.write(f"Final Score at Best NMI: {str(best_nmi_score)}\n")
        file.write(f"Final Score at Best ARI: {str(best_ari_score)}\n")

        #file.write("-" * 50 + "\n")
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
def tsne_visualization_original(embedding, labels, save_path, title=None):
    """执行t-SNE降维和可视化"""
    #tsne = TSNE(n_components=2, random_state=42, perplexity=15, n_iter=300)#perplexity=30
    tsne = TSNE(n_components=2, random_state=42)
    embedding_2d = tsne.fit_transform(embedding)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels,
        #cmap="tab20",  # 使用离散颜色方案
        cmap="viridis",
        s=10,
        #alpha=0.8
    )
    # 添加图例（优化显示）
    # plt.legend(
    #     #*scatter.legend_elements(),
    #     *scatter.legend_elements(markeredgewidth=0.5, markersize=8),
    #     title=None,
    #     loc="lower left",  # 改为左下角#loc="lower left",  # 改为左下角
    #     bbox_to_anchor=(0, 0),  # 定位到左下角坐标(0,0)
    #     ncol=1,
    #
    #     frameon=False,
    #     borderaxespad=0.5  # 控制图例与坐标轴边框的间距
    # # )
    # plt.legend(
    #     *scatter.legend_elements(),
    #     bbox_to_anchor=(1.05, 1),  # 移到图外右侧
    #     loc="upper left",
    #     borderpad=1
    # )
    if np.issubdtype(labels.dtype, np.integer):  # 离散标签
        legend = plt.legend(
            *scatter.legend_elements(),
            #title="Classes",
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),  # 稍微向内偏移
            #frameon=True,
            #framealpha=0.8,
            #edgecolor='gray',
            handleheight=1.5,
            handlelength=1.5,
            borderaxespad=0
        )
        # 自动调整图例大小避免重叠
        plt.tight_layout(rect=[0, 0, 0.85, 1]) if len(np.unique(labels)) > 5 else None
    if title:  # 可选：只有title有值时才显示
        plt.title(title)

    plt.xlabel(None)
    plt.ylabel(None)
    plt.grid(False)  # 关闭网格

    # 保存图像
    #plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.savefig(save_path)
    plt.close()
def tsne_visualization(embedding, labels, save_path, title=None):
    """执行t-SNE降维和可视化"""
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)

    embedding_2d = tsne.fit_transform(to_numpy_safe(embedding))

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels,
        #cmap="tab20",  # 使用离散颜色方案
        cmap="viridis",
        s=10,
        alpha=0.8
    )
    plt.legend(
        *scatter.legend_elements(
            prop="colors",  # 必须保留，指明基于颜色生成图例
            markeredgewidth=0  # ✅ 正确位置：在legend_elements()中设置边缘宽度
        ),
        title=None,
        loc="upper right",
        frameon=False,
        markerscale=1.5,  # ✅ 正确位置：在plt.legend()中设置点大小缩放
        handletextpad=0.1,
        borderaxespad=0.3
    )
    if title:  # 可选：只有title有值时才显示
        plt.title(title)

    plt.xlabel(None)
    plt.ylabel(None)
    plt.grid(False)  # 关闭网格

    # 保存图像
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
def to_numpy_safe(x):
    """
    安全地将输入转换为 numpy.ndarray：
    - 如果是 torch.Tensor，则先 .detach()，再 .cpu()，最后 .numpy()
    - 如果本身就是 numpy，则直接返回
    - 如果是 list，则转换为 numpy
    - 其他类型抛出错误
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        return np.array(x)
    else:
        raise TypeError(f"无法转换类型: {type(x)} 为 numpy.ndarray")
def tsne_visualization_new(embedding, labels, save_path, title=None):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as mpatches

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    embedding_2d = tsne.fit_transform(embedding)

    plt.figure(figsize=(8, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        idx = labels == label
        plt.scatter(
            embedding_2d[idx, 0],
            embedding_2d[idx, 1],
            color=colors[i],
            s=10,
            alpha=0.8,
            label=str(label)
        )

    if title:
        plt.title(title)

    # 图例放图外
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=False,
        title="Classes"
    )

    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    '''
    path = f"./embedding/{args.dataset}/"
        tSNE_path = os.path.join(path, f'scDSC_{args.dataset}_tsne.png')
        os.makedirs(os.path.dirname(tSNE_path), exist_ok=True)
        tsne_visualization(
                                best_embedding,
                                res2,
                                tSNE_path,
                                None
                            )
    '''
#多GPU使用==================================
class GPUManager:
    def __init__(self, gpu_ids="0,1"):
        self.gpu_ids = gpu_ids
        self.setup_gpus()

    def setup_gpus(self):
        """设置使用的GPU"""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print(f"🎯 设置使用GPU: {self.gpu_ids}")

        if torch.cuda.is_available():
            print(f"✅ 可用GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("❌ 没有可用的GPU")

    def enable_multi_gpu(self, model):
        """启用多GPU支持"""
        if torch.cuda.device_count() > 1:
            print(f"🚀 启用多GPU训练，使用 {torch.cuda.device_count()} 个GPU")
            return nn.DataParallel(model)
        return model


def save_results(args,path, dataset, acc_score,ari_score, embedding, predict_label, save_args=1):
    """
    保存模型嵌入和指标到 HDF5 文件

    参数:
        path (str): 保存路径
        dataset (str): 数据集名称
        best_Z (dict): 包含最佳指标的字典，需包含 'acc', 'ari', 'nmi', 'epoch'
        pred (torch.Tensor | np.ndarray): 模型嵌入
        Z_y_pred (np.ndarray | list): 聚类预测标签
        save_args (int): 是否保存（默认 1 表示保存）
    """
    if save_args != 1:
        return None

    # 确保目录存在
    if not os.path.exists(path):
        os.makedirs(path)

    # 构造文件路径（保留 acc 四位小数）
    h5_path = os.path.join(path, f"{dataset}_acc={acc_score['accuracy']:.4f}_ari={acc_score['ARI']:.4f}_nmi={acc_score['NMI']:.4f}.h5")

    # 如果是 torch.Tensor，转为 numpy
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()

    # 保存到 HDF5
    with h5py.File(h5_path, 'w') as h5f:
        h5f.create_dataset("embedding", data=embedding)
        h5f.create_dataset("predict_label", data=predict_label)
        h5f.create_dataset("metrics/best_acc/acc", data=acc_score['accuracy'])
        h5f.create_dataset("metrics/best_acc/ari", data=acc_score['ARI'])
        h5f.create_dataset("metrics/best_acc/nmi", data=acc_score['NMI'])
        #h5f.create_dataset("metrics/epoch", data=score['epoch'])
        if 'NMI' in acc_score:
            h5f.create_dataset("metrics/best_acc/ami", data=acc_score['AMI'])
        if 'Purity' in acc_score:
            h5f.create_dataset("metrics/best_acc/purity", data=acc_score['Purity'])
        if 'precision' in acc_score:
            h5f.create_dataset("metrics/best_acc/precision", data=acc_score['precision'])
        if 'f_measure' in acc_score:
            h5f.create_dataset("metrics/best_acc/f_measure", data=acc_score['f_measure'])
        if 'recall' in acc_score:
            h5f.create_dataset("metrics/best_acc/recall", data=acc_score['recall'])

        h5f.create_dataset("metrics/best_ari/ari", data=ari_score['ARI'])
        h5f.create_dataset("metrics/best_ari/acc", data=ari_score['accuracy'])
        h5f.create_dataset("metrics/best_ari/nmi", data=ari_score['NMI'])
        if 'NMI' in ari_score:
            h5f.create_dataset("metrics/best_ari/ami", data=ari_score['AMI'])
        if 'Purity' in ari_score:
            h5f.create_dataset("metrics/best_ari/purity", data=ari_score['Purity'])
        if 'precision' in ari_score:
            h5f.create_dataset("metrics/best_ari/precision", data=ari_score['precision'])
        if 'f_measure' in ari_score:
            h5f.create_dataset("metrics/best_ari/f_measure", data=ari_score['f_measure'])
        if 'recall' in ari_score:
            h5f.create_dataset("metrics/best_ari/recall", data=ari_score['recall'])
    print(f"结果已保存到 {h5_path}")
    return h5_path
def show_info():
    #计算消耗内存
    pid = os.getpid()
    # 模块名比较容易理解：获得当前进程的pid
    p = psutil.Process(pid)
    # 根据pid找到进程，进而找到占用的内存值
    info = p.memory_full_info()
    memory = info.uss / 1024 / 1024
    return memory


def adjust_labels(labels):
    unique_labels = np.unique(labels)  # Find all the different tags and sort them
    unique_labels_sorted = np.sort(unique_labels)
    new_labels = np.zeros_like(labels, dtype=np.int64)  # Create a new label tensor with data type long
    for i in range(len(labels)):  # Adjust label range
        label = labels[i]
        new_label = np.where(unique_labels_sorted == label)[0][0]
        new_labels[i] = new_label
    return new_labels


import numpy as np


def add_gaussian_noise(X, noise_level=0.1):
    """
    添加高斯噪声

    参数:
    X: 原始数据
    noise_level: 噪声级别 (标准差系数)

    返回:
    添加噪声后的数据
    """
    if torch.is_tensor(X):
        # PyTorch张量
        std_val = torch.std(X).item()
        noise = torch.randn_like(X) * noise_level * std_val
        return X + noise
    else:
        # NumPy数组
        noise = np.random.normal(0, noise_level * np.std(X), X.shape)
        return X + noise
    return X + noise
def tsne_visualization3(embedding, labels, save_path, title=None):
    """执行t-SNE降维和可视化"""
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
    embedding_2d = tsne.fit_transform(to_numpy_safe(embedding))

    # 将字符串标签转换为数值
    unique_labels = np.unique(labels)
    label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_num[label] for label in labels])

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 8))  # 增加宽度以容纳图例

    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=numeric_labels,
        cmap="tab20",
        s=10,
        alpha=0.8
    )

    # 创建自定义图例并放在图外右侧
    legend_elements = []
    for i, label in enumerate(unique_labels):
        color = plt.cm.tab20(i / len(unique_labels))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=8, label=label))

    # 将图例放在图外右侧
    legend = ax.legend(handles=legend_elements,
                       title=None,
                       loc='center left',  # 图例基于左侧居中
                       bbox_to_anchor=(1.05, 0.5),  # 放在图外右侧，垂直居中
                       frameon=False,  # 无图例边框
                       markerscale=1.5,
                       handletextpad=0.1,
                       borderaxespad=0.3)

    if title:
        ax.set_title(title)

    # 设置坐标轴标签
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    # 移除网格
    ax.grid(False)

    # 设置边框：只保留左边和下边
    ax.spines['top'].set_visible(False)  # 移除上边框
    ax.spines['right'].set_visible(False)  # 移除右边框
    ax.spines['left'].set_visible(True)  # 保留左边框
    ax.spines['bottom'].set_visible(True)  # 保留下边框

    # 调整布局以确保图例不被裁剪
    plt.tight_layout()

    # 保存图像时确保图例完整包含
    plt.savefig(save_path, bbox_extra_artists=(legend,), bbox_inches="tight", dpi=300)
    plt.close()



def load_scRNAdata(dataset,args):

    args.file_csv_1 = ["biase", "Romanov", "zeisel", "deng", "darmanis", "goolam", "Baron_mouse"]
    args.file_csv_2 = ["kolo", "lawlor"]  # "bmcite",
    args.file_h5_1 = ["MCA", "Camp", "human_kidney_counts_top2000",
                      "Human_Pancreas_cell_2", "Human_Pancreas_cell_7266",
                      "Mouse_Pancreas_cell_2", "Mouse_Pancreas_cell_1886","Mouse_Pancreas_cell_1","Mouse_Pancreas_cell_2"
                      "mouse_bladder_cell_select_2100", "mouse_bladder_cell",
                  "mouse_ES_cell_select_2100", "mouse_ES_cell",
                      "worm_neuron_cell_select_2100", "worm_neuron_cell",
                      "10X_PBMC_select_2100", "10X_PBMC_4271",
                      "10X_PBMC_4340","Macosko","Shekhar_mouse_retina_raw_data",]  # "Macosko","Baron","HumanLiver_counts_top5000",
    args.file_h5_2 = ["Adam", "Young", "klein", "Chen", "Pollen", "Muraro","muraro_2",
                      "Quake_10x_Limb_Muscle", "Quake_10x_Bladder","Quake_10x_Trachea","Quake_10x_Spleen",
                      "Quake_Smart-seq2_Lung", "Quake_Smart-seq2_Limb_Muscle", "Quake_Smart-seq2_Heart", "Wang_Lung",
                      "Quake_Smart-seq2_Trachea",
                      "Quake_Smart-seq2_Diaphragm"]  # "Plasschaert","Wang_Lung","Quake_10x_Spleen","Quake_10x_Trachea","Adam",

    args.file_sc10X = ["sc10X"]
    args.batch_dataset = ["human_pancreas_norm_complexBatch"]
    args.batch_pancreas =["pancreas"]
    args.h5mu=["multiome_training"]#我跑这个数据集，最高ari在第一个epoch，是0.37最高，scDCCA早就是0.47多，scGCOT报内存不足
    if args.dataset in args.file_csv_1 or args.dataset in args.file_csv_2:
        args.type = ".csv"
    elif args.dataset in args.file_h5_1 or args.dataset in args.file_h5_2:
        args.type = ".h5"
    elif args.dataset in args.file_sc10X:
        args.type = ".mat"
    elif args.dataset in args.batch_dataset:
        args.type = ".h5ad"
    elif args.dataset in args.h5mu:
        args.type = ".h5mu.gz"
    elif args.dataset in args.batch_pancreas:###没必要这个
        args.type = ".csv"
    else:
        raise ValueError("Dataset not found in the specified file lists.")
    data_name = dataset + args.type
    print("dataset_name:", dataset)
    file_path = os.path.join(args.data_file, data_name)
    if dataset in args.file_csv_1:
        data_mat = pd.read_csv(file_path, header=None, index_col=None)
        # y = data_mat.iloc[1,1:].to_numpy(dtype=int)
        y = data_mat.iloc[1, 1:].to_numpy(dtype=int)
        x = data_mat.iloc[3:, 1:]
        x = x.T
        print("dataset_shape:", x.shape[0],x.shape[1])
        adata = sc.AnnData(x, dtype="float64")

        #romanov专属预处理方式，seed=42，ari达到0.66
        if args.dataset == "Romanov":
            # 对 Romanov 数据集进行预处理
            adata = read_dataset(adata,
                                 transpose=False,
                                 test_split=False,
                                 copy=True)
            adata = normalize_cell(adata,
                                   size_factors=True,
                                   normalize_input=True,
                                   logtrans_input=True)
        # 一般csv文件是处理过的数据，不用再预处理
        #预处理方式1
        # romanov用的话，ari为0.58
        # sc.pp.normalize_total(adata, target_sum=1e4)
        # sc.pp.log1p(adata)
        # sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        #预处理方式2
        # adata = read_dataset(adata,
        #                      transpose=False,
        #                      test_split=False,
        #                      copy=True)
        # adata = normalize_cell(adata,
        #                        size_factors=True,
        #                        normalize_input=True,
        #                        logtrans_input=True)
        #预处理方式3
        # adata = pre_normalize(adata, copy=True, highly_genes=3000, size_factors=True, normalize_input=True,
        #                       logtrans_input=True)




        sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        highvar = adata.var.highly_variable
        adata = adata[:, highvar]
        x = adata.X
    elif dataset in args.file_csv_2:
        data_path = args.data_file+dataset+"_pre.csv"
        lable_path = args.data_file + dataset + "_lable.csv"
        x = pd.read_csv(data_path, header=None, index_col=None)
        print("dataset_shape:", x.shape[0], x.shape[1])
        y = pd.read_csv(lable_path, header=None, index_col=None)
        y = adjust_labels(np.array(y).flatten())
        adata = sc.AnnData(x, dtype="float64")
        #
        # adata = read_dataset(adata,
        #                      transpose=False,
        #                      test_split=False,
        #                      copy=True)
        #
        # adata = normalize_cell(adata,
        #                        size_factors=True,
        #                        normalize_input=True,
        #                        logtrans_input=True)
        sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        highvar = adata.var.highly_variable
        adata = adata[:, highvar]
        x=adata.X

    elif dataset in args.file_h5_1:
        if dataset in ["Camp","Baron","MCA","10X_PBMC_4340"]:
            data_mat = h5py.File(file_path, 'r')
            x = np.array(data_mat['X']).astype('float')
            y = np.array(data_mat['obs']['Group'])
        else:
            data_mat = h5py.File(file_path, 'r')
            x = np.array(data_mat['X']).astype('float')
            y = np.array(data_mat['Y'])
        print("dataset_shape:", x.shape[0], x.shape[1])
        adata = sc.AnnData(x, dtype=x.dtype)
        adata.obs['Group'] = adjust_labels(y)
        # if sp.sparse.issparse(adata.X):
        #     X_dense = adata.X.toarray()
        # else:
        #     X_dense = adata.X
        #
        #     # 检查是否所有值均为整数
        # is_all_integer = np.allclose(X_dense, np.round(X_dense))
        #
        # if is_all_integer:
        #     print("数据是浮点类型，但所有值均为整数（如 1.0, 2.0）")
        # else:
        #     print("数据是真正的浮点数（含非零小数部分）")
        #预处理方式1,适应于Mouse ES cells等通用,mouse_bladder_cell_2100最好用的预处理方式
        #check_X_range(adata)
        adata = read_dataset(adata, transpose=False, test_split=False, copy=True)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        #sc.pp.scale(adata)

        #adata = normalize_cell(adata, size_factors=True, normalize_input=True, logtrans_input=True)

        # check_X_range(adata)
        # means = np.asarray(adata.X.mean(axis=0)).flatten()
        # print("是否存在 inf in means:", np.isinf(means).any())
        # print("是否存在 NaN in means:", np.isnan(means).any())

        #预处理方式2,适应于worm_neuron_cell_select_2100
        # adata = read_dataset(adata, transpose=False, test_split=False, copy=True)
        # adata = normalize_cell(adata, size_factors=True, normalize_input=True, logtrans_input=True)

        # 预处理方式3 Camp专用,从scMMN找的,预处理方式3应该和预处理方式2一样的原理，不能跑Mouse ES cells
        # adata = pre_normalize(adata, copy=True, highly_genes=3000, size_factors=True, normalize_input=True,
        #                       logtrans_input=True)

        sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        highvar = adata.var.highly_variable
        adata = adata[:, highvar]
        x = adata.X
    elif dataset in args.file_h5_2:
        x, y ,cell_name= prepro(file_path)
        x = np.ceil(x)
        print("dataset_shape:", x.shape[0], x.shape[1])
        adata = sc.AnnData(x, dtype=x.dtype)
        adata.obs['Group'] = adjust_labels(y)
        #adata.obs['batch'] = batch
        # 默认的
        # sc.pp.normalize_total(adata, target_sum=1e4)
        # sc.pp.log1p(adata)
        # sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        #预处理方式2
        # adata = read_dataset(adata, transpose=False, test_split=False, copy=True)
        # adata = normalize_cell(adata, size_factors=True, normalize_input=True, logtrans_input=True)
        # sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        #预处理方式3#"Quake_10x_Bladder"专用
        # adata = pre_normalize(adata, copy=True, highly_genes=3000, size_factors=True, normalize_input=True,
        #                      logtrans_input=True)
        # if sp.sparse.issparse(adata.X):
        #     X_dense = adata.X.toarray()
        # else:
        #     X_dense = adata.X
        #
        #     # 检查是否所有值均为整数
        # is_all_integer = np.allclose(X_dense, np.round(X_dense))
        #
        # if is_all_integer:
        #     print("数据是浮点类型，但所有值均为整数（如 1.0, 2.0）")
        # else:
        #     print("数据是真正的浮点数（含非零小数部分）")
        #预处理方式1,适应于Mouse ES cells等通用,mouse_bladder_cell_2100最好用的预处理方式
        #check_X_range(adata)
        adata = read_dataset(adata, transpose=False, test_split=False, copy=True)

        if args.dataset == "Quake_10x_Bladder":
            adata = pre_normalize(adata, copy=True, highly_genes=3000, size_factors=True, normalize_input=False,
                                 logtrans_input=True)
        else:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        #check_X_range(adata)
        highvar = adata.var.highly_variable
        adata = adata[:, highvar]


        x=adata.X
    elif dataset in args.file_sc10X:
        f1 = "/home/JiangCongCong/data/sc10X/sc10X.mat"
        data1 = loadmat(f1, mat_dtype=True)
        sc10x = data1['sc10x']
        x = sc10x.astype(np.float32)

        label = "/home/JiangCongCong/data/sc10X/label_groundtruth.mat"
        label = loadmat(label, mat_dtype=True)
        label = label['label_groundtruth']
        Y = label.astype(np.float32)
        y = Y.squeeze()
        n_clusters = 3

        # preprocessing scRNA-seq read counts matrix
        adata = sc.AnnData(x)
        adata.obs['Group'] = Y
        '''
        if sp.sparse.issparse(adata.X):
            X_dense = adata.X.toarray()
        else:
            X_dense = adata.X

            # 检查是否所有值均为整数
        is_all_integer = np.allclose(X_dense, np.round(X_dense))

        if is_all_integer:
            print("数据是浮点类型，但所有值均为整数（如 1.0, 2.0）")
        else:
            print("数据是真正的浮点数（含非零小数部分）")
        '''
        #预处理方式1,适应于Mouse ES cells等通用,mouse_bladder_cell_2100最好用的预处理方式
        #check_X_range(adata)
        adata = read_dataset(adata,
                             transpose=False,
                             test_split=False,
                             copy=True)

        adata = normalize_cell(adata,
                            size_factors=True,
                            normalize_input=True,
                            logtrans_input=True)
        sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        highvar = adata.var.highly_variable
        adata = adata[:, highvar]
        # ================= load sc data
        x = adata.X
        # x_scRNAraw = x1.raw.X
        # x_scRNA_size_factor = x1.obs['size_factors'].values
    elif dataset in args.batch_dataset:#这个批次数据集已经预处理过
        adata = load_batch_human_pancreas(file_path)
        x = adata['expr_matrix']
        y = adata['celltype_labels']
    elif dataset in args.h5mu:#没挑选高可变基因
        data = mu.read_h5mu(file_path)
        #adata = mdata.mod['rna']  # 就是标准 AnnData 对象

        x = np.array(data.mod['rna'].X.todense())


        y = (data.mod['rna'].obs['cell_type'].cat.codes).to_numpy()
        batch_info = data.mod['rna'].obs['batch'].values if 'batch' in data.mod['rna'].obs else None

        adata1 = sc.AnnData(x)
        adata1.obs['Group'] = y

        adata1 = read_dataset(adata1,
                              transpose=False,
                              test_split=False,
                              copy=True)

        adata = normalize_cell(adata1,
                           size_factors=True,
                           normalize_input=True,
                           logtrans_input=True)
    elif dataset in args.batch_pancreas:

        we_use = [1, 2]
        # data
        adata1 = pd.read_csv('/home/JiangCongCong/data/pancreas/RAWseries_' + str(we_use[0]) + '.csv', header=None)[
                 1:].values.astype('single')
        adata2 = pd.read_csv('/home/JiangCongCong/data/pancreas/RAWseries_' + str(we_use[1]) + '.csv', header=None)[
                 1:].values.astype('single')
        Alldata = np.concatenate([adata1.T, adata2.T])  # 细胞数，基因数
        BID_data = np.array(Alldata)

        # label
        label1 = pd.read_csv('/home/JiangCongCong/data/pancreas/realseries_' + str(we_use[0]) + '.csv', header=None)[
                 1:].values
        label2 = pd.read_csv('/home/JiangCongCong/data/pancreas/realseries_' + str(we_use[1]) + '.csv', header=None)[
                 1:].values
        Alllabel = np.concatenate([label1, label2])

        genename = pd.read_csv('/home/JiangCongCong/data/pancreas/pancreas_genename.csv', header=None)[1:][0].values

        # batch
        Allbatch = np.concatenate([np.zeros(label1.shape[0]), np.zeros(label2.shape[0]) + 1])
        BID_batch = np.array(Allbatch)

        # ok, we select some interesting cell types
        chosen_cluster = ['alpha', 'beta', 'ductal', 'acinar', 'delta', 'gamma', 'endothelial', 'epsilon']

        chosen_index = np.arange(Alllabel.shape[0])
        for i in range(Alllabel.shape[0]):
            if Alllabel[i] in chosen_cluster:
                chosen_index[i] = 1  # 只挑选感兴趣的细胞类型
            else:
                chosen_index[i] = 0
        Alldata = Alldata[chosen_index == 1, :]
        Allbatch = Allbatch[chosen_index == 1]
        Alllabel = Alllabel[chosen_index == 1]
        ###############################################################################
        # and them, use numbers to replace the name of cell types
        Numlabels = np.zeros(Alllabel.shape[0])
        cluster_index2 = {'alpha': 0, 'beta': 1, 'ductal': 2, 'acinar': 3, 'delta': 4, 'gamma': 5, 'endothelial': 6,
                          'epsilon': 7}
        for i in range(Alllabel.shape[0]):
            Numlabels[i] = cluster_index2[Alllabel[i][0]]
            # Numlabels[i] = cluster_index2[Alllabel[i]]
        y = np.array(Numlabels)
        # BID_labels = [int(x) for item in BID_label for x in item] #

        # BID_batches = [int(x) for item in BID_batch for x in item] #

        anndata = sc.AnnData(pd.DataFrame(Alldata, columns=genename))
        anndata.raw = anndata
        sc.pp.normalize_total(anndata, target_sum=1e4)
        sc.pp.log1p(anndata)
        sc.pp.highly_variable_genes(anndata, n_top_genes=3000)
        highvar = anndata.var.highly_variable
        adata = anndata[:, highvar]
        x=adata.X
    n_clusters = np.unique(y).size
    X_input = torch.Tensor(np.array(x))
    return X_input,y,adata,n_clusters,cell_name



def prepro(filename):
    data_path = filename
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_type1"])
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)
    return X, cell_label,cell_name

def read_data(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index = decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index = decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns
def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data
def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d
class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn
decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
#======================载入9个批次的human_pancreas数据集,自己用gpt生成的=======================
def load_batch_human_pancreas(filename):#据集来自Benchmarking atlas-level data integration in single-cell genomics
    """
    提取密集矩阵格式的单细胞数据

    参数:
    filename: HDF5文件路径

    返回:
    包含所有数据的字典
    """
    with h5py.File(filename, 'r') as f:
        # 1. 提取基因表达矩阵 (密集矩阵)
        # 优先从X获取，如果没有则从layers/counts获取
        if 'X' in f:#X中存储预处理后的数据，数据集来自Benchmarking atlas-level data integration in single-cell genomics
            expr_matrix = f['X'][...]  # 直接读取密集矩阵
        elif 'layers' in f and 'counts' in f['layers']:#代表未预处理的数据
            expr_matrix = f['layers/counts'][...]  # 从layers/counts读取
        else:
            raise ValueError("未找到表达矩阵数据")

        # 2. 提取细胞名称 (obs/_index)
        cell_names = f['obs/_index'][...]
        # 解码字节字符串
        if isinstance(cell_names[0], bytes):
            cell_names = np.array([name.decode('utf-8') for name in cell_names])

        # 3. 提取基因名称 (var/_index)
        gene_names = f['var/_index'][...]
        if isinstance(gene_names[0], bytes):
            gene_names = np.array([name.decode('utf-8') for name in gene_names])

        # 4. 提取细胞类型数字标签 (obs/celltype)
        celltype_labels = f['obs/celltype'][...]

        # 5. 提取细胞类型名称 (obs/__categories/celltype)
        celltype_categories = f['obs/__categories/celltype'][...]
        if isinstance(celltype_categories[0], bytes):
            celltype_categories = np.array([cat.decode('utf-8') for cat in celltype_categories])

        # 6. 提取技术信息 (obs/tech)
        tech_data = f['obs/tech'][...]
        if isinstance(tech_data[0], bytes):
            tech_data = np.array([tech.decode('utf-8') for tech in tech_data])

        # 7. 提取大小因子 (obs/size_factors)
        size_factors = f['obs/size_factors'][...]

        return {
            'expr_matrix': expr_matrix,  # 密集表达矩阵 16382×19093
            'cell_names': cell_names,  # 细胞名称 16382
            'gene_names': gene_names,  # 基因名称 19093
            'celltype_labels': celltype_labels,  # 细胞类型数字标签 16382
            'celltype_categories': celltype_categories,  # 细胞类型名称 14
            'tech_data': tech_data,  # 技术信息
            'size_factors': size_factors  # 大小因子
        }

#数据增强dropout=================================================================================
def x_drop(x, p=0.2):
    mask_list = [torch.rand(x.shape[1]) < p for _ in range(x.shape[0])]
    mask = torch.vstack(mask_list)
    new_x = x.clone()
    #new_x = x
    new_x[mask] = 0.0
    return new_x
#细胞标签重新排序为0到n_cluster(以防止细胞标签不为int，防止为不连续整数)=======================================================================================================
def adjust_labels(labels):
    unique_labels = np.unique(labels)  # Find all the different tags and sort them
    unique_labels_sorted = np.sort(unique_labels)
    new_labels = np.zeros_like(labels, dtype=np.int64)  # Create a new label tensor with data type long
    for i in range(len(labels)):  # Adjust label range
        label = labels[i]
        new_label = np.where(unique_labels_sorted == label)[0][0]
        new_labels[i] = new_label
    return new_labels
#------------------------------------------------------------------------------------------------------------------
#单细胞数据预处理
#函数定义read_dataset
def read_dataset(adata, transpose=False, test_split=False, copy=False):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error

    if adata.X.size < 50e6:  # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(float) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(float) == adata.X), norm_error

    if transpose: adata = adata.transpose()

    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['DCA_split'] = spl.values
    else:
        adata.obs['DCA_split'] = 'train'

    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    print('### Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata


def normalize_cell(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        #sc.pp.normalize_per_cell(adata)
        sc.pp.normalize_total(adata, target_sum=1e4)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata
#预处理数据集=========================================
#代码来自scMMN
def pre_normalize(adata, copy=True, highly_genes = None, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata
#余弦相似度（不完全是）==========================================================================================================
def consine_similarity(Z, center):
    center = center.float()
    #similarity = torch.mm(Z.to('cpu').detach(),(torch.from_numpy(center).T))
    similarity = torch.mm(Z, (center).T)
    return similarity


#OT损失================================================================================================================================================================================
def sinkhorn(out, epsilon, sinkhorn_iterations):
    """
    from https://github.com/facebookresearch/swav
    """
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    # Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

def w_dist(cos_dist, T, m,args, eps=1):
    temp_1 = torch.mm(cos_dist.t(), T)
    temp_2 = eps * torch.mm(T.t(), torch.log(T))
    a = torch.eye(m).to(args.device)
    b = a * temp_1
    c = a * temp_2
    distance = torch.sum(b)
    entropy = torch.sum(c)
    return distance, entropy

#将数据处理成多视图数据集形式
def get_multiview_data(mv_data, batch_size):
    num_views = len(mv_data.data_views)  # 视图数量
    num_samples = len(mv_data.labels)  # 样本数量
    num_clusters = len(np.unique(mv_data.labels))  # 聚类数量

    # 创建数据加载器
    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        #collate_fn=multiview_collate_fn
    )

    return mv_data_loader, num_views, num_samples, num_clusters
# 自定义 collate_fn
def multiview_collate_fn(batch):
    views = [torch.stack([item[0][i] for item in batch]) for i in range(len(batch[0][0]))]
    labels = torch.stack([item[1] for item in batch])
    return views, labels
class MultiviewDataset(Dataset):
    def __init__(self, data_views, labels):
        """
        初始化多视图数据集。

        参数:
            data_views (list): 多个视图的数据，例如 [X_1, X_2]。
            labels (list or np.array): 样本的标签。
        """
        self.data_views = data_views  # 多个视图的数据
        self.labels = labels  # 样本的标签

    def __len__(self):
        """返回数据集的大小。"""
        return len(self.labels)

    def __getitem__(self, index):
        """
        返回第 index 个样本的多视图数据和标签。

        参数:
            index (int): 样本的索引。

        返回:
            views (list): 第 index 个样本的多个视图数据。
            label (int): 第 index 个样本的标签。
        """
        views = [view[index] for view in self.data_views]  # 获取第 index 个样本的多个视图数据
        label = self.labels[index]  # 获取第 index 个样本的标签
        return views, label
def check_X_range_1(adata):
    X = adata.X

    if sp.sparse.issparse(X):
        X_data = X.data  # 稀疏矩阵只需要取非零元素
    else:
        X_data = np.array(X)  # 稠密矩阵直接转 NumPy

    has_negative = np.any(X_data < 0)
    min_val = X_data.min()
    max_val = X_data.max()

    print(f"是否存在负值: {'✅ 有' if has_negative else '❌ 没有'}")
    print(f"最小值: {min_val:.4f}")
    print(f"最大值: {max_val:.4f}")

    return has_negative, min_val, max_val

def check_X_range(adata):
    #X = adata.raw.X if adata.raw is not None else adata.X
    X = adata.X
    if sp.sparse.issparse(X):
        X_data = X.data  # 稀疏矩阵只检查非零元素
    else:
        X_data = np.array(X)  # 稠密矩阵转 NumPy

    has_negative = np.any(X_data < 0)
    has_inf = np.any(np.isinf(X_data))
    has_nan = np.any(np.isnan(X_data))

    min_val = np.nanmin(X_data)
    max_val = np.nanmax(X_data)

    print(f"是否存在负值: {'✅ 有' if has_negative else '❌ 没有'}")
    print(f"是否存在 inf: {'⚠️ 有无穷大值' if has_inf else '❌  没有'}")
    print(f"是否存在 NaN: {'⚠️ 有NaN' if has_nan else '❌  没有'}")
    print(f"最小值（排除inf/NaN）: {min_val:.4f}")
    print(f"最大值（排除inf/NaN）: {max_val:.4f}")

    # return {
    #     "has_negative": has_negative,
    #     "has_inf": has_inf,
    #     "has_nan": has_nan,
    #     "min": min_val,
    #     "max": max_val
    # }
def my_func(adata):
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=200)
    #sc.pp.recipe_zheng17
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    # sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    #sc.tl.pca(adata)
    #sce.pp.harmony_integrate(adata, 'batch')


    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=True)

    adata = normalize_cell(adata,
                           size_factors=True,
                           normalize_input=True,
                           logtrans_input=True)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    highvar = adata.var.highly_variable
    adata = adata[:, highvar]
    return adata
#代码来自scHetG
def load_batch_dataset(dataset):
    if dataset == 'mouse_pancreas':#adata1细胞个数为822，adata细胞个数为1064，细胞总个数1886,批次差异没有
        csv_data = pd.read_csv('/home/JiangCongCong/data/mouse_pancreas/GSM2230761_mouse1_umifm_counts.csv')
        barcode = csv_data['barcode'].values
        assigned_cluster = csv_data['assigned_cluster'].values
        gene_expression = csv_data.iloc[:, 3:].values
        adata1 = ad.AnnData(X=gene_expression, obs={'barcode': barcode, 'celltype': assigned_cluster})
        csv_data = pd.read_csv('/home/JiangCongCong/data/mouse_pancreas/GSM2230762_mouse2_umifm_counts.csv')
        barcode = csv_data['barcode'].values
        assigned_cluster = csv_data['assigned_cluster'].values
        gene_expression = csv_data.iloc[:, 3:].values
        adata2 = ad.AnnData(X=gene_expression, obs={'barcode': barcode, 'celltype': assigned_cluster})
        adata = [adata1, adata2]
        adata = ad.concat(adata, merge="same")
        adata.obs['batch'] = np.concatenate((np.array(['batch1'] * adata1.n_obs), np.array(['batch2'] * adata2.n_obs)), axis=0)
        adata.var_names = csv_data.columns[3:]
        batch_name = np.array(['batch1', 'batch2'], dtype=object)

    elif dataset == 'mouse_atlas':#adata1细胞数量4239，adata2细胞数量为2715，adata细胞数量为6954
        adata1 = ad.read_h5ad('/home/JiangCongCong/data/mouse_atlas/rna_seq_mi.h5ad')
        adata2 = ad.read_h5ad('/home/JiangCongCong/data/mouse_atlas/rna_seq_sm.h5ad')
        adata = [adata1, adata2]
        adata = ad.concat(adata, merge="same")
        adata.obs['batch'] = np.concatenate((np.array(['batch1']*adata1.n_obs), np.array(['batch2']*adata2.n_obs)), axis=0)
        batch_name = np.array(['batch1', 'batch2'], dtype=object)

    elif dataset == 'mouse_brain':#adata1是33231个细胞，adata2是6769个细胞，共四万个细胞
        adata = sc.read_h5ad("/home/JiangCongCong/data/mouse_brain/sub_mouse_brain.h5ad")
        adata.obs.rename(columns={'BATCH': 'batch'}, inplace=True)
        adata1 = adata[adata.obs['batch'].values == 'batch1']
        adata2 = adata[adata.obs['batch'].values == 'batch2']
        adata = ad.concat([adata1, adata2], merge='same')
        batch_name = np.array(['batch1', 'batch2'], dtype=object)

    elif dataset == 'human_pancreas_2':#adata1是8569个细胞，adata2是2127个细胞，共10696个细胞
        adata1 = ad.read_h5ad('/home/JiangCongCong/data/human_pancreas_2/rna_seq_baron.h5ad')
        adata2 = ad.read_h5ad('/home/JiangCongCong/data/human_pancreas_2/rna_seq_segerstolpe.h5ad')
        adata = [adata1, adata2]
        adata = ad.concat(adata, merge="same")
        adata.obs.rename(columns={'cell_type1': 'celltype'}, inplace=True)
        adata.obs['batch'] = np.concatenate((np.array(['batch1']*adata1.n_obs), np.array(['batch2']*adata2.n_obs)), axis=0)
        batch_name = np.array(['batch1', 'batch2'], dtype=object)

    elif dataset == 'human_pancreas':#共8451个细胞
        adata = sc.read("/home/JiangCongCong/data/human_pancreas/human_pancreas.h5ad")
        batch_name = np.array(['human1', 'human2', 'human3', 'human4'], dtype=object)

    elif dataset == 'human_lung':#总共3202
        adata = sc.read_h5ad("/home/JiangCongCong/data/human_lung/human_lung_marker.h5ad")
        adata.obs['batch'].replace('muc3843', 'batch1', inplace=True)
        adata.obs['batch'].replace('muc4658', 'batch2', inplace=True)
        adata.obs['batch'].replace('muc5103', 'batch3', inplace=True)
        adata.obs['batch'].replace('muc5104', 'batch4', inplace=True)
        batch_name = np.array(['batch1', 'batch2', 'batch3', 'batch4'], dtype=object)


    elif dataset == 'human_heart':#共39187个细胞
        adata = sc.read("/home/JiangCongCong/data/human_heart/healthy_human_heart.h5ad")
        adata.obs.rename(columns={'sampleID': 'batch'}, inplace=True)
        unique_sampleIDs = adata.obs['batch'].values.unique()[-10:]
        adata = adata[adata.obs['batch'].isin(unique_sampleIDs)]
        batch_name = unique_sampleIDs
        for i in range(len(batch_name)):
            adata.obs['batch'].replace(batch_name[i], 'batch'+str(i+1), inplace=True)
        batch_name=np.array(['batch1', 'batch2', 'batch3','batch4','batch5','batch6','batch7','batch8','batch9','batch10'])


    start = time()
    adata_c = my_func(adata)

    #corrd = pd.DataFrame(adata_c.obsm['X_pca_harmony'])
    adata_batch = ad.AnnData(adata_c.X, obs=adata_c.obs, dtype='float64')
    #adata_corrd.obsm['X_pca_harmony'] = adata_c.obsm['X_pca_harmony']
    adata_batch.obs['celltype'] = np.array(adata_c.obs['celltype'])
    adata_batch.obs['batch'] = np.array(adata_c.obs['batch'])
    unique_celltypes = adata_batch.obs['celltype'].unique()
    return adata.X,adata_batch.obs['celltype'],adata,len(unique_celltypes)