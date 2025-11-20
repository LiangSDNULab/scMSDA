import sys
import torch
import numpy as np
from sklearn import metrics
from munkres import Munkres
import torch.nn.functional as F
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
from sklearn.manifold import TSNE
#import leidenalg
import igraph as ig
import numpy as np
import sys
from kmeans_gpu import kmeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        
        xrs, zs, cs = model(X)
        
        z = torch.stack(cs, dim=0)
        rs = torch.mean(z, dim=0)
        pred = torch.argmax(rs, dim=1).cpu().numpy()

        nmi = metrics.normalized_mutual_info_score(y, pred)
        ari = metrics.adjusted_rand_score(y, pred)
        f = metrics.fowlkes_mallows_score(y, pred)
        pred_adjusted = get_y_preds(y, pred, len(set(y)))
        acc = metrics.accuracy_score(pred_adjusted, y)

    return nmi, ari, f, acc

def evaluation(model, X, Y, device):
    model.eval()


    with torch.no_grad():
        #xrs, zs, cs = model(X)
        xrs, zs = model(X)
        common_z = model.fusion(zs).cpu().numpy()
        scores,pseudo_labels= clustering([common_z], Y)



    return scores,common_z,pseudo_labels

def clustering(x_list, y):
    global fig_name
    n_clusters = np.size(np.unique(y))
    x_final_concat = np.concatenate(x_list[:], axis=1)
   
    kmeans_assignments, km = get_cluster_sols(x_final_concat, ClusterClass=KMeans, n_clusters=n_clusters,
                                            init_args={'n_init': 10})
    if np.min(y) == 1:
        y = y - 1
    scores, _ = clustering_metric(y, kmeans_assignments, n_clusters)

    
    ret = {}
    ret['kmeans'] = scores

    return ret,kmeans_assignments

def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(C[:, j])  
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix

def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels

def get_y_preds(y_true, cluster_assignments, n_clusters):
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments.astype(int)]
    return y_pred

def classification_metric(y_true, y_pred, average='macro', verbose=True, decimals=4):
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    accuracy = np.round(accuracy, decimals)

    precision = metrics.precision_score(y_true, y_pred, average=average)
    precision = np.round(precision, decimals)

    recall = metrics.recall_score(y_true, y_pred, average=average)
    recall = np.round(recall, decimals)

    f_score = metrics.f1_score(y_true, y_pred, average=average)
    f_score = np.round(f_score, decimals)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_measure': f_score}, confusion_matrix

def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
    cluster_assignments = None
    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args)
        for _ in range(10):          # 10 
            try:
                cluster_obj.fit(x)
                break
            except:
                print("Unexpected error:", sys.exc_info())
        else:
            return np.zeros((len(x),)), cluster_obj

    cluster_assignments = cluster_obj.predict(x)
    return cluster_assignments, cluster_obj


def get_cluster_sols_after(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
    cluster_assignments = None

    # 如果没有提供已有的聚类对象，使用 Leiden 算法
    if cluster_obj is None:
        # 将数据 x 转换为图数据（邻接矩阵或k-近邻图）
        # 假设 x 是一个 n x m 的数据矩阵（n 个样本，m 个特征）

        # 创建一个邻接矩阵，作为图的基础（这里使用欧几里得距离计算邻接矩阵）
        from sklearn.neighbors import NearestNeighbors

        # 使用 k-近邻构建图
        knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
        knn.fit(x)
        adjacency_matrix = knn.kneighbors_graph(x).toarray()

        # 创建 igraph 图对象
        g = ig.Graph.Weighted_Adjacency(adjacency_matrix.tolist(), mode=ig.ADJ_UNDIRECTED, weight_attr='weight')

        # 使用 Leiden 算法进行社区检测
        try:
            partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, **init_args)
            cluster_assignments = np.array(partition.membership)  # 获取每个节点的社区标签
        except Exception as e:
            print("Unexpected error:", e)
            return np.zeros((len(x),)), cluster_obj

    return cluster_assignments, cluster_obj
def clustering_metric(y_true, y_pred, n_clusters, verbose=True, decimals=4):
    y_pred_ajusted = get_y_preds(y_true, y_pred, n_clusters)#获得对齐的标签

    classification_metrics, confusion_matrix = classification_metric(y_true, y_pred_ajusted)

    ami = metrics.adjusted_mutual_info_score(y_true, y_pred)
    ami = np.round(ami, decimals)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    nmi = np.round(nmi, decimals)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ari = np.round(ari, decimals)
    pur = calculate_purity(y_true, y_pred)
    pur = np.round(pur, decimals)

    return dict({'AMI': ami, 'NMI': nmi, 'ARI': ari, 'Purity': pur}, **classification_metrics), confusion_matrix

def calculate_purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster_index in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster_index], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster_index] = winner

    return metrics.accuracy_score(y_true, y_voted_labels)
#===============================phi    eva评估函数===========================================================================
def phi(feature, true_labels, cluster_num):
    predict_labels, centers = kmeans(
        X=feature, num_clusters=cluster_num, distance="euclidean", device="cuda")
    acc, nmi, ari, f1 = eva(
        true_labels, predict_labels.numpy(), show_details=True)
    scores, _ = clustering_metric(true_labels, predict_labels.numpy(), cluster_num)
    return scores,acc, nmi, ari, f1, predict_labels.numpy(), centers
def eva(y_true, y_pred, show_details=True):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if show_details:
        print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
              ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1
def normalize(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x
def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro
#===============================