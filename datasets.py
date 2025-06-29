from scipy import sparse
import h5py
import pandas as pd
from utils import *
import scipy as sp
from torch.utils.data import Dataset

def loda_scRNAdata(dataset,args):
    args.file_csv = [ "Romanov", "deng",  "goolam"]
    args.file_h5_1 = ["MCA","Human_Pancreas_cell_2", "Mouse_Pancreas_cell_1886","mouse_bladder_cell","mouse_ES_cell"]
    args.file_h5_2 = ["klein", "Muraro","Quake_10x_Bladder", "Quake_Smart-seq2_Limb_Muscle",  "Wang_Lung","Quake_Smart-seq2_Trachea"]
    args.file_sc10X = ["sc10X"]
    if args.dataset in args.file_csv:
        args.type = ".csv"
    elif args.dataset in args.file_h5_1 or args.dataset in args.file_h5_2:
        args.type = ".h5"
    elif args.dataset in args.file_sc10X:
        args.type = ".mat"
    else:
        raise ValueError("Dataset not found in the specified file lists.")
    data_name = dataset + args.type
    print("dataset_name:", dataset)
    file_path = os.path.join(args.data_file, data_name)
    if dataset in args.file_csv:
        data_mat = pd.read_csv(file_path, header=None, index_col=None)
        y = data_mat.iloc[1, 1:].to_numpy(dtype=int)
        x = data_mat.iloc[3:, 1:]
        x = x.T
        print("dataset_shape:", x.shape[0],x.shape[1])
        adata = sc.AnnData(x, dtype="float64")
        sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        highvar = adata.var.highly_variable
        adata = adata[:, highvar]
        x = adata.X
    elif dataset in args.file_h5_1:
        if dataset in ["MCA"]:
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
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        highvar = adata.var.highly_variable
        adata = adata[:, highvar]
        x = adata.X
    elif dataset in args.file_h5_2:
        x, y = prepro(file_path)
        x = np.ceil(x)
        print("dataset_shape:", x.shape[0], x.shape[1])
        adata = sc.AnnData(x, dtype=x.dtype)
        adata.obs['Group'] = adjust_labels(y)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        highvar = adata.var.highly_variable
        adata = adata[:, highvar]
        x=adata.X
    n_clusters = np.unique(y).size
    X_input = torch.Tensor(np.array(x))
    return X_input,y,adata,n_clusters
#prepro===============================================================================================

def prepro(filename):
    data_path = filename
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_type1"])
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)
    return X, cell_label

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

#数据增强dropout=================================================================================
def x_drop(x, p=0.2):
    mask_list = [torch.rand(x.shape[1]) < p for _ in range(x.shape[0])]
    mask = torch.vstack(mask_list)
    new_x = x.clone()
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
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata
#预处理数据集=========================================

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
#余弦相似度==========================================================================================================
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

