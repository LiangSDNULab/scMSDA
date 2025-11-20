import argparse
import utils
import metrics
from models import scMSDA
import torch.optim as optim
from time import time as get_time
import warnings
from losses import cluster_contrastive_loss, Dgan_loss, OT_Loss, Recon_loss
from datasets import *
import time
import torch
import numpy as np
import matplotlib
from utils import GPUManager, show_info
matplotlib.use('Agg')

# 固定随机种子
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='DCCL Super Parameters')
parser.add_argument("--version", default="all")
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=3e-4, metavar='LR', help='learning rate (default: 1e-4)')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 0)')  #
parser.add_argument('--weight_decay', type=float, default=0, metavar='M', help='weight decay (default: 0)')
parser.add_argument('--data_file', default=f'dataset/')
parser.add_argument('--cluster_dims', type=int, default=512)
parser.add_argument('--input_dim', default=[3000, 3000])
parser.add_argument('--embedding_dims', default=[2048, 1024, 512])  # [1024, 1024, 512])
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--save_args", default=1)
parser.add_argument("--p", default=0.1)
parser.add_argument("--lambda1", default=6)
parser.add_argument("--lambda2", default=0.6)
parser.add_argument("--lambda3", default=0.6)
parser.add_argument("--neighbors", default=10)
parser.add_argument("--tau", default=0.8)
parser.add_argument("--alpha", default=0.55)
parser.add_argument("--beta", default=0.4)

args = parser.parse_args()

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {args.device}")

time_start = get_time()  ###开始计时
start_memory2 = show_info()
print("开始内存：%fMB" % (start_memory2))


for args.dataset in ["Wang_Lung","Quake_10x_Bladder","Quake_Smart-seq2_Limb_Muscle","Quake_Smart-seq2_Trachea"]:
    for args.seed in [42]:  # 4108,3407,215,114514,42
        X_list = []
        utils.set_seed(args.seed)

        print('=========================================')
        print(args)
        print('=========================================')

        start_time = time.time()
        X, y, adata, n_clusters = load_scRNAdata(args.dataset, args)
        Y = adjust_labels(y)
        adata.obs['Group'] = Y
        X_1 = x_drop(X, args.p)
        args.input_dim = [X.shape[1], X_1.shape[1]]
        X = [X.numpy(), X_1.numpy()]
        view = len(X)
        n_clusters = np.unique(Y).size
        print('The clusters of datasets:', n_clusters)
        for i in range(len(X)):
            print(X[i].shape)
            X[i] = torch.from_numpy(X[i]).float().to(args.device)

        best_acc = -1
        best_nmi = -1  # 初始值可以根据具体需求调整
        best_ari = -1
        best_nmi_epoch = -1
        best_ari_epoch = -1
        best_acc_epoch = -1
        best_ari_z_fusion = None
        best_nmi_z_fusion = None
        best_acc_z_fusion = None
        loss_rc_list, loss_cc_list, loss_cl_list, loss_ot, loss_loss_list = [], [], [], [], []
        acc_list, nmi_list, ari_list = [], [], []

        model = scMSDA(view, args.input_dim, args.embedding_dims,
                      args.cluster_dims, n_clusters, args.device).to(
            args.device)


        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        model.train()

        for epoch in range(args.epochs):
            loss_rc, loss_cc, loss_cl, loss_loss = 0, 0, 0, 0
            loss_list = []
            all_common_z = []
            X1, X2 = X[0].to(args.device), X[1].to(args.device)
            mv_data = MultiviewDataset(data_views=[X1, X2], labels=Y)
            t0 = time.time()
            mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, args.batch_size)
            total_enc_time = 0.0
            total_dec_time = 0.0
            total_fusion_time = 0.0
            total_recon_time = 0.0
            total_cluster_time = 0.0
            total_cwcl_time = 0.0
            total_ot_time = 0.0
            for batch_idx, (sub_data_views, _) in enumerate(mv_data_loader):
                optimizer.zero_grad()
                batch_loss = 0
                xrs, zs = model(sub_data_views)
                common_z = model.fusion(zs)
                recon_loss = Recon_loss(sub_data_views, xrs)


                cl, _, _ = model.cluster_alignment(common_z, zs)
                batch_psedo_label = torch.argmax(cl, dim=1)
                # 融合表示的聚类中心
                k_centers_i = model.compute_centers(zs[0], batch_psedo_label)  # 视图 i 的聚类中心
                k_centers_j = model.compute_centers(zs[1], batch_psedo_label)
                q_centers = model.compute_centers(common_z, batch_psedo_label)

                dcd_loss_1 = model.compute_cluster_loss(q_centers, k_centers_i, batch_psedo_label)
                dcd_loss_2 = model.compute_cluster_loss(q_centers, k_centers_j, batch_psedo_label)
                dcd_loss = dcd_loss_1 + dcd_loss_2

                dgan_Loss = Dgan_loss(zs, common_z, args)

                OT_loss = OT_Loss(zs, common_z, q_centers, args)

                Loss_total = args.lambda1 * recon_loss + args.lambda2 * dcd_loss + args.lambda3 * dgan_Loss + OT_loss

                Loss_total.backward()
                optimizer.step()

            if (epoch + 1) % 1 == 0:
                model.eval()
                score, latent_fusion, predict_labels = metrics.evaluation(model, X, Y, args.device)

                print("epoch:%.0f" % (epoch + 1))
                print(score)

                # 获取当前 epoch 的 NMI 和 ARI
                nmi = score['kmeans']['NMI']  # 需要根据实际返回值的结构调整
                ari = score['kmeans']['ARI']  # 需要根据实际返回值的结构调整
                acc = score['kmeans']['accuracy']

                if acc > best_acc:
                    best_acc = acc
                    best_acc_epoch = epoch + 1  # 记录最大 NMI 对应的 epoch
                    best_acc_score = score
                    best_acc_z_fusion = latent_fusion  # yiqian shi common_z
                    best_acc_pseudo_labels = predict_labels

                # 更新最大 NMI 和 ARI 及其对应的 epoch
                if nmi > best_nmi:
                    best_nmi = nmi
                    best_nmi_epoch = epoch + 1  # 记录最大 NMI 对应的 epoch
                    best_nmi_score = score
                    best_nmi_z_fusion = latent_fusion
                    best_nmi_pseudo_labels = predict_labels
                if ari > best_ari:
                    best_ari = ari
                    best_ari_epoch = epoch + 1  # 记录最大 ACC 对应的 epoch
                    best_ari_score = score
                    best_ari_z_fusion = latent_fusion
                    best_ari_pseudo_labels = predict_labels

            t = int(time.time() - t0)



        print("dataset:", args.dataset)
        print(f"Highest NMI: {best_nmi:.6f} in epoch {best_nmi_epoch}")
        print(f"Highest ARI: {best_ari:.6f} in epoch {best_ari_epoch}")
        print(f"Highest ACC: {best_acc:.6f} in epoch {best_acc_epoch}")
        print(f"Highest ACC_score: {best_acc_epoch} in epoch {best_acc_score}")
        print(f"Highest NMI_score: {best_nmi_epoch} in epoch {best_nmi_score}")
        print(f"Highest ARI_score: {best_ari_epoch} in epoch {best_ari_score}")



    zuihou = show_info()
    print("使用内存：{}%MB".format(zuihou - start_memory2))
    print("结束内存：{}%MB".format(zuihou))


