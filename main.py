import argparse
import utils
import metrics
from models import scMSDA
import torch.optim as optim
import warnings
from losses import Cwcl_loss,OT_Loss,Recon_loss
from datasets import *
import time
import torch
import numpy as np
import os
import matplotlib

matplotlib.use('Agg')

# 固定随机种子
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='scMSDA Super Parameters')
parser.add_argument("--version", default="1")
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=3e-4, metavar='LR', help='learning rate (default: 1e-4)')
parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: 0)')
parser.add_argument('--weight_decay', type=float, default=0, metavar='M', help='weight decay (default: 0)')  #
parser.add_argument('--device', type=str, default='cuda:1', metavar='M')
parser.add_argument('--data_file', default=f'dataset/')
parser.add_argument('--cluster_dims', type=int, default=512)
parser.add_argument('--input_dim', default=[3000, 3000])
parser.add_argument('--embedding_dims', default=[2048, 1024, 512])#[1024, 1024, 512])
parser.add_argument("--temperature_f", default=0.5)  # 加权自适应模块的温度参数,
parser.add_argument("--save_args", default=1)
parser.add_argument("--p", default=0.1)
parser.add_argument("--lambda0", default=3)#3
parser.add_argument("--lambda1", default=0.3)#cluster_loss
parser.add_argument("--lambda2", default=0.3)#cwcl_Loss
parser.add_argument("--lambda3", default=0.5)#OT_loss0.5
parser.add_argument("--neighbors",default=10)
parser.add_argument("--run", default=0)
parser.add_argument("--tau", default=0.8)
parser.add_argument("--alpha ", default=0.55)
parser.add_argument("--beta", default=0.4)
args = parser.parse_args()




#"deng", "MCA","mouse_bladder_cell","mouse_ES_cell","Mouse_Pancreas_cell_1886","Muraro","Quake_10x_Bladder","Quake_Smart-seq2_Limb_Muscle", "Quake_Smart-seq2_Trachea", "Romanov","Wang_Lung","sc10X","goolam","Human_Pancreas_cell_2","klein"
for args.dataset in [ "Wang_Lung","sc10X","goolam","Human_Pancreas_cell_2","klein"]:

    for args.run in [10]:#range(10)
        X_list = []
        #utils.set_seed(args.seed)

        # print('=========================================')
        # print(args)
        # print('=========================================')

        start_time = time.time()
        X, Y,adata,n_clusters = loda_scRNAdata(args.dataset, args)


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
        # 保存到文件

        model.train()

        for epoch in range(args.epochs):
            loss_rc, loss_cc, loss_cl, loss_loss = 0, 0, 0, 0
            loss_list = []
            optimizer.zero_grad()
            for i in range(0, view - 1):
                for j in range(i + 1, view):
                    X1, X2 = X[i].to(args.device), X[j].to(args.device)
                    mv_data = MultiviewDataset(data_views=[X1, X2], labels=Y)
                    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, args.batch_size)

                    for batch_idx, (sub_data_views,_) in enumerate(mv_data_loader):

                        output = model(sub_data_views)
                        xrs, zs=output[0],output[1]
                        common_z = model.fusion(zs)
                        recon_loss = Recon_loss(sub_data_views,xrs)
                        cl, _, _ = model.cluster_alignment(common_z, zs)

                        with torch.no_grad():
                            batch_psedo_label = torch.argmax(cl, dim=1)

                            k_centers_i = model.compute_centers(zs[0], batch_psedo_label)  # 视图 i 的聚类中心
                            k_centers_j = model.compute_centers(zs[1], batch_psedo_label)

                        q_centers = model.compute_centers(common_z, batch_psedo_label)
                        cl_loss_1 = model.compute_cluster_loss(q_centers, k_centers_i, batch_psedo_label)
                        cl_loss_2 = model.compute_cluster_loss(q_centers, k_centers_j, batch_psedo_label)
                        cl_loss = cl_loss_1 + cl_loss_2

                        #加权自适应实例级对比损失
                        CWCL_Loss =Cwcl_loss(zs,common_z,args)

                        #OT_loss
                        OT_loss = OT_Loss(zs, common_z, q_centers, args)

                        Loss_total=args.lambda0*recon_loss+args.lambda1*cl_loss+ args.lambda2*CWCL_Loss+ args.lambda3*OT_loss




            Loss_total.backward()
            optimizer.step()

            if (epoch + 1) % 1 == 0:
                score, latent_fusion, pseudo_labels = metrics.evaluation(model, X, Y, args.device)
                print("epoch:%.0f" % (epoch + 1))
                print(score)

                # 获取当前 epoch 的 NMI 和 ARI
                nmi = score['kmeans']['NMI']
                ari = score['kmeans']['ARI']
                acc = score['kmeans']['accuracy']
                if acc > best_acc:
                    best_acc = acc
                    best_acc_epoch = epoch + 1  # 记录最大 NMI 对应的 epoch
                    best_acc_score = score
                    best_acc_z_fusion = latent_fusion
                    best_acc_pseudo_labels = pseudo_labels

                # 更新最大 NMI 和 ARI 及其对应的 epoch
                if nmi > best_nmi:
                    best_nmi = nmi
                    best_nmi_epoch = epoch + 1  # 记录最大 NMI 对应的 epoch
                    best_nmi_score = score
                    best_nmi_z_fusion = latent_fusion
                    best_nmi_pseudo_labels = pseudo_labels
                if ari > best_ari:
                    best_ari = ari
                    best_ari_epoch = epoch + 1  # 记录最大 ACC 对应的 epoch
                    best_ari_score = score
                    best_ari_z_fusion = latent_fusion
                    best_ari_pseudo_labels = pseudo_labels

        # 保存ari和nmi值最高的性能指标
        path = f"./result/"
        csv_dir = f"./result/result.csv"
        csv_path = os.path.dirname(csv_dir)
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        if args.save_args == 1:

            file = open(csv_dir, "a+")
            file.write(f"dataset: {args.dataset}\n")
            file.write(f"run: {args.run}\n")
            file.close()
            write_results(csv_dir, args, best_nmi, best_nmi_epoch, best_ari, best_ari_epoch,
                          best_acc, best_acc_epoch, best_acc_score, best_nmi_score, best_ari_score)
        if args.save_args == 1:
            target_dir = os.path.join(path, "embedding"
                                      )
            # 检查文件夹是否存在，如果不存在则创建
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            h5_path = os.path.join(target_dir, f'{args.dataset}_results.h5')
            # 以写入模式 ('w') 打开 h5 文件
            with h5py.File(h5_path, 'w') as h5f:
                # 保存嵌入向量
                h5f.create_dataset("embedding/best_acc_embedding", data=best_acc_z_fusion)

                # 保存伪标签
                h5f.create_dataset("pseudo_labels/best_acc_label", data=best_acc_pseudo_labels)

        print("dataset:", args.dataset)
        print(f"Highest NMI: {best_nmi:.6f} in epoch {best_nmi_epoch}")
        print(f"Highest ARI: {best_ari:.6f} in epoch {best_ari_epoch}")
        print(f"Highest ACC: {best_acc:.6f} in epoch {best_acc_epoch}")
        print(f"Highest ACC_score: {best_acc_epoch} in epoch {best_acc_score}")
        print(f"Highest NMI_score: {best_nmi_epoch} in epoch {best_nmi_score}")
        print(f"Highest ARI_score: {best_ari_epoch} in epoch {best_ari_score}")

