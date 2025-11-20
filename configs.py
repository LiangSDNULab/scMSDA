def get_config(args):
    args.input_dim = [3000, 3000]
    if args.dataset == 'synthetic3d':
        args.input_dim = [3, 3, 3]
        args.embedding_dims = [1024, 1024, 16]         # [1024, 1024, 128]
        args.cluster_dims = 256                         # 512
        args.temperature = 0.5
        args.batch_size = 180                            # 128
        args.lr = 0.0001
        args.alpha = 0.1
        args.beta = 1.0
        args.gamma = 1.0
    if args.dataset == 'Romanov':
        args.input_dim = [2000, 2000]
        args.embedding_dims = [1024, 1024,512]         # [1024, 1024, 128]#[512, 256, 128] 是0.85
        args.cluster_dims = 512                         # 512,256
        args.temperature = 0.5
        args.batch_size = 128                           # 128,180
        args.lr = 0.0001
        #args.alpha = 1.0
        #args.beta = 1.0
        #args.gamma = 1.0
        args.thea_1 = 10
        args.thea_2 = 2
        args.n_clusters = 7
    if args.dataset in ["biase","human_kidney_counts_top2000","zeisel"]:
        args.input_dim = [2000, 2000]
    if args.dataset =="deng":
        args.input_dim = [1974, 1974]
    if args.dataset =="goolam":
        args.input_dim = [1704, 1704]
    if args.dataset == "Camp":
        args.input_dim = [16270, 16270]
    # if args.dataset =="darmanis":
    #     args.input_dim = [1980, 1980]
    args.tau= 0.8
    args.alpha =0.55
    args.beta =0.4


    return args