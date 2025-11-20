# scMSDA


The pytorch version of scMSDA : A Novel Multi-View Fusion Framework for Single-Cell RNA-seq Data Clustering with Semantic and Distribution Alignment. <br/>

## Table of contents

- [Network diagram](#diagram)
- [Requirements](#requirements)
- [Dataset](#Dataset)
- [Usage](#usage)
- [Parameters](#parameters)

## <a name="diagram"></a>Network diagram


![model](https://github.com/user-attachments/assets/f1888850-30da-42ae-b0fa-1a65a817528c)



## <a name="requirements"></a>Requirements

torch==1.10.1 

pandas==2.3.0 

numpy==1.26.4 

scanpy: 1.10.3 

scikit-learn: 1.4.2

## <a name="dataset"></a>Dataset

Deng, Goolam, Human_Pancreas, Klein, MCA, Mouse_Bladder, Mouse_ES, Mouse_Pancreas, Muraro, Qx_Bladder, Qs_LM, Qs_Trachea, Romanov, Wang_Lung, sc10X, Chen and Human lung dataset

## <a name="Usage"></a>Usage

```
python main.py
```

## <a name="parameters"></a>Parameters

**--batch_size:** batch size, default =512.<br/>

**--data_file:** file name of data.<br/>

**--epochs:** max number of iterations, default = 500.<br/>

**--p:** For data augmentation, dropout regularization is applied with a dropout probability of p = 0.1.

**--lambda1:** Weight of reconstruction loss, default = 6.

**--lambda2:** Weight of dynamic center-driven multi-view consistency loss (Ldcd), default = 0.6.

**--lambda3:** Weight of distance-guided adaptive negatives contrastive loss (Ldgan), default = 0.6.

**--neighbors:** Number of neighbors in Distance-Guided Adaptive Negatives Contrastive Learning, default = 10.



