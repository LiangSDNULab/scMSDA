# scMSDA
scMSDA:A Novel Multi-View Fusion Framework for Single-Cell RNA-seq Data Clustering with Semantic and Distribution Alignment

## scMSDA Model
![model_1](https://github.com/user-attachments/assets/33b1c0dd-c425-43b3-abbc-2071423e6de6)
Single-cell RNA sequencing technology have improved cellular heterogeneity resolution but face challenges like high dimensionality, sparsity, and technical noise in downstream analysis. Existing methods often treat all negative samples equally, ignoring local structures that are essential for capturing meaningful semantic relationships within the data. In this paper, we propose scMSDA, a novel multi-view fusion framework for single-cell RNA-seq data clustering, which leverages semantic consistency and distribution alignment to effectively learn robust representations for downstream tasks. Our model first performs data augmentation on the original data by introducing dropout regularization. Then, we perform global feature aggregation on two latent representations obtained from the encoders with non-shared parameters. To further alleviate the representation conflict problem in traditional contrastive learning, we propose a distance-guided adaptive negatives contrastive learning strategy, which dynamically adjusts the contribution of negative sample pairs through a neighborhood-aware weight matrix. In addition, our method enhances intra-cluster compactness while maximizing inter-cluster separation through iterative centroids refinement process guided by pseudo-labels. Finally, the optimal transport(OT)-based cross-view alignment explicitly minimizes transport costs between semantically related instances and target clusters, effectively enforcing distribution alignment across views. We evaluate our model on 15 publicly available datasets and the experimental results show our model outperforms 9 baseline methods in terms of various clustering metrics.
## Requirements
torch==1.10.1
pandas==2.3.0
numpy==1.26.4
scanpy: 1.10.3
scikit-learn: 1.4.2

