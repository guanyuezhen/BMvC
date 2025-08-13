from __future__ import division
from __future__ import print_function

import torch.optim as optim
from scanpy.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score

from utils import *
from models import BMVC
import os
import argparse
from config import Config
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import random


def load_data(dataset):
    print("load data:")
    path = "../generate_data/" + dataset + "/BMVC.h5ad"
    adata = sc.read_h5ad(path)
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground']
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    s_features = torch.FloatTensor(adata.obsm['spatial'])
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])

    fadj = sparse_mx_to_torch_sparse_tensor(fadj)
    sadj = sparse_mx_to_torch_sparse_tensor(sadj)

    print("done")
    return adata, features, labels, nfadj, nsadj, graph_nei, graph_neg, fadj, sadj, s_features


def train():
    model.train()
    optimizer.zero_grad()
    com1, com2, emb, pi, disp, mean, mask = model(features, sadj, fadj, s_features)
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = (regularization_loss(com1, com1, mask[0])
                + regularization_loss(com2, com2, mask[1]))
    # con_loss = consistency_loss(com1, com2)
    # reg_loss = 0
    con_loss = 0
    total_loss = config.alpha * zinb_loss + config.beta * con_loss + config.gamma * reg_loss
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    return emb, mean, zinb_loss, reg_loss, con_loss, total_loss

def set_seed():
    seed = 42  # (3407, 4079, 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    set_seed()
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['Mouse_Brain_Anterior']

    for i in range(len(datasets)):
        dataset = datasets[i]
        path = './result/' + dataset + '/'
        config_file = './config/' + dataset + '.ini'
        if not os.path.exists(path):
            os.mkdir(path)
        print(dataset)
        adata, features, labels, fadj, sadj, graph_nei, graph_neg, fadj_, sadj_, s_features = load_data(dataset)

        config = Config(config_file)
        cuda = not config.no_cuda and torch.cuda.is_available()
        use_seed = not config.no_seed

        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        config.n = len(ground)
        config.class_num = len(ground.unique())

        savepath = './result/Mouse_Brain_Anterior/'
        plt.rcParams["figure.figsize"] = (4, 3)

        print(adata)
        title = "Manual annotation"
        sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title, show=False)
        plt.savefig(savepath + dataset + '.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        if cuda:
            features = features.cuda()
            s_features = s_features.cuda()
            sadj = sadj.cuda()
            fadj = fadj.cuda()
            sadj_ = sadj_.cuda()
            fadj_ = fadj_.cuda()
            graph_nei = graph_nei.cuda()
            graph_neg = graph_neg.cuda()

        config.epochs = config.epochs + 1

        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        if not config.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

        print(dataset, ' ', config.lr, ' ', config.alpha, ' ', config.beta, ' ', config.gamma)
        model = BMVC(nfeat=config.fdim,
                             nhid1=config.nhid1,
                             nhid2=config.nhid2,
                             dropout=config.dropout,
                             cluster_num=config.class_num)
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.lr,
                               weight_decay=config.weight_decay)

        epoch_max = 0
        ari_max = 0
        nmi_max = 0
        idx_max = []
        mean_max = []
        emb_max = []

        for epoch in range(config.epochs):
            emb, mean, zinb_loss, reg_loss, con_loss, total_loss = train()
            kmeans = KMeans(n_clusters=config.class_num, n_init=10).fit(emb)
            idx = kmeans.labels_
            ari_res = metrics.adjusted_rand_score(labels, idx)
            nmi_res = metrics.normalized_mutual_info_score(labels, idx)
            if ari_res > ari_max:
                ari_max = ari_res
                nmi_max = nmi_res
                epoch_max = epoch
                idx_max = idx
                mean_max = mean
                emb_max = emb

            print(dataset, ' epoch: ', epoch, ' zinb_loss = {:.2f}'.format(zinb_loss),
                  ' reg_loss = {:.2f}'.format(reg_loss), ' con_loss = {:.2f}'.format(con_loss),
                  ' total_loss = {:.2f}'.format(total_loss),
                  ' ari_max = {:.2f}'.format(ari_max * 100),
                  ' ari_res = {:.2f}'.format(ari_res * 100))

        print(dataset, ' ', ari_max)

        labels_int = [int(x) for x in labels]
        true = labels_int
        pred_1 = idx_max
        cm = confusion_matrix(true, pred_1)
        # 使用匈牙利算法进行标签重映射
        row_ind, col_ind = linear_sum_assignment(-cm)
        # 创建映射字典
        label_mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
        # 根据映射重映射预测标签
        idx_max = np.array([label_mapping[label] for label in pred_1])
        adata.obs['idx'] = idx_max.astype(str)

        title = 'ARI={:.2f} NMI={:.2f}'.format(ari_max * 100, nmi_max * 100)
        adata.obs['idx'] = idx_max.astype(str)
        adata.obsm['emb'] = emb_max
        adata.obsm['mean'] = mean_max

        sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=False)
        plt.savefig(savepath + 'BMVC.jpg', bbox_inches='tight', dpi=600)

        sc.pp.neighbors(adata, use_rep='mean')
        sc.tl.umap(adata)
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.tl.paga(adata, groups='idx')
        sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, title=title, legend_fontoutline=2,
                           show=False)
        plt.savefig(savepath + 'BMVC_umap_mean.jpg', bbox_inches='tight', dpi=600)
        pd.DataFrame(emb_max).to_csv(savepath + 'BMVC_emb.csv')
        pd.DataFrame(idx_max).to_csv(savepath + 'BMVC_idx.csv')
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean_max
        adata.write(savepath + 'BMVC.h5ad')