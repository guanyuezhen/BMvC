import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        return x


class decoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]

class ClusteringModule(nn.Module):
    def __init__(self, latent_dim, cluster_num, view_num):
        super(ClusteringModule, self).__init__()
        self.latent_dim = latent_dim
        self.cluster_num = cluster_num
        self.view_num = view_num
        self.clustering_layers = nn.ModuleList()
        for v in range(self.view_num):
            self.clustering_layers.append(nn.Linear(self.latent_dim, self.cluster_num))

    @staticmethod
    def orthogonal_transformation(y):
        indicators, _ = torch.linalg.qr(y, mode='reduced')
        return indicators

    def forward(self, z_list):
        y_list = []
        for v in range(self.view_num):
            y = self.clustering_layers[v](z_list[v])
            y = self.orthogonal_transformation(y)
            y_list.append(y)

        return y_list

def contrastive_loss(z_v1, z_v2, mask_v1_v2):
    z_v1 = F.normalize(z_v1, dim=1, p=2)
    z_v2 = F.normalize(z_v2, dim=1, p=2)
    s = torch.mm(z_v1, z_v2.t())
    pos_mask = mask_v1_v2.fill_diagonal_(0)
    neg_mask = 1 - mask_v1_v2.fill_diagonal_(1)
    pos_loss = torch.sum((pos_mask * (1 - s)).pow(2)) / (torch.sum(pos_mask) + 1e-4)
    neg_loss = torch.sum((neg_mask * s).pow(2)) / (torch.sum(neg_mask) + 1e-4)
    loss_con = pos_loss + neg_loss
    return loss_con


def knn_graph(data, k=25): # 25 in previous settings
    N = data.shape[0]
    distances = torch.cdist(data, data) ** 2
    idx = torch.argsort(distances, dim=1)[:, :k + 1]  # Shape (N, k + 1)
    neighbors_idx = idx[:, 1:k + 1]  # Exclude the first column (self)
    d = distances[torch.arange(N).unsqueeze(1), neighbors_idx]  # Shape (N, k)
    adjacency_matrix = torch.zeros((N, N), dtype=data.dtype, device=data.device)
    eps = 1e-8
    d_k_minus_1 = d[:, -1]  # k-th nearest distance (last in the row)
    sum_d = torch.sum(d, dim=1)  # Sum of distances for each row
    adjacency_matrix[torch.arange(N).unsqueeze(1), neighbors_idx] \
        = (d_k_minus_1.unsqueeze(1) - d) / (k * d_k_minus_1.unsqueeze(1) - sum_d.unsqueeze(1) + eps)
    adjacency_matrix.fill_diagonal_(1)
    adjacency_matrix = 0.5 * (adjacency_matrix + adjacency_matrix.t())
    return adjacency_matrix


class BMVC(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout, cluster_num):
        super(BMVC, self).__init__()
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        nemb = int(1.5 * nhid2)
        self.fusion_layer = nn.Linear(nhid2 * 2, nemb)
        self.clustering_module = ClusteringModule(nhid2, cluster_num, 2)
        self.ZINB = decoder(nfeat, nhid1, nemb)
        self.dropout = dropout

    def forward(self, x, sadj, fadj, s_x):
        emb1 = self.SGCN(x, sadj)  # Spatial_GCN
        emb2 = self.FGCN(x, fadj)  # Feature_GCN
        specific_z = [emb1, emb2]
        specific_y = self.clustering_module(specific_z)
        emb = self.fusion_layer(torch.cat(specific_z, dim=1))
        [pi, disp, mean] = self.ZINB(emb)
        mask = []
        mask_v = [knn_graph(s_x), knn_graph(x)]
        for v in range(2):
            mask_z_v = knn_graph(emb)
            mask.append(0.5 * mask_z_v + 0.5 * mask_v[v])

        return specific_y[0], specific_y[1], emb, pi, disp, mean, mask
