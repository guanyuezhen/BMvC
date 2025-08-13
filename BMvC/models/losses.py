import torch
import torch.nn.functional as F


def contrastive_loss(z, mask):
    z_n = F.normalize(z, dim=1, p=2)
    s = torch.mm(z_n, z_n.t())
    pos_mask = mask.fill_diagonal_(0)
    neg_mask = 1 - mask.fill_diagonal_(1)
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

