import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, act=nn.ReLU()):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.act = act

        encoder_layers = []
        for i in range(len(self.hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(self.input_dim, self.hidden_dims[i]))
            else:
                encoder_layers.append(nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]))
            encoder_layers.append(self.act)
        encoder_layers.append(nn.Linear(self.hidden_dims[-1], self.latent_dim))
        encoder_layers.append(self.act)
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        z = self.encoder(x)

        return z


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, act=nn.ReLU()):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(reversed(hidden_dims))
        self.latent_dim = latent_dim
        self.act = act

        decoder_layers = []

        for i in range(len(self.hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(self.latent_dim, self.hidden_dims[i]))
            else:
                decoder_layers.append(nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]))
            decoder_layers.append(self.act)
        decoder_layers.append(nn.Linear(self.hidden_dims[-1], self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x_rec = self.decoder(x)

        return x_rec


class FusionModule(nn.Module):
    def __init__(self, latent_dim, view_num, embedding_dim, fusion_mode="concat"):
        super(FusionModule, self).__init__()
        self.latent_dim = latent_dim
        self.view_num = view_num
        self.embedding_dim = embedding_dim
        self.fusion_mode = fusion_mode
        if fusion_mode == "concat":
            self.embedding_layer = nn.Linear(latent_dim * view_num, embedding_dim)
        elif fusion_mode == "weighted_sum":
            self.weight_assignment = nn.Sequential(
                nn.Linear(latent_dim * view_num, view_num),
                nn.Softmax(dim=-1)
            )
            self.embedding_layer = nn.Linear(latent_dim, embedding_dim)
        else:
            self.embedding_layer = nn.Linear(latent_dim, embedding_dim)

    def forward(self, specific_z):
        if self.fusion_mode == "concat":
            fusion_z = torch.cat(specific_z, dim=1)
            joint_z = self.embedding_layer(fusion_z)
        elif self.fusion_mode == "weighted_sum":
            weights = self.weight_assignment(torch.cat(specific_z, dim=1))
            weights_chunk = torch.chunk(weights, self.view_num, dim=1)
            fusion_z = []
            for v in range(self.view_num):
                fusion_z.append(specific_z[v] * weights_chunk[v])
            fusion_z = sum(fusion_z)
            joint_z = self.embedding_layer(fusion_z)
        elif self.fusion_mode == "average_sum":
            fusion_z = sum(specific_z) / len(specific_z)
            joint_z = self.embedding_layer(fusion_z)
        else:
            pass

        return joint_z


class ClusteringModule(nn.Module):
    def __init__(self, latent_dim, view_num, cluster_num):
        super(ClusteringModule, self).__init__()
        self.latent_dim = latent_dim
        self.view_num = view_num
        self.cluster_num = cluster_num
        self.view_specific_clustering_layers = nn.ModuleList()
        for v in range(view_num):
            self.view_specific_clustering_layers.append(nn.Linear(latent_dim, cluster_num))
    def forward(self, specific_z):
        specific_y = []
        for v in range(self.view_num):
            y = self.view_specific_clustering_layers[v](specific_z[v])
            y, _ = torch.linalg.qr(y, mode='reduced')  # orthogonal transformation
            specific_y.append(y)

        return specific_y


