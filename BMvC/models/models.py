import torch
import torch.nn as nn
from .modules import Encoder, Decoder, FusionModule, ClusteringModule


class MvAEModel(nn.Module):
    def __init__(self,
                 input_dims,
                 view_num,
                 latent_dim,
                 hid_dims=None,
                 cluster_num=None,
                 act=nn.ReLU()
                 ):
        super().__init__()
        if hid_dims is None:
            hid_dims = [192, 128]
        self.input_dims = input_dims
        self.view_num = view_num
        self.latent_dim = latent_dim
        self.hid_dims = hid_dims
        self.cluster_num = cluster_num
        self.act = act
        self.embedding_dim = int(1.5 * latent_dim)
        # encoder decoder
        self.view_specific_encoders = nn.ModuleList()
        self.view_specific_decoders = nn.ModuleList()
        for v in range(self.view_num):
            self.view_specific_encoders.append(Encoder(self.input_dims[v], self.hid_dims, self.latent_dim))
            self.view_specific_decoders.append(Decoder(self.input_dims[v], self.hid_dims, self.embedding_dim))
        # feature fusion layer
        self.fusion_layer = FusionModule(self.latent_dim, self.view_num, self.embedding_dim)
        # clustering layer
        self.cluster_layer = ClusteringModule(self.latent_dim, self.view_num, self.cluster_num)

    def forward(self, x_list, is_training=False):
        specific_z = []
        for v in range(self.view_num):
            z = self.view_specific_encoders[v](x_list[v])
            specific_z.append(z)

        joint_z = self.fusion_layer(specific_z)
        specific_y = self.cluster_layer(specific_z)

        recs = []
        for v in range(self.view_num):
            rec = self.view_specific_decoders[v](joint_z)
            recs.append(rec)

        if is_training:
            return recs, joint_z, specific_y
        else:
            return joint_z

