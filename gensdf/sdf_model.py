#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F

from einops import reduce

from .encoder.conv_pointnet import ConvPointnet
from .decoder.sdf_decoder import SdfDecoder


class SdfModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        model_config = self.config["SdfModelSpecs"]
        self.hidden_dim = model_config["hidden_dim"]
        self.latent_dim = model_config["latent_dim"]
        self.skip_connection = model_config["skip_connection"]
        self.tanh_act = model_config["tanh_act"]
        self.pn_hidden = model_config["pn_hidden_dim"]

        self.pointnet = ConvPointnet(c_dim=self.latent_dim,
                                     hidden_dim=self.pn_hidden,
                                     plane_resolution=64)

        self.model = SdfDecoder(latent_size=self.latent_dim,
                                hidden_dim=self.hidden_dim,
                                skip_connection=self.skip_connection,
                                tanh_act=self.tanh_act)

    # def training_step(self, x, idx):

    #     xyz = x['xyz'] # (B, 16000, 3)
    #     gt = x['gt_sdf'] # (B, 16000)
    #     pc = x['point_cloud'] # (B, 1024, 3)

    #     shape_features = self.pointnet(pc, xyz)

    #     pred_sdf = self.model(xyz, shape_features)

    #     sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction = 'none')
    #     sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

    #     return sdf_loss



    def forward(self, pc, xyz):
        shape_features = self.pointnet(pc, xyz)

        return self.model(xyz, shape_features).squeeze()

    def forward_with_plane_features(self, plane_features, xyz):
        '''
        plane_features: B, D*3, res, res (e.g. B, 768, 64, 64)
        xyz: B, N, 3
        '''
        point_features = self.pointnet.forward_with_plane_features(plane_features, xyz) # point_features: B, N, D
        pred_sdf = self.model( torch.cat((xyz, point_features),dim=-1) )
        return pred_sdf # [B, num_points]
