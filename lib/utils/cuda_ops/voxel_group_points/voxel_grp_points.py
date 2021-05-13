import torch
from torch import nn
from torch.autograd import Function

from ..voxel_ball_query import voxel_ball_query
from mmdet3d.ops import grouping_operation


class VoxelQueryAndGroup(nn.Module):

    def __init__(self,
                 max_radius,
                 sample_num,
                 min_radius=0,
                 use_xyz=True,
                 normalize_xyz=False):
        super(VoxelQueryAndGroup, self).__init__()
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.sample_num = sample_num
        self.use_xyz = use_xyz
        self.normalize_xyz = normalize_xyz

    def forward(self, points_xyz, center_xyz, features=None):

        idx = voxel_ball_query(self.min_radius, self.max_radius,
                               self.sample_num, points_xyz, center_xyz)
        # print(f'voxel ball query no error...{idx.max()}')

        xyz_trans = points_xyz.transpose(1, 2).contiguous()

        grouped_xyz = grouping_operation(xyz_trans, idx)
        # print(f'group points no error...{grouped_xyz.max()}')
        grouped_xyz -= center_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.max_radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            # print(f'group features no error...{grouped_features.max()}')
            if self.use_xyz:
                # (B, C + 3, npoint, sample_num)
                new_features = torch.cat([grouped_xyz, grouped_features],
                                         dim=1)
            else:
                new_features = grouped_features
        else:
            assert (self.use_xyz
                    ), 'Cannot have not features and not use xyz as a feature!'
            new_features = grouped_xyz

        return new_features
