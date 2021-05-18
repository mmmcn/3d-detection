import torch
from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet3d.ops import PointFPModule, build_sa_module
from mmdet.models import BACKBONES
# from mmdet3d.models.backbones.base_pointnet import BasePointNet
from .base_pointnet import BasePointNet


@BACKBONES.register_module()
class ManhattanPointNet2SASSGTest(BasePointNet):
    """
    the 'test' suffix is used because this class already
    registered in mmdet3d.
    """

    def __init__(self,
                 in_channels,
                 num_points=(2048, 1024, 512, 256),
                 radius=(0.2, 0.4, 0.8, 1.2),
                 num_samples=(64, 32, 16, 16),
                 sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                              (128, 128, 256)),
                 fp_channels=((256, 256), (256, 256)),
                 manhattan_weights=((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
                 density_fps_r=None,
                 norm_cfg=dict(type='BN2d'),
                 sa_cfg=dict(
                     type='PointSAModuleMSGWithManhattan',
                     use_xyz=True,
                     normalize_xyz=True)
                 ):
        super(ManhattanPointNet2SASSGTest, self).__init__()
        self.num_sa = len(sa_channels)
        self.num_fp = len(fp_channels)

        assert len(num_points) == len(radius) == len(num_samples) == len(
            sa_channels)
        assert len(sa_channels) >= len(fp_channels)

        self.SA_modules = nn.ModuleList()
        sa_in_channel = in_channels - 3
        skip_channel_list = [sa_in_channel]

        for sa_index in range(self.num_sa):
            cur_sa_mlps = list(sa_channels[sa_index])
            cur_sa_mlps = [sa_in_channel] + cur_sa_mlps
            sa_out_channel = cur_sa_mlps[-1]
            if isinstance(manhattan_weights[sa_index], tuple):
                cur_manhattan_weights = [list(manhattan_weights[sa_index])]
            else:
                cur_manhattan_weights = list([manhattan_weights[sa_index]])
            if isinstance(density_fps_r[sa_index], tuple):
                cur_density_fps_r = list(density_fps_r[sa_index])
            else:
                cur_density_fps_r = [density_fps_r[sa_index]]
            self.SA_modules.append(
                build_sa_module(
                    num_point=num_points[sa_index],
                    radii=radius[sa_index],
                    sample_nums=num_samples[sa_index],
                    mlp_channels=cur_sa_mlps,
                    manhattan_weights=cur_manhattan_weights,
                    density_fps_r=cur_density_fps_r,
                    norm_cfg=norm_cfg,
                    cfg=sa_cfg))
            skip_channel_list.append(sa_out_channel)
            sa_in_channel = sa_out_channel

        self.FP_modules = nn.ModuleList()

        fp_source_channel = skip_channel_list.pop()
        fp_target_channel = skip_channel_list.pop()
        for fp_index in range(len(fp_channels)):
            cur_fp_mlps = list(fp_channels[fp_index])
            cur_fp_mlps = [fp_source_channel + fp_target_channel] + cur_fp_mlps
            self.FP_modules.append(PointFPModule(mlp_channels=cur_fp_mlps))
            if fp_index != len(fp_channels) - 1:
                fp_source_channel = cur_fp_mlps[-1]
                fp_target_channel = skip_channel_list.pop()

    @auto_fp16(apply_to=('points',))
    def forward(self, points):
        xyz, features = self._split_point_feats(points)

        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
            batch, 1).long()

        sa_xyz = [xyz]
        sa_features = [features]
        sa_indices = [indices]

        for i in range(self.num_sa):
            cur_xyz, cur_features, cur_indices = self.SA_modules[i](
                sa_xyz[i], sa_features[i])
            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))

        fp_xyz = [sa_xyz[-1]]
        fp_features = [sa_features[-1]]
        fp_indices = [sa_indices[-1]]

        for i in range(self.num_fp):
            fp_features.append(self.FP_modules[i](
                sa_xyz[self.num_sa - i - 1], sa_xyz[self.num_sa - i],
                sa_features[self.num_sa - i - 1], fp_features[-1]))
            fp_xyz.append(sa_xyz[self.num_sa - i - 1])
            fp_indices.append(sa_indices[self.num_sa - i - 1])

        ret = dict(
            fp_xyz=fp_xyz, fp_features=fp_features, fp_indices=fp_indices)
        return ret
