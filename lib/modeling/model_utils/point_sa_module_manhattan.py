import torch
from mmcv.cnn import ConvModule
from torch import nn as nn
from typing import List
from torch.nn import functional as F
from mmdet3d.ops import GroupAll, QueryAndGroup, gather_points
from mmdet3d.ops.pointnet_modules.registry import SA_MODULES
from lib.utils.cuda_ops.density_and_manhattan_weights_meanwhile_fps import PointsSamplerDensityAndManhattanMeanwhile


@SA_MODULES.register_module()
class PointSAModuleMSGWithManhattanTest(nn.Module):

    def __init__(self,
                 num_point: int,
                 radii: List[float],
                 sample_nums: List[int],
                 mlp_channels: List[List[int]],
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1],
                 manhattan_weights: List[float] = [[1.0, 1.0, 1.0]],
                 density_fps_r: List[float] = [0.5],
                 dilated_group: bool = False,
                 norm_cfg: dict = dict(type='BN2d'),
                 use_xyz: bool = True,
                 pool_mod='max',
                 normalize_xyz: bool = False,
                 bias='auto'):
        super(PointSAModuleMSGWithManhattanTest, self).__init__()
        if isinstance(num_point, int):
            self.num_point = [num_point]
        elif isinstance(num_point, list) or isinstance(num_point, tuple):
            self.num_point = num_point
        else:
            raise NotImplementedError('Error type of num_point!')

        self.use_xyz = use_xyz
        self.points_sampler = PointsSamplerDensityAndManhattanMeanwhile(
            num_point=self.num_point,
            r=density_fps_r,
            manhattan_weights=manhattan_weights,
            fps_mod_list=fps_mod,
            fps_sample_range_list=fps_sample_range_list)
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i in range(len(radii)):
            radius = radii[i]
            sample_num = sample_nums[i]
            if num_point is not None:
                if dilated_group and i != 0:
                    min_radius = radii[i - 1]
                else:
                    min_radius = 0
                grouper = QueryAndGroup(
                    radius,
                    sample_num,
                    min_radius=min_radius,
                    use_xyz=use_xyz,
                    normalize_xyz=normalize_xyz)
            else:
                grouper = GroupAll(use_xyz)
            self.groupers.append(grouper)

            mlp_spec = mlp_channels[i]
            if use_xyz:
                mlp_spec[0] += 3

            mlp = nn.Sequential()
            for i in range(len(mlp_spec) - 1):
                mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        mlp_spec[i],
                        mlp_spec[i + 1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                        bias=bias))
            self.mlps.append(mlp)

    def forward(self, points_xyz, features = None,
                indices = None, target_xyz = None):
        new_features_list = []
        xyz_flipped = points_xyz.transpose(1, 2).contiguous()
        if indices is not None:
            assert (indices.shape[1] == self.num_point[0])
            new_xyz = gather_points(xyz_flipped, indices).transpose(
                1, 2).contiguous() if self.num_point is not None else None
        elif target_xyz is not None:
            new_xyz = target_xyz.contiguous()
        else:
            indices = self.points_sampler(points_xyz, features)
            new_xyz = gather_points(xyz_flipped, indices).transpose(
                1, 2).contiguous() if self.num_point is not None else None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](points_xyz, new_xyz, features)

            new_features = self.mlps[i](new_features)

            # only use max pooling
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)])
            new_features = new_features.squeeze(-1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1), indices


@SA_MODULES.register_module()
class PointSAModuleTest(PointSAModuleMSGWithManhattanTest):
    def __init__(self,
                 num_point: int,
                 mlp_channels: List[int],
                 radii: float = None,
                 sample_nums: int = None,
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1],
                 manhattan_weights: List[float] = [[1.0, 1.0, 1.0]],
                 density_fps_r: List[float] = [0.5],
                 dilated_group: bool = False,
                 norm_cfg: dict = dict(type='BN2d'),
                 use_xyz: bool = True,
                 pool_mod='max',
                 normalize_xyz: bool = False,
                 bias='auto'):
        super(PointSAModuleTest, self).__init__(
            mlp_channels=[mlp_channels],
            num_point=num_point,
            radii=[radii],
            sample_nums=[sample_nums],
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            manhattan_weights=manhattan_weights,
            density_fps_r=density_fps_r,
            dilated_group=dilated_group,
            norm_cfg=norm_cfg,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            normalize_xyz=normalize_xyz,
            bias=bias)
