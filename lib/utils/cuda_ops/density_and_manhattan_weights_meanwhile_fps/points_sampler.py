import torch
from mmcv.runner import force_fp32
from torch import nn as nn
from typing import List

from .density_and_manhattan_weights_meanwhile_furthest_point_sample import (
    density_and_manhattan_meanwhile_furthest_point_sample, furthest_point_sample_with_dist)
from mmdet3d.ops.furthest_point_sample.utils import calc_square_dist


def get_sampler_type(sampler_type):
    """Get the type and mode of points sampler.

    Args:
        sampler_type (str): The type of points sampler.
            The valid value are "D-FPS", "F-FPS", or "FS".

    Returns:
        class: Points sampler type.
    """
    if sampler_type == 'D-FPS':
        sampler = DFPS_Sampler
    elif sampler_type == 'F-FPS':
        sampler = FFPS_Sampler
    elif sampler_type == 'FS':
        sampler = FS_Sampler
    else:
        raise ValueError('Only "sampler_type" of "D-FPS", "F-FPS", or "FS"'
                         f' are supported, got {sampler_type}')

    return sampler


class PointsSamplerDensityAndManhattanMeanwhile(nn.Module):
    """
    the weight of density and manhattan is set inner currently.
    """

    def __init__(self,
                 num_point: List[int],
                 r: List[float] = [0.5],
                 manhattan_weights: List[float] = [[1.0, 1.0, 1.0]],
                 fps_mod_list: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1]):
        super(PointsSamplerDensityAndManhattanMeanwhile, self).__init__()
        assert len(num_point) == len(fps_mod_list) == len(fps_mod_list) ==\
            len(manhattan_weights) == len(r)
        self.num_point = num_point
        self.fps_sample_range_list = fps_sample_range_list
        self.samplers = nn.ModuleList()
        for fps_mod, cur_r, cur_mat_weights in zip(fps_mod_list, r, manhattan_weights):
            sampler = get_sampler_type(fps_mod)
            if fps_mod != 'F-FPS':
                self.samplers.append(sampler(cur_mat_weights, cur_r))
            else:
                self.samplers.append(sampler())
        self.fp16_enabled = False

    @force_fp32()
    def forward(self, points_xyz, features):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) Descriptors of the features.

        Return：
            Tensor: (B, npoint, sample_num) Indices of sampled points.
        """
        indices = []
        last_fps_end_index = 0

        for fps_sample_range, sampler, npoint in zip(
                self.fps_sample_range_list, self.samplers, self.num_point):
            assert fps_sample_range < points_xyz.shape[1]

            if fps_sample_range == -1:
                sample_points_xyz = points_xyz[:, last_fps_end_index:]
                sample_features = features[:, :, last_fps_end_index:]
            else:
                sample_points_xyz = \
                    points_xyz[:, last_fps_end_index:fps_sample_range]
                sample_features = \
                    features[:, :, last_fps_end_index:fps_sample_range]

            fps_idx = sampler(sample_points_xyz.contiguous(), sample_features,
                              npoint)

            indices.append(fps_idx + last_fps_end_index)
            last_fps_end_index += fps_sample_range
        indices = torch.cat(indices, dim=1)

        return indices


class DFPS_Sampler(nn.Module):
    """DFPS_Sampling.

    Using Euclidean distances of points for FPS.
    """

    def __init__(self, manhattan_weights, r, alpha=1.0):
        super(DFPS_Sampler, self).__init__()
        if not isinstance(manhattan_weights, list):
            manhattan_weights = list(manhattan_weights)
        self.manhattan_weights = manhattan_weights
        self.r = r
        self.alpha = alpha

    def forward(self, points, features, npoint):
        """Sampling points with D-FPS."""
        fps_idx = density_and_manhattan_meanwhile_furthest_point_sample(points.contiguous(), npoint, self.r,
                                                                        self.manhattan_weights, self.alpha)
        return fps_idx


class FFPS_Sampler(nn.Module):
    """FFPS_Sampler.

    Using feature distances for FPS.
    """

    def __init__(self):
        super(FFPS_Sampler, self).__init__()

    def forward(self, points, features, npoint):
        """Sampling points with F-FPS."""
        features_for_fps = torch.cat([points, features.transpose(1, 2)], dim=2)
        features_dist = calc_square_dist(
            features_for_fps, features_for_fps, norm=False)
        fps_idx = furthest_point_sample_with_dist(features_dist, npoint)
        return fps_idx


class FS_Sampler(nn.Module):
    """FS_Sampling.

    Using F-FPS and D-FPS simultaneously.
    """

    def __init__(self, manhattan_weights, r, alpha=1.0):
        super(FS_Sampler, self).__init__()
        if not isinstance(manhattan_weights, list):
            manhattan_weights = list(manhattan_weights)
        self.manhattan_weights = manhattan_weights
        self.r = r
        self.alpha = alpha

    def forward(self, points, features, npoint):
        """Sampling points with FS_Sampling."""
        features_for_fps = torch.cat([points, features.transpose(1, 2)], dim=2)
        features_dist = calc_square_dist(
            features_for_fps, features_for_fps, norm=False)
        fps_idx_ffps = furthest_point_sample_with_dist(features_dist, npoint)
        fps_idx_dfps = density_and_manhattan_meanwhile_furthest_point_sample(points.contiguous(), npoint, self.r,
                                                                             self.manhattan_weights, self.alpha)
        fps_idx = torch.cat([fps_idx_ffps, fps_idx_dfps], dim=1)
        return fps_idx