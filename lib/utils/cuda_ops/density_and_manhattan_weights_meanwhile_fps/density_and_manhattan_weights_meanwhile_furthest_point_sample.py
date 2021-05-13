from typing import List

import torch
from torch.autograd import Function
from . import density_and_manhattan_weights_meanwhile_fps_ext


class DensityAndManhattanWeightsMeanwhileFurthestPointSampling(Function):
    """Use Density and Manhattan weights Furthest Point Sampling simultaneously.
    """

    @staticmethod
    def forward(ctx, points_xyz: torch.Tensor,
                num_points: int, r: float,
                manhattan_weights: List[float], alpha: float,) -> torch.Tensor:
        """

        Args:
            ctx:
            points_xyz (B, N, 3): raw points to be down sampled.
            num_points: sampling nums.
            r: the radius used to calculate density weights.
            manhattan_weights:
            alpha:
        """

        assert points_xyz.is_contiguous()

        device = points_xyz.device

        B, N, _ = points_xyz.size()
        density_weights = torch.cuda.FloatTensor(B, N).contiguous()
        density_and_manhattan_weights_meanwhile_fps_ext.num_points_within_r_wrapper(B, N, r,
                                                                                    points_xyz,
                                                                                    density_weights)
        _max = density_weights.max(1)[0].unsqueeze(-1)
        _min = density_weights.min(1)[0].unsqueeze(-1)
        density_weights = 1 - torch.div(density_weights - _min, _max - _min)

        manhattan_weights = torch.FloatTensor(manhattan_weights).to(device).contiguous()

        output = torch.cuda.IntTensor(B, num_points)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        density_and_manhattan_weights_meanwhile_fps_ext.\
            density_and_manhattan_weights_meanwhile_furthest_point_sampling_wrapper(B, N, num_points, alpha, points_xyz,
                                                                                    manhattan_weights, density_weights,
                                                                                    temp, output)
        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


class FurthestPointSamplingWithDist(Function):
    """Furthest Point Sampling With Distance.

    Uses iterative furthest point sampling to select a set of features whose
    corresponding points have the furthest distance.
    """

    @staticmethod
    def forward(ctx, points_dist: torch.Tensor,
                num_points: int) -> torch.Tensor:
        """forward.

        Args:
            points_dist (Tensor): (B, N, N) Distance between each point pair.
            num_points (int): Number of points in the sampled set.

        Returns:
             Tensor: (B, num_points) indices of the sampled points.
        """
        assert points_dist.is_contiguous()

        B, N, _ = points_dist.size()
        output = points_dist.new_zeros([B, num_points], dtype=torch.int32)
        temp = points_dist.new_zeros([B, N]).fill_(1e10)

        density_and_manhattan_weights_meanwhile_fps_ext.furthest_point_sampling_with_dist_wrapper(
            B, N, num_points, points_dist, temp, output)
        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


density_and_manhattan_meanwhile_furthest_point_sample = DensityAndManhattanWeightsMeanwhileFurthestPointSampling.apply
furthest_point_sample_with_dist = FurthestPointSamplingWithDist.apply



