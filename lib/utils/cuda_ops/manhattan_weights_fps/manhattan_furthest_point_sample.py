from typing import List

import torch
from torch.autograd import Function

from . import manhattan_weights_fps_ext


class MatWeightsFurthestPointSampling(Function):
    """Furthest Point Sampling.

    Uses iterative furthest point sampling to select a set of features
    whose corresponding points have the furthest distance.
    """

    @staticmethod
    def forward(ctx, points_xyz: torch.Tensor,
                num_points: int, weights: List[float]) -> torch.Tensor:

        assert points_xyz.is_contiguous()

        device = points_xyz.device
        weights = torch.FloatTensor(weights).to(device).contiguous()

        B, N, _ = points_xyz.size()
        output = torch.cuda.IntTensor(B, num_points)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        manhattan_weights_fps_ext.manhattan_weights_furthest_point_sampling_wrapper(
            B, N, num_points, points_xyz, weights, temp, output)
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

        manhattan_weights_fps_ext.furthest_point_sampling_with_dist_wrapper(
            B, N, num_points, points_dist, temp, output)
        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


mat_weights_furthest_point_sample = MatWeightsFurthestPointSampling.apply
furthest_point_sample_with_dist = FurthestPointSamplingWithDist.apply
