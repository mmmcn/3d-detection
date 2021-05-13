import torch
from torch.autograd import Function

from . import density_weights_fps_ext


class DensityWeightsFurthestPointSampling(Function):
    """Furthest Point Sampling.

    Uses iterative furthest point sampling to select a set of features
    whose corresponding points have the furthest distance.
    """

    @staticmethod
    def forward(ctx, points_xyz: torch.Tensor,
                num_points: int, r: float) -> torch.Tensor:
        assert points_xyz.is_contiguous()

        B, N, _ = points_xyz.size()
        weights = torch.cuda.FloatTensor(B, N).contiguous()
        density_weights_fps_ext.num_points_within_r_wrapper(B, N, r, points_xyz, weights)

        _max = weights.max(1)[0].unsqueeze(-1)
        _min = weights.min(1)[0].unsqueeze(-1)
        weights = 1 - torch.div(weights - _min, _max - _min)

        output = torch.cuda.IntTensor(B, num_points)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        density_weights_fps_ext.density_weights_furthest_point_sampling_wrapper(
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

        density_weights_fps_ext.furthest_point_sampling_with_dist_wrapper(
            B, N, num_points, points_dist, temp, output)
        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


density_weights_furthest_point_sample = DensityWeightsFurthestPointSampling.apply
furthest_point_sample_with_dist = FurthestPointSamplingWithDist.apply
