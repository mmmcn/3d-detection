import torch
from torch.autograd import Function

from . import voxel_fps_ext


class VoxelFurthestPointSampling(Function):

    @staticmethod
    def forward(ctx, points_xyz: torch.Tensor,
                num_points: int) -> torch.Tensor:

        assert points_xyz.is_contiguous()
        # assert weights.is_contiguous()

        B, N, _ = points_xyz.size()
        weights = torch.cuda.FloatTensor(3).fill_(1.0)
        output = torch.cuda.IntTensor(B, num_points)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        voxel_fps_ext.voxel_furthest_point_sampling_wrapper(
            B, N, num_points, points_xyz, weights, temp, output)
        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


voxel_furthest_point_sample = VoxelFurthestPointSampling.apply
