import numpy as np
from .point_cloud_ops import points_to_voxel_nusc


class VoxelGenerator:
    def __init__(self, voxel_generator_cfg):
        point_cloud_range = voxel_generator_cfg['point_cloud_range']  # (-50, 50, -4, 2, -50, 50)
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)

        voxel_size = voxel_generator_cfg['voxel_size']
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        max_num_points = int(voxel_generator_cfg['max_number_of_points_per_voxel'])
        max_voxels = int(voxel_generator_cfg['max_number_of_voxels'])

        self.max_cur_sample_num = voxel_generator_cfg['max_cur_sample_num']

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate_nusc(self, cur_sweep_points, other_sweep_points):
        return points_to_voxel_nusc(
            cur_sweep_points, other_sweep_points, self._voxel_size,
            self._point_cloud_range, self._max_num_points,
            self._max_voxels, self.max_cur_sample_num)

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size