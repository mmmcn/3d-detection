from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
from setuptools import setup, Extension


if __name__ == '__main__':

    setup(
        name='voxel_farthest_point_sample',
        ext_modules=[
            CUDAExtension(
                name='voxel_fps_ext',
                sources=['src/voxel_farthest_point_sample.cpp',
                         'src/voxel_farthest_point_sample_cuda.cu']
            )],
        cmdclass={'build_ext': BuildExtension})
