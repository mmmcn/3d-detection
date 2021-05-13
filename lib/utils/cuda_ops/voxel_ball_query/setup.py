from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
from setuptools import setup, Extension


if __name__ == '__main__':

    setup(
        name='voxel_ball_query',
        ext_modules=[
            CUDAExtension(
                name='voxel_ball_query_ext',
                sources=['src/voxel_ball_query.cpp',
                         'src/voxel_ball_query_cuda.cu']
            )],
        cmdclass={'build_ext': BuildExtension})
