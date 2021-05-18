from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

if __name__ == '__main__':

    setup(
        name='cuda_ops',
        ext_modules=[
            CUDAExtension(
                name='group_points_ext',
                sources=['src/group_points.cpp',
                         'src/group_points_cuda.cu']),
        ],
        cmdclass={'build_ext': BuildExtension})
