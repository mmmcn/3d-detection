from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

if __name__ == '__main__':

    setup(
        name='cuda_ops',
        ext_modules=[
            CUDAExtension(
                name='furthest_point_sample_ext',
                sources=['src/furthest_point_sample.cpp',
                         'src/furthest_point_sample_cuda.cu']),
        ],
        cmdclass={'build_ext': BuildExtension})
