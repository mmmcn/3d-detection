from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup, find_packages

if __name__ == '__main__':

    setup(
        name='cuda_ops',
        ext_modules=[
            CUDAExtension(
                name='ball_query_ext',
                sources=['src/ball_query.cpp',
                         'src/ball_query_cuda.cu']),
        ],
        cmdclass={'build_ext': BuildExtension})
