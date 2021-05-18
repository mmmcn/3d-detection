from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

if __name__ == '__main__':

    setup(
        name='cuda_ops',
        ext_modules=[
            CUDAExtension(
                name='interpolate_ext',
                sources=['src/interpolate.cpp',
                         'src/three_interpolate_cuda.cu',
                         'src/three_nn_cuda.cu']),
        ],
        cmdclass={'build_ext': BuildExtension})
