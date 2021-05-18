from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

if __name__ == '__main__':

    setup(
        name='cuda_ops',
        ext_modules=[
            CUDAExtension(
                name='roiaware_pool3d_ext',
                sources=['src/points_in_boxes_cpu.cpp',
                         'src/points_in_boxes_cuda.cu',
                         'src/roiaware_pool3d.cpp',
                         'src/roiaware_pool3d_kernel.cu']),
        ],
        cmdclass={'build_ext': BuildExtension})
