from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

if __name__ == '__main__':

    setup(
        name='cuda_ops',
        ext_modules=[
            CUDAExtension(
                name='iou3d_cuda',
                sources=['src/iou3d.cpp',
                         'src/iou3d_kernel.cu']),
        ],
        cmdclass={'build_ext': BuildExtension})
