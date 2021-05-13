from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup, find_packages

if __name__ == '__main__':

    setup(
        name='cuda_ops',
        ext_modules=[
            CUDAExtension(
                name='density_and_manhattan_weights_fps_ext',
                sources=['src/density_and_manhattan_weights_furthest_point_sample.cpp',
                         'src/density_and_manhattan_weights_furthest_point_sample_cuda.cu']),

        ]
    )