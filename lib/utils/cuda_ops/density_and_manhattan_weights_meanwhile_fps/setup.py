from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
from setuptools import setup, Extension


if __name__ == '__main__':

    setup(
        name='density_and_manhattan_weights_meanwhile_farthest_point_sample',
        ext_modules=[
            CUDAExtension(
                name='density_and_manhattan_weights_meanwhile_fps_ext',
                sources=['src/density_and_manhattan_weights_meanwhile_furthest_point_sample.cpp',
                         'src/density_and_manhattan_weights_meanwhile_furthest_point_sample_cuda.cu']
            )],
        cmdclass={'build_ext': BuildExtension})
