from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
from setuptools import setup, Extension


if __name__ == '__main__':

    setup(
        name='density_weights_farthest_point_sample',
        ext_modules=[
            CUDAExtension(
                name='density_weights_fps_ext',
                sources=['src/density_weights_farthest_point_sample.cpp',
                         'src/density_weights_farthest_point_sample_cuda.cu']
            )],
        cmdclass={'build_ext': BuildExtension})
